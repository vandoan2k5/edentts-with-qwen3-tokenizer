"""
Inference for CifTTSModel.

Loads a trained checkpoint, takes raw text, runs the acoustic model in
inference mode (no target tokens -> decoder auto-regressively samples each
RVQ layer), decodes the resulting 16-layer codebook tokens to a waveform via
the Qwen3 12Hz audio tokenizer, and writes:

    <out_dir>/tokens.npy                   [T_aud, 16] int64   (audio codebook IDs)
    <out_dir>/alignment.npy                [T_txt, T_aud]       (hard 1-1 CIF firing alignment)
    <out_dir>/alignment_soft.npy           [T_txt, T_aud]       (soft contributions matrix)
    <out_dir>/alignment_stretched.npy      [T_txt, T_aud]       (each text token stretched to its duration)
    <out_dir>/durations.npy                [T_txt]              (integer per-text-token durations)
    <out_dir>/text_audio_alignment.txt                          (human-readable alignment table)
    <out_dir>/alignment.png                                     (heatmap of stretched alignment)
    <out_dir>/tokens.png                                        (16 codebook layers x T_aud)
    <out_dir>/output.wav                                        (decoded waveform, if Qwen3 is available)

Usage
-----
    python inference.py -t "Hello world."
    python inference.py -t "Hello." --checkpoint checkpoints/ljs/cif_tts/latest_weights.pyt
    python inference.py -t "Hello." --temperature 0.7 --top_k 20 -o outputs/demo
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Use a non-interactive matplotlib backend so this works on headless boxes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hparams import hparams as hp
from models.cifTTS import CifTTSModel
from models.components import get_mask_from_lengths
from utils.log_util import get_logger

log = get_logger(__name__)

# Qwen3 12 Hz tokenizer -> 12 audio frames per second of speech.
AUDIO_FRAME_RATE_HZ = 12.0


# ---------------------------------------------------------------------------
# 1. Self-contained text -> phone-id sequence.
#
#    We re-implement the bits we need so the script does not depend on
#    `text.en_util` (which requires g2p_en + unidecode, neither of which is
#    guaranteed to be installed in an inference-only environment).  The
#    training data was tokenized with `token_type="char"` + `english_cleaners`
#    so this matches the on-disk phone files exactly.
# ---------------------------------------------------------------------------

_PUNCTUATION = "!'(),.:;? "
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_SPECIAL = "-"
_PAD = "_"
_SILENCES = ["@sp", "@spn", "@sil"]

_SYMBOLS: List[str] = (
    [_PAD]
    + list(_SPECIAL)
    + list(_PUNCTUATION)
    + list(_LETTERS)
    + _SILENCES
)
_SYMBOL_TO_ID: Dict[str, int] = {s: i for i, s in enumerate(_SYMBOLS)}
_ID_TO_SYMBOL: Dict[int, str] = {i: s for i, s in enumerate(_SYMBOLS)}

_WHITESPACE_RE = re.compile(r"\s+")


def _load_symbol_table() -> Dict[str, int]:
    """Prefer the on-disk table saved by `text.en_util` if present, so the
    inference symbol IDs exactly match those used at training time."""
    table_path = Path("resource/symbol_to_id_en.json")
    if table_path.exists():
        with open(table_path, "r", encoding="utf-8") as f:
            table = json.load(f)
        return {s: i for s, i in table.items() if s in _SYMBOL_TO_ID or s.startswith("@")}
    return _SYMBOL_TO_ID


def _clean_text(text: str) -> str:
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def text_to_phone_ids(text: str) -> torch.Tensor:
    """Convert a string into a [1, T] LongTensor of phone IDs."""
    table = _load_symbol_table()
    cleaned = _clean_text(text)
    if not cleaned:
        raise ValueError("text is empty after cleaning")

    seq: List[int] = []
    for ch in cleaned:
        if ch in table:
            seq.append(table[ch])
        elif ch == " ":
            seq.append(table[" "])
        else:
            log.warning(f"character {ch!r} not in symbol table, dropping")

    # Bracket with start/end tokens (space and '.' in the LJSpeech alphabet).
    if not seq or seq[0] != table.get(" ", _SYMBOL_TO_ID[" "]):
        seq = [table.get(" ", _SYMBOL_TO_ID[" "])] + seq
    if seq[-1] != table.get(".", _SYMBOL_TO_ID["."]):
        seq = seq + [table.get(".", _SYMBOL_TO_ID["."])]

    return torch.tensor([seq], dtype=torch.long)


def id_to_char(phone_id: int) -> str:
    """Best-effort conversion of a phone id back to a printable character."""
    return _ID_TO_SYMBOL.get(int(phone_id), f"<{int(phone_id)}>")


# ---------------------------------------------------------------------------
# 2. Model construction + checkpoint loading.
# ---------------------------------------------------------------------------

def build_model(checkpoint_path: Optional[str], device: torch.device) -> CifTTSModel:
    cif_cfg = hp.get_cif_config()
    decoder_cfg = hp.get_decoder_config()
    text_cfg = {
        "encoder_layer": hp.text_encoder_layers,
        "encoder_head": 2,
        "encoder_hidden": hp.text_encoder_hidden,
        "encoder_dropout": hp.text_encoder_dropout,
        "vocab_size": hp.vocab_size,
    }
    model = CifTTSModel(text_cfg, cif_cfg, decoder_cfg).to(device)

    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        log.info(f"loading weights from {ckpt}")
        model.load(str(ckpt))

    model.eval()
    return model


# ---------------------------------------------------------------------------
# 3. Synthesis: text -> tokens + alignment.
# ---------------------------------------------------------------------------

@torch.no_grad()
def synthesize(
    model: CifTTSModel,
    phone_ids: torch.Tensor,
    device: torch.device,
    temperature: float = 0.8,
    top_k: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Run the model in inference mode.

    Returns
    -------
    tokens        : [1, T_aud, 16] LongTensor of sampled audio codebook IDs.
    hard_align    : [T_txt, T_aud] FloatTensor binary 1-1 alignment (firing positions only).
    soft_align    : [T_txt, T_aud] FloatTensor soft contributions matrix (per-position weights).
    T_txt         : number of text tokens.
    T_aud         : number of audio frames actually produced.
    """
    phone_ids = phone_ids.to(device)
    B, T_txt = phone_ids.shape
    phone_lengths = torch.tensor([T_txt], dtype=torch.long, device=device)

    # ---- Text encoder ----
    _, text_value = model.text_encoder(phone_ids, phone_lengths)

    encoder_padding_mask = get_mask_from_lengths(phone_lengths, max_len=T_txt)
    encoder_outputs = {
        "encoder_raw_out": text_value,
        "encoder_padding_mask": encoder_padding_mask,
    }

    # ---- CIF middleware (inference: target_lengths=None) ----
    cif_ret = model.cif(encoder_outputs, target_lengths=None)
    cif_out = cif_ret["cif_out"]
    cif_padding_mask = cif_ret["cif_out_padding_mask"]
    fired_marks = cif_ret["fired_marks"]                  # [1, T_txt]
    accumulated_weights = cif_ret["accumulated_weights"]  # [1, T_txt]
    contributions = cif_ret["contributions"]              # [1, T_txt, T_aud_ub]

    # ---- Decoder (sampling) ----
    decoder_out = model.decoder(
        text_features=cif_out,
        targets=None,            # sampling mode
        temperature=temperature,
        top_k=top_k,
    )                            # [1, T_aud, 16]

    T_aud = int(cif_padding_mask.sum(dim=1).item())

    # Hard 1-1 alignment (firing positions only). Each row i of fired_marks
    # tells us whether text position i produced an audio frame; the j-th 1 in
    # the sequence corresponds to the j-th audio frame. This is the canonical
    # "raw" CIF alignment.
    marks = fired_marks[0].cpu()
    hard_align = torch.zeros(T_txt, T_aud, dtype=torch.float32)
    audio_idx = 0
    for txt_idx in range(T_txt):
        if marks[txt_idx].item() == 1 and audio_idx < T_aud:
            hard_align[txt_idx, audio_idx] = 1.0
            audio_idx += 1
    if audio_idx < T_aud:
        for txt_idx in range(T_txt):
            if audio_idx >= T_aud:
                break
            if hard_align[txt_idx].sum() == 0 and accumulated_weights[0, txt_idx].item() > 0:
                hard_align[txt_idx, audio_idx] = 1.0
                audio_idx += 1

    # Soft alignment: contributions matrix sliced to T_aud.
    soft_align = contributions[0, :, :T_aud].cpu().float()

    return decoder_out, hard_align, soft_align, T_txt, T_aud


# ---------------------------------------------------------------------------
# 4. Alignment construction (per-text-token duration + stretched matrix).
# ---------------------------------------------------------------------------

def compute_durations(soft_align: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a soft [T_txt, T_aud] alignment to per-text-token integer
    durations that sum to T_aud.

    Uses largest-remainder rounding on `soft_align.sum(axis=1)` and then
    rebalances +1/-1 so the total exactly matches T_aud.  Text tokens with
    essentially zero soft weight (no contribution) get duration 0; those get
    "absorbed" into the neighbouring text tokens when the stretched matrix is
    built.
    """
    T_txt, T_aud = soft_align.shape
    raw = soft_align.sum(axis=1)                # fractional durations
    floors = np.floor(raw).astype(np.int64)
    fractions = raw - floors
    durations = floors.copy()
    residual = int(T_aud - int(durations.sum()))
    if residual > 0:
        # Add 1 to the positions with the largest fractions.
        order = np.argsort(-fractions)  # descending
        for k in range(residual):
            durations[order[k]] += 1
    elif residual < 0:
        # Subtract 1 from the positions with the smallest fractions that are
        # still > 0 (never go below 0).
        order = np.argsort(fractions)  # ascending
        k = 0
        while residual < 0 and k < T_txt:
            idx = order[k]
            if durations[idx] > 0:
                durations[idx] -= 1
                residual += 1
            k += 1
    return durations, fractions


def build_stretched_alignment(durations: np.ndarray, T_aud: int) -> np.ndarray:
    """Build a [T_txt, T_aud] binary matrix where row i has `durations[i]`
    consecutive ones, placed contiguously so the total coverage is exactly
    T_aud frames."""
    T_txt = len(durations)
    alignment = np.zeros((T_txt, T_aud), dtype=np.float32)
    cursor = 0
    for i, d in enumerate(durations):
        d = int(d)
        if d <= 0 or cursor >= T_aud:
            continue
        # Clamp to the remaining frames.
        end = min(cursor + d, T_aud)
        alignment[i, cursor:end] = 1.0
        cursor = end
    return alignment


# ---------------------------------------------------------------------------
# 5. Visualization + human-readable output.
# ---------------------------------------------------------------------------

def plot_alignment(
    alignment: np.ndarray,
    phone_ids: torch.Tensor,
    out_path: Path,
    text: str,
    title_suffix: str = "",
) -> None:
    """Save a heat-map of an alignment matrix."""
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(alignment, aspect="auto", origin="lower",
                   interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Audio frame")
    ax.set_ylabel("Text token index")
    ax.set_title(
        f"CIF alignment{(' ' + title_suffix) if title_suffix else ''} "
        f"({alignment.shape[0]} phones -> {alignment.shape[1]} frames)\n{text!r}"
    )
    if alignment.shape[0] <= 60:
        ax.set_yticks(range(alignment.shape[0]))
        ax.set_yticklabels(
            [f"{i}:{id_to_char(int(t))!r}" for i, t in enumerate(phone_ids[0].tolist())],
            fontsize=6,
        )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(text i -> audio j)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"saved alignment plot -> {out_path}")


def plot_token_grid(tokens: np.ndarray, out_path: Path) -> None:
    """Visualize the 16-layer codebook tokens as a 16 x T_aud heatmap."""
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(tokens.T, aspect="auto", origin="lower",
                   interpolation="nearest", cmap="tab20")
    ax.set_xlabel("Audio frame")
    ax.set_ylabel("RVQ layer")
    ax.set_title(f"Sampled audio tokens (shape {tokens.shape})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"saved token grid -> {out_path}")


def write_alignment_report(
    text: str,
    phone_ids: torch.Tensor,
    soft_align: np.ndarray,
    durations: np.ndarray,
    fractions: np.ndarray,
    T_aud: int,
    out_path: Path,
) -> None:
    """Write a human-readable text/audio alignment file.

    The file has three sections:
      1. Per-text-token table: which audio frames each phone covers, plus the
         soft (fractional) weight that phone contributes to each frame.
      2. Per-audio-frame table: which text positions contributed to each frame.
      3. The raw phone sequence, alignment matrix shapes and stretch summary.
    """
    T_txt = len(phone_ids[0])
    dur_s = T_aud / AUDIO_FRAME_RATE_HZ

    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("CifTTS text <-> audio alignment")
    lines.append("=" * 78)
    lines.append(f"Text         : {text!r}")
    lines.append(f"Phone IDs    : {phone_ids[0].tolist()}")
    lines.append(f"Phone chars  : {''.join(id_to_char(int(t)) for t in phone_ids[0])}")
    lines.append(f"Audio frames : {T_aud}  (frame rate {AUDIO_FRAME_RATE_HZ:g} Hz ~ {dur_s:.2f}s)")
    lines.append(f"Text tokens  : {T_txt}")
    lines.append(f"Soft weights sum per text token : {['%.3f'%v for v in soft_align.sum(axis=1).tolist()]}")
    lines.append("")

    # ---- Per-text-token table ----
    lines.append("-" * 78)
    lines.append("Per-text-token alignment (each text token stretched to its duration)")
    lines.append("-" * 78)
    header = (
        f"{'idx':>3} | {'phone':>5} | {'char':>4} | {'audio span':>11} | "
        f"{'dur':>3} | {'top frame':>9} | {'soft dur':>8} | contributions"
    )
    lines.append(header)
    lines.append("-" * len(header))

    # Build the cumulative audio ranges for the stretched alignment.
    starts = np.zeros(T_txt, dtype=np.int64)
    ends = np.zeros(T_txt, dtype=np.int64)
    cursor = 0
    for i, d in enumerate(durations):
        d = int(d)
        starts[i] = cursor
        ends[i] = cursor + d
        cursor = ends[i]

    for i in range(T_txt):
        phone_id = int(phone_ids[0, i])
        ch = id_to_char(phone_id)
        if ch.strip() == "":
            ch_repr = repr(ch)
        else:
            ch_repr = repr(ch)
        if durations[i] == 0:
            span = "(none)     "
        else:
            span = f"[{starts[i]},{ends[i]})"
        top_frame = int(np.argmax(soft_align[i])) if soft_align[i].sum() > 0 else -1
        if top_frame >= 0 and soft_align[i, top_frame] == 0:
            top_frame = -1
        top_repr = f"{top_frame}" if top_frame >= 0 else "-"
        soft_dur = soft_align[i].sum()
        contribs = ", ".join(
            f"j={j}:{soft_align[i, j]:.3f}"
            for j in range(T_aud)
            if soft_align[i, j] > 1e-3
        ) or "-"
        lines.append(
            f"{i:>3} | {phone_id:>5} | {ch_repr:>4} | {span:>11} | "
            f"{int(durations[i]):>3} | {top_repr:>9} | {soft_dur:>8.3f} | {contribs}"
        )
    lines.append("")

    # ---- Per-audio-frame table ----
    lines.append("-" * 78)
    lines.append("Per-audio-frame alignment (which text positions contribute)")
    lines.append("-" * 78)
    header2 = (
        f"{'frame':>5} | {'time(s)':>7} | {'sum':>5} | contributions"
    )
    lines.append(header2)
    lines.append("-" * len(header2))
    for j in range(T_aud):
        t = j / AUDIO_FRAME_RATE_HZ
        col = soft_align[:, j]
        s = col.sum()
        contribs = ", ".join(
            f"text {i}={col[i]:.3f}"
            for i in range(T_txt)
            if col[i] > 1e-3
        ) or "-"
        lines.append(f"{j:>5} | {t:>7.3f} | {s:>5.2f} | {contribs}")
    lines.append("")

    # ---- Stretched matrix as text ----
    lines.append("-" * 78)
    lines.append("Stretched alignment matrix (T_txt rows x T_aud cols, '#' = aligned)")
    lines.append("-" * 78)
    col_w = max(2, len(str(T_aud)))
    cell_w = col_w + 1  # one trailing space so single/double-digit cells line up
    header_row = "    " + "".join(f"{j:>{col_w}} " for j in range(T_aud))
    lines.append(header_row.rstrip())
    for i in range(T_txt):
        row = "".join(
            f"{'#':>{col_w}} " if (i < durations.shape[0] and starts[i] <= j < ends[i]) else f"{'.':>{col_w}} "
            for j in range(T_aud)
        )
        lines.append(f"{i:>3} {row.rstrip()}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"saved alignment report -> {out_path}")


# ---------------------------------------------------------------------------
# 6. Audio decoding (Qwen3 12Hz tokenizer).
# ---------------------------------------------------------------------------

def decode_tokens_to_wav(
    tokens: np.ndarray,
    device: torch.device,
) -> Optional[Tuple[np.ndarray, int]]:
    """Decode a [T_aud, 16] int64 token array to (wav, sample_rate).

    Returns None if the Qwen3 tokenizer is not installed.
    """
    try:
        from qwen_tts import Qwen3TTSTokenizer  # type: ignore
    except Exception as e:
        log.warning(f"qwen_tts not available, skipping wav decoding: {e}")
        return None

    try:
        qwen = Qwen3TTSTokenizer.from_pretrained(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            device_map=str(device),
        )
    except Exception as e:
        log.warning(f"could not load Qwen3 tokenizer ({e}); tokens.npy is still saved")
        return None

    # Qwen3's encode() returns [1, num_codebooks, T]; symmetrically decode()
    # expects the same layout. We saved [T_aud, 16] so transpose back.
    codes = torch.from_numpy(tokens).long().t().unsqueeze(0).to(device)  # [1, 16, T_aud]
    try:
        wavs = qwen.decode(codes)
        wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
        sr = getattr(qwen, "sample_rate", 24000) or 24000
        return wav, int(sr)
    except Exception as e:
        log.warning(f"Qwen3 decode failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 7. Entry point.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CifTTS inference")
    p.add_argument("-t", "--text", type=str, required=True,
                   help="text to synthesize")
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/ljs/cif_tts/latest_weights.pyt",
                   help="path to model weights (default: latest checkpoint)")
    p.add_argument("-o", "--output_dir", type=str, default="outputs/inference",
                   help="where to write tokens / wav / alignment")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--no-wav", action="store_true",
                   help="skip Qwen3 decoding (just write tokens + alignment)")
    p.add_argument("--cpu", action="store_true",
                   help="force CPU even if CUDA is available")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    log.info(f"device: {device}")
    log.info(f"text:   {args.text!r}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. model
    model = build_model(args.checkpoint, device=device)

    # 2. text -> phone ids
    phone_ids = text_to_phone_ids(args.text)
    log.info(f"phone ids: shape={tuple(phone_ids.shape)}  tokens={phone_ids[0].tolist()}")

    # 3. synthesize
    tokens, hard_align, soft_align, T_txt, T_aud = synthesize(
        model, phone_ids, device=device,
        temperature=args.temperature, top_k=args.top_k,
    )
    tokens_np = tokens[0].cpu().numpy().astype(np.int64)
    log.info(f"generated tokens: shape={tokens_np.shape}  (T_txt={T_txt} -> T_aud={T_aud})")

    # 4. save tokens
    np.save(out_dir / "tokens.npy", tokens_np)

    # 5. alignments
    hard_np = hard_align.cpu().numpy()
    soft_np = soft_align.cpu().numpy()
    np.save(out_dir / "alignment.npy", hard_np)
    np.save(out_dir / "alignment_soft.npy", soft_np)

    # 6. per-text-token durations + stretched alignment
    durations, fractions = compute_durations(soft_np)
    np.save(out_dir / "durations.npy", durations)
    stretched = build_stretched_alignment(durations, T_aud)
    np.save(out_dir / "alignment_stretched.npy", stretched)
    log.info(
        f"durations (sum={int(durations.sum())}, T_aud={T_aud}): "
        f"{durations.tolist()}"
    )

    # 7. plots
    plot_alignment(stretched, phone_ids, out_dir / "alignment.png",
                   args.text, title_suffix="stretched")
    plot_alignment(soft_np, phone_ids, out_dir / "alignment_soft.png",
                   args.text, title_suffix="soft (CIF contributions)")
    plot_token_grid(tokens_np, out_dir / "tokens.png")

    # 8. human-readable report
    write_alignment_report(
        text=args.text,
        phone_ids=phone_ids,
        soft_align=soft_np,
        durations=durations,
        fractions=fractions,
        T_aud=T_aud,
        out_path=out_dir / "text_audio_alignment.txt",
    )

    # 9. decode to wav (optional)
    if not args.no_wav:
        result = decode_tokens_to_wav(tokens_np, device)
        if result is not None:
            wav, sr = result
            try:
                import soundfile as sf
                sf.write(str(out_dir / "output.wav"), wav, sr)
                log.info(
                    f"saved wav -> {out_dir / 'output.wav'}  "
                    f"(sr={sr}, dur={len(wav) / sr:.2f}s)"
                )
            except Exception as e:
                log.warning(f"could not write wav: {e}")

    log.info(f"done. outputs in {out_dir}/")


if __name__ == "__main__":
    main()
