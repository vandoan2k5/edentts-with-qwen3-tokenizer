#!/bin/bash
echo "Downloading LJSpeech dataset..."
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xf LJSpeech-1.1.tar.bz2
rm LJSpeech-1.1.tar.bz2
echo "LJSpeech dataset downloaded and extracted."