#!/bin/bash
set -e

stem="${1%.*}"
ext="${1##*.}"

if [ -z "$1" ]; then
    echo "Error: no input file provided" >&2
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Error: file '$1' does not exist" >&2
    exit 1
fi

if [ "$ext" != "wav" ]; then
    echo "Error: invalid extension '$ext'" >&2
    exit 1
fi

out="${stem}_tmp.wav"


# convert to mono-channel, 16-bit PCM, 48kHz
if ! ffmpeg -i "$1" -f wav -acodec pcm_s16le -ac 1 -ar 48000 -y "$out"; then
    echo "Error: ffmpeg conversion failed for '$1'" >&2
    exit 1
fi

# create backup
mv "$1" "${1}.bak"

# modify "in-place"
mv "$out" "$1"