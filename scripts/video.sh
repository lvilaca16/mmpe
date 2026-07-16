#!/bin/bash
set -e

n_frames=32

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

case "$ext" in
    mp4|mov|mkv|avi|webm) ;;
    *)
        echo "Error: invalid extension '${ext}'" >&2
        exit 1
        ;;
esac

out_dir="$stem"

if [ -d "$out_dir" ]; then
    echo "Error: directory '${out_dir}' already exists" >&2
    exit 1
fi

# get video duration in seconds
duration="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$1")"

if [ -z "$duration" ]; then
    echo "Error: could not determine duration for '$1'" >&2
    exit 1
fi

mkdir -p "$out_dir"

# fps needed to get exactly n_frames evenly spaced frames across the clip
fps="$(echo "$n_frames / $duration" | bc -l)"

if ! ffmpeg -i "$1" -vf "fps=${fps}" -frames:v "$n_frames" -y "${out_dir}/%04d.jpg"; then
    echo "Error: ffmpeg frame extraction failed for '$1'" >&2
    rmdir "$out_dir" 2>/dev/null
    exit 1
fi

echo "Extracted ${n_frames} frames to '${out_dir}'"