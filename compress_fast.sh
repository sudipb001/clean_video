#!/bin/bash

# ================================
# Fast MP4 Compression Script
# ================================

INPUT="$1"
OUTPUT="$2"

# Default output name
if [ -z "$OUTPUT" ]; then
  BASENAME=$(basename "$INPUT")
  OUTPUT="compressed_$BASENAME"
fi

# Check input
if [ ! -f "$INPUT" ]; then
  echo "Input file not found!"
  exit 1
fi

echo "======================================"
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "======================================"

# ================================
# FAST COMPRESSION (H.264)
# ================================
# Key speed optimizations:
# - preset=veryfast (huge speed gain)
# - CRF 23 (balanced quality)
# - multi-threading auto
# - no progress parsing overhead

ffmpeg -y -i "$INPUT" \
  -c:v libx264 \
  -preset veryfast \
  -crf 23 \
  -movflags +faststart \
  -c:a aac \
  -b:a 128k \
  "$OUTPUT"

echo "Compression completed."