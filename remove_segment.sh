#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [options] <input.mp4> <remove_start> <remove_end>

Remove a segment from an mp4.

Positional:
  input.mp4           Input file
  remove_start        Start time to remove (HH:MM:SS, MM:SS or seconds)
  remove_end          End time to remove (HH:MM:SS, MM:SS or seconds)

Options:
  -o FILE, --output FILE    Output filename (default: output.mp4)
  --fast                    Fast mode: try lossless copy/remux (may produce black frames if not keyframe aligned)
  --crf N                   CRF for re-encode mode (default: 18)
  --preset NAME             x264 preset (default: veryfast)
  --audio-bitrate RATE      AAC audio bitrate (default: 192k)
  -y                        Overwrite output if exists
  -h                        Show this help

Examples:
  $0 input.mp4 00:16:22 00:21:10 -o output.mp4
  $0 --fast input.mp4 982 1270 -o out_fast.mp4
EOF
}

# parse time support HH:MM:SS, MM:SS, or seconds
time_to_seconds() {
  local t="$1"
  if [[ "$t" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf "%s" "$t"
    return
  fi
  IFS=":" read -r -a parts <<< "$t"
  local n=${#parts[@]}
  local sec=0
  if (( n == 2 )); then
    # MM:SS
    sec=$(awk -v m="${parts[0]}" -v s="${parts[1]}" 'BEGIN{print m*60 + s}')
  elif (( n == 3 )); then
    # HH:MM:SS
    sec=$(awk -v h="${parts[0]}" -v m="${parts[1]}" -v s="${parts[2]}" 'BEGIN{print h*3600 + m*60 + s}')
  else
    echo "ERR" >&2
    return 1
  fi
  printf "%s" "$sec"
}

# default options
OUTPUT="output.mp4"
FAST_MODE=0
CRF=18
PRESET="veryfast"
AUDIO_BITRATE="192k"
OVERWRITE=0

# parse args
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output) OUTPUT="$2"; shift 2;;
    --fast) FAST_MODE=1; shift;;
    --crf) CRF="$2"; shift 2;;
    --preset) PRESET="$2"; shift 2;;
    --audio-bitrate) AUDIO_BITRATE="$2"; shift 2;;
    -y) OVERWRITE=1; shift;;
    -h) usage; exit 0;;
    --) shift; break;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 2;;
    *) ARGS+=("$1"); shift;;
  esac
done

if (( ${#ARGS[@]} != 3 )); then
  usage
  exit 2
fi

INPUT="${ARGS[0]}"
REMOVE_START_RAW="${ARGS[1]}"
REMOVE_END_RAW="${ARGS[2]}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found in PATH. Install ffmpeg first." >&2
  exit 2
fi

if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: input file not found: $INPUT" >&2
  exit 2
fi

# convert times
REMOVE_START="$(time_to_seconds "$REMOVE_START_RAW")" || { echo "Bad start time"; exit 2; }
REMOVE_END="$(time_to_seconds "$REMOVE_END_RAW")" || { echo "Bad end time"; exit 2; }

# numeric compare (use awk because times can be floats)
if (( $(awk "BEGIN{print ($REMOVE_END <= $REMOVE_START)}") )); then
  echo "ERROR: remove_end must be greater than remove_start" >&2
  exit 2
fi

if [[ $OVERWRITE -eq 1 ]]; then
  FF_YFLAG="-y"
else
  FF_YFLAG="-n"
fi

TMP_PREFIX="$(mktemp -u remove_segment.XXXX)"
cleanup() {
  rm -f "${TMP_PREFIX}"* || true
}
trap cleanup EXIT

echo "Input: $INPUT"
echo "Removing: $REMOVE_START_RAW -> $REMOVE_END_RAW (seconds: ${REMOVE_START} -> ${REMOVE_END})"
echo "Output: $OUTPUT"
if [[ $FAST_MODE -eq 1 ]]; then
  echo "Mode: FAST (copy/remux)"
else
  echo "Mode: RE-ENCODE (safe)"
fi

if [[ $FAST_MODE -eq 1 ]]; then
  # Fast path: remux input, cut at nearest keyframes, concat
  REMUX="${TMP_PREFIX}_remux.mp4"
  echo "Remuxing to clean container: $REMUX"
  ffmpeg $FF_YFLAG -i "$INPUT" -c copy -map 0 -movflags +faststart "$REMUX"

  PART1="${TMP_PREFIX}_part1.mp4"
  PART2="${TMP_PREFIX}_part2.mp4"
  echo "Extracting part1 (0 -> $REMOVE_START) to $PART1 (keyframe-seek)"
  ffmpeg $FF_YFLAG -ss 0 -i "$REMUX" -t "$REMOVE_START" -c copy -avoid_negative_ts 1 -fflags +genpts "$PART1"

  echo "Extracting part2 ($REMOVE_END -> end) to $PART2 (keyframe-seek)"
  ffmpeg $FF_YFLAG -ss "$REMOVE_END" -i "$REMUX" -c copy -avoid_negative_ts 1 -fflags +genpts "$PART2"

  # concat
  FILES_LIST="${TMP_PREFIX}_files.txt"
  printf "file '%s'\nfile '%s'\n" "$PART1" "$PART2" > "$FILES_LIST"
  echo "Concatenating into $OUTPUT"
  ffmpeg $FF_YFLAG -f concat -safe 0 -i "$FILES_LIST" -c copy -fflags +genpts "$OUTPUT"
  echo "Done (fast mode). Verify output for micro-flash or sync issues."
  exit 0
else
  # Re-encode path using filter_complex (guaranteed clean)
  echo "Running re-encode remove (this will re-encode video/audio; may take time)"
  FILTER_COMPLEX="[0:v]trim=0:${REMOVE_START},setpts=PTS-STARTPTS[v0];\
[0:a]atrim=0:${REMOVE_START},asetpts=PTS-STARTPTS[a0];\
[0:v]trim=${REMOVE_END},setpts=PTS-STARTPTS[v1];\
[0:a]atrim=${REMOVE_END},asetpts=PTS-STARTPTS[a1];\
[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]"

  CMD=(ffmpeg $FF_YFLAG -i "$INPUT" -filter_complex "$FILTER_COMPLEX" -map "[outv]" -map "[outa]" \
    -c:v libx264 -crf "$CRF" -preset "$PRESET" -c:a aac -b:a "$AUDIO_BITRATE" -movflags +faststart "$OUTPUT")

  echo "${CMD[@]}"
  "${CMD[@]}"
  echo "Done (re-encode)."
  exit 0
fi
