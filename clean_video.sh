#!/usr/bin/env bash

# Exit immediately if a pipeline returns a non-zero status
set -o pipefail

# --- Default Configuration ---
TARGET_LOUDNESS="-16.0"
NOISE_REDUCTION="0.25"
PRESERVE_VOICE="0.6"
APPLY_COMPRESSION=true
APPLY_DEESS=true
ENHANCE_EQ=true
INFO_ONLY=false
INSTALL_DEPS_FLAG=false
INPUT_FILE=""
OUTPUT_FILE=""

# --- Helper Functions ---
print_usage() {
    cat << EOF
Advanced video audio cleaning using native FFmpeg filters (macOS & Linux).

Usage:
  $(basename "$0") <input_file> [options]
  $(basename "$0") --install-deps

Options:
  -o, --output <file>    Custom output file path
  --target <float>       Target loudness in LUFS (default: -16.0)
  --noise <float>        Noise reduction strength 0.0-1.0 (default: 0.25)
  --voice <float>        Voice preservation 0.0-1.0 (default: 0.6)
  --no-compression       Disable dynamic range compression
  --no-deess             Disable de-esser
  --no-eq                Disable EQ enhancement
  --info                 Show video information and exit
  --install-deps         Check/Install FFmpeg on macOS or Linux
EOF
}

check_dependencies() {
    if ! command -v ffmpeg &> /dev/null || ! command -v ffprobe &> /dev/null; then
        echo "❌ Error: FFmpeg/FFprobe not found. Run with '--install-deps' or install them manually." >&2
        exit 1
    fi
}

install_dependencies() {
    echo "📦 Checking and installing system dependencies..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            echo "Installing ffmpeg via Homebrew..."
            brew install ffmpeg
        else
            echo "❌ Homebrew is missing. Please install Homebrew first: https://brew.sh/"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y ffmpeg
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y ffmpeg
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm ffmpeg
        else
            echo "❌ Unknown Linux distribution. Please install FFmpeg manually using your package manager."
            exit 1
        fi
    else
        echo "❌ Unsupported OS environment."
        exit 1
    fi
    echo "✅ System dependencies installation completed."
}

show_info() {
    echo "📹 Video Information for: $1"
    ffprobe -v quiet -print_format json -show_format -show_streams "$1" | grep -E '"(duration|size|width|height|codec_name|codec_type|sample_rate)"' || echo "Failed to read properties."
}

# --- CLI Arguments Parser ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-deps)   INSTALL_DEPS_FLAG=true; shift ;;
        --info)           INFO_ONLY=true; shift ;;
        --no-compression) APPLY_COMPRESSION=false; shift ;;
        --no-deess)       APPLY_DEESS=false; shift ;;
        --no-eq)          ENHANCE_EQ=false; shift ;;
        --target)         TARGET_LOUDNESS="$2"; shift 2 ;;
        --noise)          NOISE_REDUCTION="$2"; shift 2 ;;
        --voice)          PRESERVE_VOICE="$2"; shift 2 ;;
        -o|--output)      OUTPUT_FILE="$2"; shift 2 ;;
        -h|--help)        print_usage; exit 0 ;;
        -*)               echo "❌ Unknown option: $1" >&2; print_usage; exit 1 ;;
        *)                INPUT_FILE="$1"; shift ;;
    esac
done

# Execute install-deps early if called
if [ "$INSTALL_DEPS_FLAG" = true ]; then
    install_dependencies
    exit 0
fi

# Input validation
if [ -z "$INPUT_FILE" ]; then
    echo "❌ Error: Missing input video file." >&2
    print_usage
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: File not found -> $INPUT_FILE" >&2
    exit 1
fi

check_dependencies

if [ "$INFO_ONLY" = true ]; then
    show_info "$INPUT_FILE"
    exit 0
fi

# Auto-generate output filename if missing
if [ -z "$OUTPUT_FILE" ]; then
    dir=$(dirname "$INPUT_FILE")
    base=$(basename "$INPUT_FILE")
    OUTPUT_FILE="${dir}/cleaned_${base}"
fi

# --- Core Audio Engine (FFmpeg Filter-Graph Construction) ---
AUDIO_FILTERS=""

# 1. Highpass Filter (Strips sub-80Hz environmental mic hums)
AUDIO_FILTERS+="highpass=f=80"

# 2. Advanced Noise Reduction (afftdn)
noise_reduction_db=$(awk "BEGIN {print 10 + ($NOISE_REDUCTION * 30)}")
AUDIO_FILTERS+=",afftdn=nr=${noise_reduction_db}"

# 3. Dynamic Speech Equalisers (EQ Enhancement)
if [ "$ENHANCE_EQ" = true ]; then
    AUDIO_FILTERS+=",equalizer=f=300:width_type=h:width=200:g=2"
    AUDIO_FILTERS+=",equalizer=f=1000:width_type=h:width=300:g=1.5"
fi

# 4. Sibilance Suppression (De-esser)
if [ "$APPLY_DEESS" = true ]; then
    # Fixed 's=0' parameter to explicitly target standard monaural processing across all versions safely
    AUDIO_FILTERS+=",deesser=f=0.5:s=0:i=0.5"
fi

# 5. Lowpass Filter (Rolls off extreme high end high hiss)
AUDIO_FILTERS+=",lowpass=f=12000"

# 6. Smooth Compand (Dynamic Range Compression)
if [ "$APPLY_COMPRESSION" = true ]; then
    AUDIO_FILTERS+=",compand=attacks=0.1:decays=0.2:points=-50/-50|-40/-30|-20/-20|0/-10"
fi

# 7. Final Hard Peak Limiter and LUFS Normalization
AUDIO_FILTERS+=",alimiter=level_out=0.95:limit=0.95"
AUDIO_FILTERS+=",loudnorm=I=${TARGET_LOUDNESS}:TP=-1.5:LRA=11"

# --- Main Engine Processing ---
start_time=$(date +%s)

echo "🎬 Processing: $INPUT_FILE"
echo "📁 Output:     $OUTPUT_FILE"
echo "--------------------------------------------------"
echo "🎛️  Running combined filter pipeline..."

# Combined high-speed pipeline processing
ffmpeg -y -i "$INPUT_FILE" \
    -af "$AUDIO_FILTERS" \
    -c:v copy \
    -map 0:v:0 -map 0:a:0? \
    -shortest \
    -stats \
    "$OUTPUT_FILE"

if [ $? -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    file_size=$(du -sh "$OUTPUT_FILE" | cut -f1)
    echo "--------------------------------------------------"
    echo "✅ Success! Output created: $OUTPUT_FILE ($file_size)"
    echo "⏱️  Total processing time: ${elapsed} seconds"
    exit 0
else
    echo "❌ Error: Final file merge operation failed." >&2
    exit 1
fi