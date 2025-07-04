# # Install dependencies first
# python clean_video.py --install-deps
#
# # Basic cleaning (recommended for most users)
# python clean_video.py "DevOps076 CI-CD with Kubernetes.mp4"
#
# # More aggressive noise reduction for very noisy audio
# python clean_video.py video.mp4 --noise 0.5
#
# # Preserve more original voice character
# python clean_video.py video.mp4 --voice 0.8
#
# # Make it louder (higher target)
# python clean_video.py video.mp4 --target -12
#
# # Disable specific features if needed
# python clean_video.py video.mp4 --no-compression --no-deess
#
# # Get video info before processing
# python clean_video.py video.mp4 --info


#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile
import shutil
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
import time

# Try to import audio processing libraries with fallback
try:
    import noisereduce as nr
    import librosa
    import soundfile as sf
    import numpy as np
    from scipy import signal
    AUDIO_LIBS_AVAILABLE = True
except ImportError as e:
    AUDIO_LIBS_AVAILABLE = False
    MISSING_LIBS = str(e)

def install_dependencies():
    """Install required audio processing libraries."""
    print("Installing required audio processing libraries...")
    required_packages = [
        "noisereduce",
        "librosa", 
        "soundfile",
        "scipy",
        "numpy",
        "tqdm"
    ]
    
    for package in required_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {package}")
    
    print("Dependencies installation complete. Please restart the script.")

def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_video_info(video_path):
    """Get video information using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Warning: Could not analyze video: {e}")
        return None

def extract_audio_with_progress(input_file, audio_path, duration=None):
    """Extract audio from video with progress tracking."""
    print("📤 Extracting audio from video...")
    
    cmd = [
        "ffmpeg", "-i", str(input_file),
        "-q:a", "0", "-map", "a",
        "-y", str(audio_path),
        "-progress", "pipe:1"
    ]
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
        universal_newlines=True
    )
    
    if duration:
        pbar = tqdm(total=100, desc="Extracting audio", unit="%")
        
        for line in process.stdout:
            if "out_time_ms=" in line:
                try:
                    time_ms = int(line.split("=")[1])
                    current_time = time_ms / 1000000
                    progress = min(int((current_time / duration) * 100), 100)
                    pbar.n = progress
                    pbar.refresh()
                except:
                    pass
        pbar.close()
    
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.SubprocessError("Failed to extract audio")

def apply_noise_reduction(audio, sr, strength=0.25, chunk_size=None):
    """Apply noise reduction with memory management for large files."""
    print(f"🔇 Applying noise reduction (strength: {strength:.2f})...")
    
    # For very large audio files, process in chunks to manage memory
    if chunk_size is None:
        # Calculate chunk size based on available memory (default ~30 seconds)
        chunk_size = sr * 30
    
    if len(audio) <= chunk_size:
        # Process entire audio at once for smaller files
        return nr.reduce_noise(
            y=audio, 
            sr=sr,
            prop_decrease=strength,
            stationary=False,
            n_fft=2048,
            hop_length=512,
            n_std_thresh_stationary=1.5,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
            n_jobs=1  # Conservative to prevent memory issues
        )
    else:
        # Process in chunks for large files
        print(f"Processing large audio file in chunks...")
        processed_chunks = []
        
        with tqdm(total=len(audio), desc="Noise reduction", unit="samples") as pbar:
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                
                # Add overlap for smooth transitions
                overlap = sr // 10  # 0.1 second overlap
                if i > 0:
                    start_idx = max(0, i - overlap)
                    chunk = audio[start_idx:i + chunk_size]
                
                processed_chunk = nr.reduce_noise(
                    y=chunk, 
                    sr=sr,
                    prop_decrease=strength,
                    stationary=False,
                    n_fft=1024,  # Smaller FFT for chunks
                    hop_length=256,
                    n_std_thresh_stationary=1.5,
                    n_jobs=1
                )
                
                # Remove overlap for seamless concatenation
                if i > 0:
                    processed_chunk = processed_chunk[overlap:]
                
                processed_chunks.append(processed_chunk)
                pbar.update(len(chunk))
        
        return np.concatenate(processed_chunks)

def apply_voice_enhancement(audio, sr, preserve_voice=0.6):
    """Apply voice enhancement and preservation filters."""
    print(f"🎤 Enhancing voice clarity (preservation: {preserve_voice:.2f})...")
    
    # Voice frequency range (human speech: ~85-255 Hz fundamental, ~300-3400 Hz important)
    nyquist = sr / 2
    
    # Create multiple filters for different aspects
    
    # 1. Voice band enhancement (300-3400 Hz)
    voice_low = 300 / nyquist
    voice_high = 3400 / nyquist
    voice_b, voice_a = signal.butter(4, [voice_low, voice_high], btype='band')
    voice_enhanced = signal.filtfilt(voice_b, voice_a, audio)
    
    # 2. High-pass filter to remove low-frequency rumble
    hp_b, hp_a = signal.butter(3, 80 / nyquist, btype='high')
    high_passed = signal.filtfilt(hp_b, hp_a, audio)
    
    # 3. Gentle low-pass to remove harsh high frequencies
    lp_b, lp_a = signal.butter(3, 8000 / nyquist, btype='low')
    low_passed = signal.filtfilt(lp_b, lp_a, high_passed)
    
    # Mix the enhanced voice with the processed audio
    enhanced_audio = (1 - preserve_voice) * low_passed + preserve_voice * voice_enhanced
    
    return enhanced_audio

def apply_dynamic_range_compression(audio, sr, ratio=3.0, threshold_db=-20):
    """Apply gentle dynamic range compression to even out volume levels."""
    print("🎚️  Applying dynamic range compression...")
    
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold_db / 20)
    
    # Calculate envelope using RMS
    window_size = int(sr * 0.01)  # 10ms window
    envelope = np.sqrt(np.convolve(audio**2, np.ones(window_size)/window_size, mode='same'))
    
    # Apply compression
    compressed = np.where(
        envelope > threshold_linear,
        audio * (threshold_linear + (envelope - threshold_linear) / ratio) / envelope,
        audio
    )
    
    return compressed

def apply_de_esser(audio, sr, frequency=6000, reduction_db=3):
    """Apply de-esser to reduce sibilance."""
    print("🔉 Reducing sibilance...")
    
    # Create notch filter for sibilant frequencies
    nyquist = sr / 2
    center_freq = frequency / nyquist
    quality_factor = 2.0
    
    # Design notch filter
    b, a = signal.iirnotch(center_freq, quality_factor)
    
    # Apply gentle reduction
    reduction_factor = 10 ** (-reduction_db / 20)
    filtered = signal.filtfilt(b, a, audio)
    
    # Mix original and filtered based on reduction amount
    mix_factor = 1 - reduction_factor
    de_essed = audio * reduction_factor + filtered * mix_factor
    
    return de_essed

def normalize_audio_advanced(audio, target_lufs=-16.0, peak_limit_db=-1.0):
    """Advanced audio normalization with LUFS targeting."""
    print(f"🔊 Normalizing audio (target: {target_lufs} LUFS)...")
    
    # Calculate RMS for basic loudness estimation
    rms = np.sqrt(np.mean(audio**2))
    current_db = 20 * np.log10(rms) if rms > 0 else -80.0
    
    # Estimate gain needed (rough approximation)
    # Note: This is not true LUFS but a reasonable approximation
    estimated_lufs = current_db - 3  # Rough conversion from RMS dB to LUFS-like measure
    gain_db = target_lufs - estimated_lufs
    gain_linear = 10 ** (gain_db / 20.0)
    
    print(f"Estimated current loudness: {estimated_lufs:.1f} LUFS")
    print(f"Applying gain: {gain_db:.1f} dB")
    
    # Apply gain
    normalized = audio * gain_linear
    
    # Apply peak limiting
    peak_limit_linear = 10 ** (peak_limit_db / 20)
    peak = np.max(np.abs(normalized))
    
    if peak > peak_limit_linear:
        limiter_ratio = peak_limit_linear / peak
        normalized = normalized * limiter_ratio
        print(f"Applied peak limiting: {-20*np.log10(limiter_ratio):.1f} dB reduction")
    
    return normalized

def merge_audio_with_progress(input_file, audio_path, output_file, duration=None, enhance_eq=True):
    """Merge cleaned audio back with video using optional EQ enhancement."""
    print("🎬 Merging enhanced audio with video...")
    
    # Build audio filter chain
    audio_filters = []
    
    if enhance_eq:
        # Voice enhancement EQ chain
        eq_filters = [
            "highpass=f=80",           # Remove low rumble
            "lowpass=f=12000",         # Remove harsh highs
            "equalizer=f=300:width_type=h:width=200:g=2",   # Boost low voice
            "equalizer=f=1000:width_type=h:width=300:g=1.5", # Boost mid voice
            "equalizer=f=3000:width_type=h:width=500:g=1",   # Subtle high voice boost
            "compand=0.1,0.2:-50/-50,-40/-30,-20/-20,0/-10:0.1:0.1", # Gentle compression
        ]
        audio_filters.extend(eq_filters)
    
    # Additional processing
    audio_filters.extend([
        "alimiter=level_in=1:level_out=0.95:limit=0.95", # Soft limiter
        "loudnorm=I=-16:TP=-1.5:LRA=11"  # Professional loudness normalization
    ])
    
    af_chain = ",".join(audio_filters)
    
    cmd = [
        "ffmpeg", "-i", str(input_file),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-map", "0:v:0", "-map", "1:a:0",
        "-af", af_chain,
        "-shortest",
        "-y", str(output_file),
        "-progress", "pipe:1"
    ]
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    if duration:
        pbar = tqdm(total=100, desc="Merging audio", unit="%")
        
        for line in process.stdout:
            if "out_time_ms=" in line:
                try:
                    time_ms = int(line.split("=")[1])
                    current_time = time_ms / 1000000
                    progress = min(int((current_time / duration) * 100), 100)
                    pbar.n = progress
                    pbar.refresh()
                except:
                    pass
        pbar.close()
    
    return_code = process.wait()
    if return_code != 0:
        stderr_output = process.stderr.read()
        raise subprocess.SubprocessError(f"Failed to merge audio: {stderr_output}")

def clean_video_audio(input_file, output_file=None, **options):
    """
    Main function to clean video audio with comprehensive processing.
    """
    if not AUDIO_LIBS_AVAILABLE:
        print(f"❌ Required audio libraries not available: {MISSING_LIBS}")
        print("Run with --install-deps to install them.")
        return None
    
    # Set default options
    config = {
        'target_loudness': -16.0,
        'noise_reduction_strength': 0.25,
        'preserve_voice': 0.6,
        'apply_compression': True,
        'apply_de_esser': True,
        'enhance_eq': True,
        'chunk_processing': True
    }
    config.update(options)
    
    # Generate output filename
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"cleaned_{input_path.name}"
    
    # Get video info for progress tracking
    video_info = get_video_info(input_file)
    duration = None
    if video_info:
        try:
            duration = float(video_info['format']['duration'])
        except:
            pass
    
    print(f"🎬 Processing: {input_file}")
    print(f"📁 Output: {output_file}")
    if duration:
        print(f"⏱️  Duration: {duration:.1f} seconds")
    print("-" * 50)
    
    # Process audio
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "extracted_audio.wav"
        cleaned_audio_path = Path(temp_dir) / "cleaned_audio.wav"
        
        try:
            # Step 1: Extract audio
            extract_audio_with_progress(input_file, audio_path, duration)
            
            # Step 2: Load audio
            print("📊 Loading and analyzing audio...")
            y, sr = librosa.load(str(audio_path), sr=None)
            print(f"   Sample rate: {sr} Hz")
            print(f"   Duration: {len(y)/sr:.1f} seconds")
            print(f"   Channels: Mono")
            
            # Step 3: Apply noise reduction
            if config['noise_reduction_strength'] > 0:
                y = apply_noise_reduction(y, sr, config['noise_reduction_strength'])
            
            # Step 4: Apply voice enhancement
            y = apply_voice_enhancement(y, sr, config['preserve_voice'])
            
            # Step 5: Apply dynamic range compression
            if config['apply_compression']:
                y = apply_dynamic_range_compression(y, sr)
            
            # Step 6: Apply de-esser
            if config['apply_de_esser']:
                y = apply_de_esser(y, sr)
            
            # Step 7: Normalize audio
            y = normalize_audio_advanced(y, config['target_loudness'])
            
            # Step 8: Save cleaned audio
            print("💾 Saving processed audio...")
            sf.write(str(cleaned_audio_path), y, sr, subtype='PCM_24')
            
            # Step 9: Merge with video
            merge_audio_with_progress(
                input_file, cleaned_audio_path, output_file, 
                duration, config['enhance_eq']
            )
            
            # Verify output
            if Path(output_file).exists():
                file_size = Path(output_file).stat().st_size / (1024 * 1024)
                print(f"✅ Success! Output: {output_file} ({file_size:.1f} MB)")
                return str(output_file)
            else:
                print("❌ Error: Output file was not created")
                return None
                
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    parser = argparse.ArgumentParser(
        description="Advanced video audio cleaning with noise reduction, voice enhancement, and normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean_video.py video.mp4                    # Basic cleaning
  python clean_video.py video.mp4 -o clean.mp4      # Custom output
  python clean_video.py video.mp4 --noise 0.5       # More aggressive noise reduction
  python clean_video.py video.mp4 --voice 0.8       # Preserve more original voice
  python clean_video.py video.mp4 --target -14       # Louder output
        """
    )
    
    parser.add_argument("input_file", nargs='?', help="Input video file")
    parser.add_argument("-o", "--output", help="Output file (default: 'cleaned_input.mp4')")
    
    # Audio processing options
    parser.add_argument("--target", type=float, default=-16.0,
                        help="Target loudness in LUFS (default: -16.0)")
    parser.add_argument("--noise", type=float, default=0.25,
                        help="Noise reduction strength 0.0-1.0 (default: 0.25)")
    parser.add_argument("--voice", type=float, default=0.6,
                        help="Voice preservation 0.0-1.0 (default: 0.6)")
    
    # Feature toggles
    parser.add_argument("--no-compression", action="store_true",
                        help="Disable dynamic range compression")
    parser.add_argument("--no-deess", action="store_true",
                        help="Disable de-esser")
    parser.add_argument("--no-eq", action="store_true",
                        help="Disable EQ enhancement")
    
    # Utility options
    parser.add_argument("--install-deps", action="store_true",
                        help="Install required audio processing libraries")
    parser.add_argument("--info", action="store_true",
                        help="Show video information and exit")
    
    args = parser.parse_args()
    
    # Handle utility commands first (before checking input file)
    if args.install_deps:
        install_dependencies()
        return 0
    
    # Now check if input file is provided for other operations
    if not args.input_file:
        print("❌ Error: input_file is required (except for --install-deps)")
        parser.print_help()
        return 1
    
    # Check requirements
    if not check_ffmpeg():
        print("❌ FFmpeg not found. Please install FFmpeg.")
        return 1
    
    if not Path(args.input_file).exists():
        print(f"❌ Input file not found: {args.input_file}")
        return 1
    
    if args.info:
        info = get_video_info(args.input_file)
        if info:
            print(f"📹 Video Information for: {args.input_file}")
            print(f"Duration: {float(info['format']['duration']):.1f} seconds")
            print(f"Size: {int(info['format']['size']) / 1024 / 1024:.1f} MB")
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    print(f"Video: {stream['width']}x{stream['height']}, {stream['codec_name']}")
                elif stream['codec_type'] == 'audio':
                    print(f"Audio: {stream['codec_name']}, {stream.get('sample_rate', 'unknown')} Hz")
        return 0
    
    # Validate parameters
    if not (0.0 <= args.noise <= 1.0):
        print("❌ Noise reduction must be between 0.0 and 1.0")
        return 1
    
    if not (0.0 <= args.voice <= 1.0):
        print("❌ Voice preservation must be between 0.0 and 1.0")
        return 1
    
    # Process video
    start_time = time.time()
    
    options = {
        'target_loudness': args.target,
        'noise_reduction_strength': args.noise,
        'preserve_voice': args.voice,
        'apply_compression': not args.no_compression,
        'apply_de_esser': not args.no_deess,
        'enhance_eq': not args.no_eq
    }
    
    result = clean_video_audio(args.input_file, args.output, **options)
    
    if result:
        processing_time = time.time() - start_time
        print(f"⏱️  Total processing time: {processing_time:.1f} seconds")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())