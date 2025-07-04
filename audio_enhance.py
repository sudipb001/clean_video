#!/usr/bin/env python3
"""
Advanced Audio Enhancement Tool for Videos
Provides real-world audio improvements including noise reduction, voice enhancement,
dynamic range processing, and professional audio normalization.

# Smart audio analysis → Adaptive processing
1. 📊 Audio analysis (noise level, speech content, dynamics)
2. 🔇 Advanced noise reduction (spectral subtraction + Wiener filtering)
3. ⚡ Electrical hum removal (50/60Hz + harmonics)
4. 🎤 Voice frequency enhancement (fundamental + formants + presence)
5. 🎚️  Dynamic range compression (smooth attack/release)
6. 🔉 De-essing (sibilance reduction)
7. 🔊 Professional loudness normalization (LUFS-based)

Usage: python audio_enhance.py input_video.mp4 [options]

# Install dependencies first
python audio_enhance.py --install-deps

# Basic enhancement (recommended)
python audio_enhance.py "DevOps076 CI-CD with Kubernetes.mp4"

# Custom output location
python audio_enhance.py video.mp4 -o enhanced_video.mp4

# Aggressive noise reduction for very noisy audio
python audio_enhance.py video.mp4 --noise 0.6

# Make it louder
python audio_enhance.py video.mp4 --target -12

# Disable specific features for music/non-speech content
python audio_enhance.py music_video.mp4 --no-voice --no-deess

# Get video info before processing
python audio_enhance.py video.mp4 --info

# Minimal processing (just noise reduction)
python audio_enhance.py video.mp4 --noise 0.2 --no-compression --no-deess
"""

#!/usr/bin/env python3
"""
Advanced Audio Enhancement Tool for Videos
Provides real-world audio improvements including noise reduction, voice enhancement,
dynamic range processing, and professional audio normalization.

Usage: python audio_enhance.py input_video.mp4 [options]
"""

import os
import sys
import argparse
import numpy as np
import subprocess
import tempfile
import warnings
import time
import json
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Check for required libraries
REQUIRED_LIBS = ['librosa', 'soundfile', 'scipy', 'noisereduce']
MISSING_LIBS = []

for lib in REQUIRED_LIBS:
    try:
        if lib == 'noisereduce':
            import noisereduce as nr
        elif lib == 'librosa':
            import librosa
        elif lib == 'soundfile':
            import soundfile as sf
        elif lib == 'scipy':
            import scipy.signal
            import scipy.ndimage
    except ImportError:
        MISSING_LIBS.append(lib)

if MISSING_LIBS:
    print("Missing required libraries. Install with:")
    print(f"pip install {' '.join(MISSING_LIBS)} tqdm")
    if '--install-deps' not in sys.argv:
        sys.exit(1)

class AdvancedAudioEnhancer:
    def __init__(self, config=None):
        """Initialize the enhanced audio processor."""
        self.config = config or {}
        self.temp_files = []  # Track temp files for cleanup

        # Default enhancement parameters
        self.defaults = {
            'noise_reduction_strength': 0.3,
            'voice_enhancement': True,
            'dynamic_compression': True,
            'normalize_loudness': True,
            'target_lufs': -16.0,
            'spectral_gate_db': 12,
            'enhance_speech': True,
            'remove_hum': True,
            'de_ess': True
        }

        # Merge with user config
        for key, value in self.defaults.items():
            if key not in self.config:
                self.config[key] = value

    def __del__(self):
        """Clean up temporary files."""
        self.cleanup_temp_files()

    def cleanup_temp_files(self):
        """Remove all temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        self.temp_files.clear()

    def check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_video_info(self, video_path):
        """Get detailed video information."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except:
            return None

    def extract_audio_advanced(self, video_path):
        """Extract high-quality audio with progress tracking."""
        print("📤 Extracting audio from video...")

        # Create temporary audio file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio.close()
        self.temp_files.append(temp_audio.name)

        # Get video duration for progress
        info = self.get_video_info(video_path)
        duration = None
        if info:
            try:
                duration = float(info['format']['duration'])
            except:
                pass

        # High-quality extraction command
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s24le',  # 24-bit PCM for better quality
            '-ar', '48000',  # 48kHz sample rate (professional standard)
            '-ac', '2',  # Preserve stereo if available
            '-af', 'highpass=f=20,lowpass=f=20000',  # Remove extreme frequencies
            '-y', temp_audio.name,
            '-progress', 'pipe:1'
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

        if process.wait() != 0:
            raise RuntimeError("Failed to extract audio from video")

        return temp_audio.name

    def analyze_audio_characteristics(self, audio, sr):
        """Analyze audio to determine optimal processing parameters."""
        print("📊 Analyzing audio characteristics...")

        # Calculate various audio metrics
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        dynamic_range = 20 * np.log10(peak / (rms + 1e-10))

        # Spectral analysis
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)

        # Detect noise floor
        noise_floor = np.percentile(magnitude, 5)

        # Detect dominant frequencies
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
        avg_spectrum = np.mean(magnitude, axis=1)

        # Voice frequency analysis (300-3400 Hz)
        voice_start_bin = int(300 * 2048 / sr)
        voice_end_bin = int(3400 * 2048 / sr)
        voice_energy = np.mean(avg_spectrum[voice_start_bin:voice_end_bin])
        total_energy = np.mean(avg_spectrum)
        voice_ratio = voice_energy / (total_energy + 1e-10)

        analysis = {
            'rms_db': 20 * np.log10(rms + 1e-10),
            'peak_db': 20 * np.log10(peak + 1e-10),
            'dynamic_range': dynamic_range,
            'noise_floor_db': 20 * np.log10(noise_floor + 1e-10),
            'voice_ratio': voice_ratio,
            'is_speech_heavy': voice_ratio > 0.3,
            'is_noisy': noise_floor > np.percentile(magnitude, 15)
        }

        print(f"   RMS Level: {analysis['rms_db']:.1f} dB")
        print(f"   Peak Level: {analysis['peak_db']:.1f} dB")
        print(f"   Dynamic Range: {analysis['dynamic_range']:.1f} dB")
        print(f"   Speech Content: {'High' if analysis['is_speech_heavy'] else 'Low'}")
        print(f"   Noise Level: {'High' if analysis['is_noisy'] else 'Low'}")

        return analysis

    def apply_advanced_noise_reduction(self, audio, sr, strength=0.3):
        """Apply sophisticated noise reduction with adaptive parameters."""
        print(f"🔇 Applying advanced noise reduction (strength: {strength:.2f})...")

        # Use multiple noise reduction strategies

        # 1. Spectral subtraction for stationary noise
        processed = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=strength,
            stationary=False,
            n_fft=2048,
            hop_length=512,
            n_std_thresh_stationary=1.5,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50
        )

        # 2. Wiener filtering for additional cleaning
        if strength > 0.4:  # Only for aggressive noise reduction
            stft = librosa.stft(processed, n_fft=1024, hop_length=256)
            magnitude, phase = librosa.magphase(stft)

            # Estimate noise from quieter portions
            power = magnitude ** 2
            noise_power = np.percentile(power, 10, axis=1, keepdims=True)

            # Wiener filter
            wiener_gain = power / (power + noise_power + 1e-10)
            wiener_gain = np.clip(wiener_gain, 0.1, 1.0)  # Prevent over-suppression

            filtered_stft = magnitude * wiener_gain * phase
            processed = librosa.istft(filtered_stft, hop_length=256)

        return processed

    def enhance_voice_frequencies(self, audio, sr):
        """Enhance voice frequencies and clarity."""
        if not self.config['voice_enhancement']:
            return audio

        print("🎤 Enhancing voice frequencies...")

        # Multi-band voice enhancement
        nyquist = sr / 2

        # 1. Fundamental frequency boost (85-255 Hz)
        fundamental_low = 85 / nyquist
        fundamental_high = 255 / nyquist
        b1, a1 = scipy.signal.butter(3, [fundamental_low, fundamental_high], btype='band')
        fundamental_enhanced = scipy.signal.filtfilt(b1, a1, audio) * 0.2

        # 2. Formant frequency boost (300-3400 Hz) - main speech intelligibility
        formant_low = 300 / nyquist
        formant_high = 3400 / nyquist
        b2, a2 = scipy.signal.butter(4, [formant_low, formant_high], btype='band')
        formant_enhanced = scipy.signal.filtfilt(b2, a2, audio) * 0.4

        # 3. Presence boost (4000-6000 Hz) - speech clarity
        presence_low = 4000 / nyquist
        presence_high = 6000 / nyquist
        b3, a3 = scipy.signal.butter(3, [presence_low, presence_high], btype='band')
        presence_enhanced = scipy.signal.filtfilt(b3, a3, audio) * 0.15

        # Combine enhancements
        enhanced = audio + fundamental_enhanced + formant_enhanced + presence_enhanced

        # Apply gentle compression to voice band
        voice_band = scipy.signal.filtfilt(b2, a2, enhanced)
        compressed_voice = self.apply_soft_compression(voice_band, ratio=2.0, threshold_db=-15)

        # Mix back with original
        final = enhanced * 0.7 + compressed_voice * 0.3

        return final

    def remove_electrical_hum(self, audio, sr):
        """Remove 50/60 Hz electrical hum and harmonics."""
        if not self.config['remove_hum']:
            return audio

        print("⚡ Removing electrical interference...")

        # Remove 50Hz and 60Hz hum + harmonics
        hum_frequencies = [50, 60, 100, 120, 150, 180]

        processed = audio.copy()
        for freq in hum_frequencies:
            if freq < sr / 2:  # Ensure frequency is below Nyquist
                # Notch filter for each hum frequency
                Q = 10  # Quality factor
                w0 = freq / (sr / 2)
                b, a = scipy.signal.iirnotch(w0, Q)
                processed = scipy.signal.filtfilt(b, a, processed)

        return processed

    def apply_soft_compression(self, audio, sr, ratio=3.0, threshold_db=-20, attack_ms=5, release_ms=50):
        """Apply dynamic range compression with smooth attack/release."""
        if not self.config['dynamic_compression']:
            return audio

        print(f"🎚️  Applying dynamic compression (ratio: {ratio:.1f}:1)...")

        # Convert threshold to linear
        threshold_linear = 10 ** (threshold_db / 20)

        # Calculate envelope with attack/release
        envelope = np.abs(audio)

        # Smooth envelope
        window_size = int(sr * 0.001)  # 1ms smoothing
        if window_size > 1:
            envelope = scipy.ndimage.uniform_filter1d(envelope, size=window_size)

        # Calculate gain reduction
        gain = np.ones_like(envelope)
        over_threshold = envelope > threshold_linear

        if np.any(over_threshold):
            # Apply compression formula
            compressed_level = threshold_linear * (envelope[over_threshold] / threshold_linear) ** (1.0 / ratio)
            gain[over_threshold] = compressed_level / envelope[over_threshold]

        # Apply gain smoothing
        gain = scipy.ndimage.uniform_filter1d(gain, size=window_size * 2)

        return audio * gain

    def apply_de_esser(self, audio, sr):
        """Reduce harsh sibilant sounds."""
        if not self.config['de_ess']:
            return audio

        print("🔉 Reducing sibilance...")

        # Detect sibilant frequencies (5-8 kHz)
        nyquist = sr / 2
        sibilant_low = 5000 / nyquist
        sibilant_high = 8000 / nyquist

        # Extract sibilant band
        b, a = scipy.signal.butter(4, [sibilant_low, sibilant_high], btype='band')
        sibilant_band = scipy.signal.filtfilt(b, a, audio)

        # Dynamic sibilance reduction
        threshold = np.percentile(np.abs(sibilant_band), 85)
        reduction_factor = 0.3

        sibilant_reduced = np.where(
            np.abs(sibilant_band) > threshold,
            sibilant_band * reduction_factor,
            sibilant_band
        )

        # Replace sibilant frequencies in original
        sibilant_original = scipy.signal.filtfilt(b, a, audio)
        return audio - sibilant_original + sibilant_reduced

    def normalize_loudness_professional(self, audio, sr, target_lufs=-16.0):
        """Professional loudness normalization."""
        if not self.config['normalize_loudness']:
            return audio

        print(f"🔊 Normalizing loudness (target: {target_lufs} LUFS)...")

        # Calculate current loudness (approximation)
        rms = np.sqrt(np.mean(audio**2))
        current_lufs = 20 * np.log10(rms + 1e-10) - 3  # Rough conversion to LUFS

        # Calculate required gain
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)

        print(f"   Current loudness: ~{current_lufs:.1f} LUFS")
        print(f"   Applying gain: {gain_db:.1f} dB")

        # Apply gain
        normalized = audio * gain_linear

        # True peak limiting to prevent clipping
        peak = np.max(np.abs(normalized))
        if peak > 0.95:
            limiter_gain = 0.95 / peak
            normalized *= limiter_gain
            print(f"   Applied peak limiting: {20*np.log10(limiter_gain):.1f} dB")

        return normalized

    def enhance_audio_comprehensive(self, audio_path):
        """Apply comprehensive audio enhancement pipeline."""
        print("🎵 Loading audio for enhancement...")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=False)

        # Handle stereo/mono
        if audio.ndim > 1:
            # Process stereo channels separately then combine
            print(f"   Loaded stereo audio: {sr} Hz, {len(audio[0])/sr:.1f}s")
            left = audio[0]
            right = audio[1] if len(audio) > 1 else audio[0]

            # Process each channel
            left_enhanced = self._process_channel(left, sr)
            right_enhanced = self._process_channel(right, sr)

            enhanced_audio = np.array([left_enhanced, right_enhanced])
        else:
            print(f"   Loaded mono audio: {sr} Hz, {len(audio)/sr:.1f}s")
            enhanced_audio = self._process_channel(audio, sr)

        # Save enhanced audio
        temp_enhanced = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_enhanced.close()
        self.temp_files.append(temp_enhanced.name)

        print("💾 Saving enhanced audio...")
        if enhanced_audio.ndim > 1:
            sf.write(temp_enhanced.name, enhanced_audio.T, sr, subtype='PCM_24')
        else:
            sf.write(temp_enhanced.name, enhanced_audio, sr, subtype='PCM_24')

        return temp_enhanced.name

    def _process_channel(self, audio, sr):
        """Process a single audio channel through the enhancement pipeline."""
        # Analyze audio first
        analysis = self.analyze_audio_characteristics(audio, sr)

        # Adaptive processing based on analysis
        strength = self.config['noise_reduction_strength']
        if analysis['is_noisy']:
            strength = min(strength * 1.3, 0.7)  # Increase for noisy audio

        processed = audio.copy()

        # 1. Noise reduction
        if strength > 0:
            processed = self.apply_advanced_noise_reduction(processed, sr, strength)

        # 2. Remove electrical hum
        processed = self.remove_electrical_hum(processed, sr)

        # 3. Enhance voice frequencies (if speech content detected)
        if analysis['is_speech_heavy'] and self.config['enhance_speech']:
            processed = self.enhance_voice_frequencies(processed, sr)

        # 4. Dynamic compression
        processed = self.apply_soft_compression(processed, sr)

        # 5. De-essing
        processed = self.apply_de_esser(processed, sr)

        # 6. Final loudness normalization
        processed = self.normalize_loudness_professional(processed, sr, self.config['target_lufs'])

        return processed

    def combine_with_video_advanced(self, video_path, enhanced_audio_path, output_path):
        """Combine enhanced audio with video using advanced encoding."""
        print("🎬 Combining enhanced audio with video...")

        # Get video info for progress tracking
        info = self.get_video_info(video_path)
        duration = None
        if info:
            try:
                duration = float(info['format']['duration'])
            except:
                pass

        # Advanced audio encoding with multiple format support
        cmd = [
            'ffmpeg', '-i', str(video_path), '-i', enhanced_audio_path,
            '-c:v', 'copy',  # Copy video stream
            '-c:a', 'aac',   # High-quality AAC audio
            '-b:a', '256k',  # High bitrate for quality
            '-ar', '48000',  # Professional sample rate
            '-af', 'alimiter=level_in=1:level_out=0.95:limit=0.95,loudnorm=I=-16:TP=-1.5:LRA=11',  # Final limiting and normalization
            '-map', '0:v:0', '-map', '1:a:0',  # Map video from input 1, audio from input 2
            '-movflags', '+faststart',  # Optimize for streaming
            '-y', str(output_path),
            '-progress', 'pipe:1'
        ]

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True
        )

        if duration:
            pbar = tqdm(total=100, desc="Combining audio & video", unit="%")

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

        if process.wait() != 0:
            stderr_output = process.stderr.read()
            raise RuntimeError(f"Failed to combine audio with video: {stderr_output}")

    def enhance_video(self, input_path, output_path=None):
        """Main enhancement pipeline."""
        if not self.check_ffmpeg():
            raise RuntimeError("FFmpeg is required but not found. Please install FFmpeg.")

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            output_path = input_path.parent / f"enhanced_{input_path.name}"

        print(f"\n{'='*60}")
        print(f"🎬 Advanced Audio Enhancement")
        print(f"📁 Input: {input_path}")
        print(f"📁 Output: {output_path}")
        print(f"{'='*60}\n")

        start_time = time.time()

        try:
            # Step 1: Extract audio
            print("[1/3] Extracting audio...")
            audio_path = self.extract_audio_advanced(input_path)

            # Step 2: Enhance audio
            print("\n[2/3] Enhancing audio...")
            enhanced_audio_path = self.enhance_audio_comprehensive(audio_path)

            # Step 3: Combine with video
            print("\n[3/3] Combining with video...")
            self.combine_with_video_advanced(input_path, enhanced_audio_path, output_path)

            # Success message
            total_time = time.time() - start_time
            file_size = Path(output_path).stat().st_size / (1024 * 1024)

            print(f"\n{'='*60}")
            print(f"✅ Enhancement completed in {total_time:.1f} seconds")
            print(f"📁 Output: {output_path} ({file_size:.1f} MB)")
            print(f"{'='*60}\n")

            # Show applied enhancements
            print("🎵 Applied enhancements:")
            if self.config['noise_reduction_strength'] > 0:
                print(f"   ✓ Noise reduction ({self.config['noise_reduction_strength']:.2f})")
            if self.config['voice_enhancement']:
                print("   ✓ Voice frequency enhancement")
            if self.config['remove_hum']:
                print("   ✓ Electrical hum removal")
            if self.config['dynamic_compression']:
                print("   ✓ Dynamic range compression")
            if self.config['de_ess']:
                print("   ✓ De-essing (sibilance reduction)")
            if self.config['normalize_loudness']:
                print(f"   ✓ Loudness normalization ({self.config['target_lufs']} LUFS)")

            return str(output_path)

        finally:
            self.cleanup_temp_files()

def install_dependencies():
    """Install required packages."""
    packages = ['numpy', 'scipy', 'librosa', 'soundfile', 'noisereduce', 'tqdm']
    print("Installing required packages...")

    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {package}")

    print("Installation complete. Please restart the script.")

def main():
    parser = argparse.ArgumentParser(
        description="Advanced audio enhancement for videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audio_enhance.py video.mp4                    # Basic enhancement
  python audio_enhance.py video.mp4 -o clean.mp4      # Custom output
  python audio_enhance.py video.mp4 --noise 0.5       # Aggressive noise reduction
  python audio_enhance.py video.mp4 --no-voice        # Disable voice enhancement
  python audio_enhance.py video.mp4 --target -14       # Louder output
        """
    )

    parser.add_argument('input_video', nargs='?', help='Input video file')
    parser.add_argument('-o', '--output', help='Output video file')

    # Enhancement options
    parser.add_argument('--noise', type=float, default=0.3,
                        help='Noise reduction strength 0.0-1.0 (default: 0.3)')
    parser.add_argument('--target', type=float, default=-16.0,
                        help='Target loudness in LUFS (default: -16.0)')

    # Feature toggles
    parser.add_argument('--no-voice', action='store_true',
                        help='Disable voice enhancement')
    parser.add_argument('--no-compression', action='store_true',
                        help='Disable dynamic compression')
    parser.add_argument('--no-deess', action='store_true',
                        help='Disable de-essing')
    parser.add_argument('--no-hum-removal', action='store_true',
                        help='Disable electrical hum removal')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable loudness normalization')

    # Utility options
    parser.add_argument('--install-deps', action='store_true',
                        help='Install required dependencies')
    parser.add_argument('--info', action='store_true',
                        help='Show video information')

    args = parser.parse_args()

    # Handle utility commands
    if args.install_deps:
        install_dependencies()
        return 0

    if not args.input_video:
        print("❌ Error: input_video is required (except for --install-deps)")
        parser.print_help()
        return 1

    # Validate parameters
    if not (0.0 <= args.noise <= 1.0):
        print("❌ Error: --noise must be between 0.0 and 1.0")
        return 1

    if MISSING_LIBS:
        print(f"❌ Missing libraries: {', '.join(MISSING_LIBS)}")
        print("Run with --install-deps to install them.")
        return 1

    # Create enhancer configuration
    config = {
        'noise_reduction_strength': args.noise,
        'target_lufs': args.target,
        'voice_enhancement': not args.no_voice,
        'dynamic_compression': not args.no_compression,
        'de_ess': not args.no_deess,
        'remove_hum': not args.no_hum_removal,
        'normalize_loudness': not args.no_normalize,
        'enhance_speech': not args.no_voice
    }

    try:
        enhancer = AdvancedAudioEnhancer(config)

        if args.info:
            info = enhancer.get_video_info(args.input_video)
            if info:
                print(f"📹 Video Information: {args.input_video}")
                duration = float(info['format']['duration'])
                size_mb = int(info['format']['size']) / (1024 * 1024)
                print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
                print(f"Size: {size_mb:.1f} MB")

                for stream in info['streams']:
                    if stream['codec_type'] == 'audio':
                        print(f"Audio: {stream['codec_name']}, {stream.get('sample_rate', 'unknown')} Hz, {stream.get('channels', 'unknown')} channels")
            return 0

        result = enhancer.enhance_video(args.input_video, args.output)
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
