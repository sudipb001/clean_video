import os
import sys
import argparse
import noisereduce as nr
import librosa
import soundfile as sf
import subprocess
import tempfile
import numpy as np
from scipy import signal

def clean_video_audio(input_file, target_loudness=-16.0, noise_reduction_strength=0.25, preserve_voice=0.6):
    """
    Function to clean the audio of a video file and normalize volume
    
    Parameters:
    input_file (str): Name of the input MP4 file
    target_loudness (float): Target loudness in dBFS
    noise_reduction_strength (float): Strength of noise reduction (0.0-1.0)
    preserve_voice (float): How much to preserve voice frequencies (0.0-1.0)
    
    Returns:
    str: Name of the output MP4 file with clean audio
    """
    # Create output filename
    base_name = os.path.basename(input_file)
    output_file = f"corrected_{base_name}"
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # File paths for temporary files
        audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        cleaned_audio_path = os.path.join(temp_dir, "cleaned_audio.wav")
        
        print(f"Extracting audio from {input_file}...")
        # Step 1: Extract audio from the video with high quality
        subprocess.run([
            "ffmpeg", "-i", input_file, 
            "-q:a", "0", "-map", "a", 
            audio_path
        ], check=True)
        
        print("Loading audio data...")
        # Step 2: Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        print(f"Applying gentle noise reduction (strength: {noise_reduction_strength})...")
        # Step 3: Apply adaptive noise reduction with voice preservation
        reduced_noise = nr.reduce_noise(
            y=y, 
            sr=sr,
            prop_decrease=noise_reduction_strength,
            stationary=False,      # Non-stationary for varying background noise
            n_fft=2048,            # Larger FFT window for better frequency resolution
            win_length=1024,       # Window length
            n_std_thresh_stationary=1.2,  # Less aggressive threshold to avoid voice distortion
            freq_mask_smooth_hz=500,      # Smoother frequency masking
            n_jobs=-1              # Use all available CPU cores
        )
        
        print("Applying voice preservation filter...")
        # Apply a bandpass filter to preserve voice frequencies (typically 300-3400 Hz)
        # This helps reduce distortion in the voice range
        voice_low = 250  # Hz
        voice_high = 3800  # Hz
        
        # Create bandpass filter for voice frequencies
        nyquist = sr / 2
        low = voice_low / nyquist
        high = voice_high / nyquist
        b, a = signal.butter(3, [low, high], btype='band')
        
        # Apply the filter
        voice_filtered = signal.filtfilt(b, a, y)
        
        # Mix the original voice frequencies with the noise-reduced audio
        # preserve_voice controls how much of the original voice to keep
        mixed_audio = (1 - preserve_voice) * reduced_noise + preserve_voice * voice_filtered
        
        # Apply a gentle de-esser to reduce sibilance ('s' sounds) that might be amplified
        print("Reducing sibilance...")
        # Simple de-esser: reduce high frequencies in specific range (5000-8000 Hz)
        deess_low = 5000 / nyquist
        deess_high = 8000 / nyquist
        b_deess, a_deess = signal.butter(2, [deess_low, deess_high], btype='bandstop')
        deessed_audio = signal.filtfilt(b_deess, a_deess, mixed_audio)
        
        # Step 4: Volume normalization
        print("Normalizing volume...")
        
        # Calculate RMS volume
        rms = np.sqrt(np.mean(deessed_audio**2))
        current_db = 20 * np.log10(rms) if rms > 0 else -80.0
        
        print(f"Current audio level: {current_db:.2f} dBFS")
        print(f"Target loudness: {target_loudness:.2f} dBFS")
        
        # Calculate gain needed
        gain_db = target_loudness - current_db
        gain_linear = 10 ** (gain_db / 20.0)
        
        print(f"Applying gain of {gain_db:.2f} dB")
        normalized_audio = deessed_audio * gain_linear
        
        # Prevent clipping
        if np.max(np.abs(normalized_audio)) > 0.95:
            normalized_audio = normalized_audio / np.max(np.abs(normalized_audio)) * 0.95
            print("Applied anti-clipping protection")
        
        print("Saving cleaned and volume-adjusted audio...")
        sf.write(cleaned_audio_path, normalized_audio, sr)
        
        # Step 5: Merge the cleaned audio back into the video
        # Use ffmpeg's built-in audio filters for the final touch
        print(f"Merging clean audio back into video to create {output_file}...")
        subprocess.run([
            "ffmpeg", "-i", input_file, 
            "-i", cleaned_audio_path,
            "-c:v", "copy",  # Copy the video stream without re-encoding
            "-map", "0:v:0",  # Use video from first input
            "-map", "1:a:0",  # Use audio from second input
            "-af", "highpass=f=80,lowpass=f=10000,equalizer=f=300:width_type=h:width=200:g=2,equalizer=f=1000:width_type=h:width=200:g=1",  # Voice enhancement EQ
            "-shortest",  # Finish encoding when the shortest input stream ends
            output_file
        ], check=True)
        
    print(f"Processing complete. Output saved as {output_file}")
    return output_file

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Clean noise, echo, and distortion from MP4 video files and normalize volume"
    )
    parser.add_argument(
        "input_file", 
        help="Input MP4 file name"
    )
    parser.add_argument(
        "--output-file", 
        help="Output file name (default: 'corrected_input.mp4')",
        default=None
    )
    parser.add_argument(
        "--target-loudness",
        type=float,
        default=-16.0,
        help="Target loudness level in dB (default: -16.0, higher values = louder)"
    )
    parser.add_argument(
        "--noise-reduction",
        type=float,
        default=0.25,
        help="Noise reduction strength from 0.0 to 1.0 (default: 0.25)"
    )
    parser.add_argument(
        "--preserve-voice",
        type=float,
        default=0.6,
        help="Voice preservation level from 0.0 to 1.0 (default: 0.6)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    if not args.input_file.lower().endswith('.mp4'):
        print(f"Warning: Input file '{args.input_file}' doesn't have .mp4 extension.")
        
    # Validate parameters
    if args.noise_reduction < 0.0 or args.noise_reduction > 1.0:
        print(f"Warning: Noise reduction value should be between 0.0 and 1.0. Using default (0.25).")
        args.noise_reduction = 0.25
        
    if args.preserve_voice < 0.0 or args.preserve_voice > 1.0:
        print(f"Warning: Voice preservation value should be between 0.0 and 1.0. Using default (0.6).")
        args.preserve_voice = 0.6
    
    try:
        # Process the video
        output_file = clean_video_audio(
            args.input_file, 
            args.target_loudness,
            args.noise_reduction,
            args.preserve_voice
        )
        
        # If output file was specified and different from default, rename
        if args.output_file and args.output_file != output_file:
            os.rename(output_file, args.output_file)
            print(f"File renamed to {args.output_file}")
            
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()