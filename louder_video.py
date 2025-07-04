# # Test analysis first (no processing)
# python3 louder_video.py "DevOps076 CI-CD with Kubernetes.mp4" --test
#
# # Make it significantly louder
# python3 louder_video.py "DevOps076 CI-CD with Kubernetes.mp4" -b 3.0
#
# # Use professional normalization
# python3 louder_video.py "DevOps076 CI-CD with Kubernetes.mp4" --normalize

import subprocess
import os
import sys
import json
import numpy as np
import argparse
import time
from pathlib import Path
from tqdm import tqdm

def install_dependencies():
    """Install required dependencies."""
    print("Installing required dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "tqdm"], 
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Dependencies installed.")

def get_video_duration(video_path):
    """Get the duration of the video in seconds using ffprobe."""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "json", 
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except:
        # Fallback method
        return 120.0  # Default duration for progress bar

def get_simple_audio_level(video_path):
    """Get simple audio level analysis using ffmpeg volumedetect filter."""
    try:
        print("Analyzing current audio level...")
        
        # Use volumedetect filter to get audio statistics
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-af", "volumedetect",
            "-f", "null",
            "-"
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = result.stdout
        
        # Parse the output to extract mean volume
        mean_volume = None
        max_volume = None
        
        for line in output.split('\n'):
            if 'mean_volume:' in line:
                try:
                    mean_volume = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                except:
                    pass
            elif 'max_volume:' in line:
                try:
                    max_volume = float(line.split('max_volume:')[1].split('dB')[0].strip())
                except:
                    pass
        
        print(f"Current audio levels - Mean: {mean_volume}dB, Max: {max_volume}dB")
        return mean_volume, max_volume
        
    except Exception as e:
        print(f"Audio analysis failed: {e}")
        return None, None

def calculate_volume_boost(mean_volume, max_volume, target_boost_db=6.0, max_boost_db=15.0):
    """Calculate how much to boost the volume."""
    
    # If we couldn't analyze the audio, apply a reasonable default boost
    if mean_volume is None:
        print("Couldn't analyze audio, applying default 2x boost (6dB)")
        return 2.0
    
    # Calculate needed boost to reach a reasonable level
    # Most videos should be around -12dB to -6dB mean volume for good loudness
    target_mean_volume = -12.0
    
    if mean_volume < target_mean_volume:
        # Calculate how much boost we need
        needed_boost_db = target_mean_volume - mean_volume
        
        # Cap the boost to prevent distortion
        boost_db = min(needed_boost_db, max_boost_db)
        
        # Make sure we apply at least the minimum target boost
        boost_db = max(boost_db, target_boost_db)
        
        # Convert dB to amplitude ratio: ratio = 10^(dB/20)
        boost_factor = 10 ** (boost_db / 20)
        
        print(f"Calculated boost: {boost_db:.1f}dB (factor: {boost_factor:.2f}x)")
        return boost_factor
    else:
        print(f"Audio is already loud enough (mean: {mean_volume:.1f}dB)")
        return 1.0

def process_video(input_path, output_path=None, custom_boost=None, normalize=False):
    """Process the video file to increase audio volume."""
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return False
    
    # Generate output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        output_dir = input_file.parent
        output_path = output_dir / f"louder_{input_file.name}"
    
    # Get video duration for progress bar
    duration = get_video_duration(input_path)
    
    if custom_boost:
        # Use custom boost factor
        boost_factor = custom_boost
        print(f"Using custom boost factor: {boost_factor}x")
    elif normalize:
        # Use audio normalization
        boost_factor = None
        print("Using audio normalization instead of boost")
    else:
        # Analyze audio and calculate boost
        mean_volume, max_volume = get_simple_audio_level(input_path)
        boost_factor = calculate_volume_boost(mean_volume, max_volume)
    
    # Prepare ffmpeg command
    if normalize:
        # Use loudnorm filter for professional audio normalization
        print("Applying audio normalization...")
        audio_filter = "loudnorm=I=-16:TP=-1.5:LRA=11"
    elif boost_factor <= 1.01:
        print("No volume boost needed. Copying file...")
        # Just copy the file
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-c", "copy",
            "-y", str(output_path)
        ]
    else:
        print(f"Boosting audio volume by {boost_factor:.2f}x...")
        audio_filter = f"volume={boost_factor}"
    
    # Build ffmpeg command
    if not (boost_factor <= 1.01 and not normalize):
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-af", audio_filter,
            "-c:v", "copy",  # Copy video without re-encoding
            "-c:a", "aac",   # Re-encode audio to apply filter
            "-b:a", "192k",  # Good audio quality
            "-y", str(output_path)
        ]
    
    # Add progress reporting
    cmd += ["-progress", "pipe:1"]
    
    try:
        # Start the ffmpeg process
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Set up progress bar
        pbar = tqdm(total=100, desc="Processing video", unit="%", ncols=80)
        
        # Track progress
        current_time = 0
        
        # Read output line by line for progress updates
        while True:
            line = process.stdout.readline()
            if not line:
                break
                
            if line.startswith("out_time_ms="):
                try:
                    # Extract time in milliseconds
                    time_ms = int(line.split("=")[1])
                    current_time = time_ms / 1000000  # Convert to seconds
                    
                    # Update progress bar
                    if duration > 0:
                        progress = min(int((current_time / duration) * 100), 100)
                        pbar.n = progress
                        pbar.refresh()
                except:
                    pass
        
        # Close the progress bar
        pbar.close()
        
        # Wait for the process to complete
        return_code = process.wait()
        
        if return_code == 0:
            # Verify output file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
                print(f"✅ Success! Output saved to '{output_path}' ({file_size:.1f} MB)")
                
                # Show before/after comparison
                if not custom_boost and not normalize:
                    print("\n📊 Volume Analysis:")
                    print(f"   Original mean volume: {mean_volume:.1f}dB" if mean_volume else "   Original: Could not analyze")
                    print(f"   Applied boost: {boost_factor:.2f}x ({20 * np.log10(boost_factor):.1f}dB)" if boost_factor > 1 else "   No boost applied")
                
                return True
            else:
                print("❌ Error: Output file was not created.")
                return False
        else:
            # Get error output
            stderr_output = process.stderr.read()
            print(f"❌ Error processing video: {stderr_output}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return False

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Make video audio louder with smart volume analysis.")
    parser.add_argument("input_file", help="Path to the input video file")
    parser.add_argument("-o", "--output", help="Path to the output file (default: 'louder_' + input filename)")
    parser.add_argument("-b", "--boost", type=float, 
                        help="Custom boost factor (e.g., 2.0 for 2x louder, 3.0 for 3x louder)")
    parser.add_argument("-n", "--normalize", action="store_true",
                        help="Use professional audio normalization instead of simple boost")
    parser.add_argument("--test", action="store_true",
                        help="Only analyze audio levels without processing")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if ffmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: ffmpeg is not installed or not in PATH.")
        print("Please install ffmpeg: https://ffmpeg.org/download.html")
        return 1
    
    # Install Python dependencies
    install_dependencies()
    
    # Test mode - just analyze
    if args.test:
        print(f"🔍 Analyzing '{args.input_file}'...")
        mean_volume, max_volume = get_simple_audio_level(args.input_file)
        if mean_volume is not None:
            boost = calculate_volume_boost(mean_volume, max_volume)
            print(f"\n📊 Analysis Results:")
            print(f"   Current mean volume: {mean_volume:.1f}dB")
            print(f"   Current max volume: {max_volume:.1f}dB")
            print(f"   Recommended boost: {boost:.2f}x ({20 * np.log10(boost):.1f}dB)")
        else:
            print("❌ Could not analyze audio levels")
        return 0
    
    # Validate boost parameter
    if args.boost and args.boost < 0.1:
        print("❌ Error: Boost factor must be at least 0.1")
        return 1
    
    if args.boost and args.boost > 10:
        print("⚠️  Warning: Boost factor > 10x may cause severe distortion")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return 1
    
    # Process the video
    print(f"🎬 Processing '{args.input_file}'...")
    success = process_video(args.input_file, args.output, args.boost, args.normalize)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())