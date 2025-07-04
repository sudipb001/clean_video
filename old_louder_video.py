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
    subprocess.run([sys.executable, "-m", "pip", "install", "ffmpeg-python", "tqdm"], 
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
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])

def get_audio_stats(video_path):
    """Get audio statistics from the video file using ffmpeg."""
    try:
        # Use ffprobe to get audio statistics in JSON format
        cmd = [
            "ffprobe", 
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "frame_tags=lavfi.r128.I",
            "-of", "json",
            "-f", "lavfi",
            f"movie='{video_path}',ebur128=metadata=1"
        ]
        
        print("Analyzing audio levels...")
        with tqdm(total=100, desc="Audio analysis", unit="%") as pbar:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            pbar.update(100)  # Complete the progress bar
            
        data = json.loads(result.stdout)
        
        # Extract loudness values
        loudness_values = []
        for frame in data.get("frames", []):
            if "tags" in frame and "lavfi.r128.I" in frame["tags"]:
                loudness_values.append(float(frame["tags"]["lavfi.r128.I"]))
        
        if not loudness_values:
            return None
        
        return {
            "mean_loudness": np.mean(loudness_values),
            "min_loudness": min(loudness_values),
            "max_loudness": max(loudness_values)
        }
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return None

def calculate_volume_adjustment(audio_stats, target_loudness=-16.0):
    """Calculate the volume adjustment needed based on audio statistics."""
    if not audio_stats:
        # If we couldn't analyze, apply a small default increase
        return 1.5
    
    # Calculate how much we need to increase to reach target loudness
    # EBU R128 integrated loudness (I) is in LUFS
    current_loudness = audio_stats["mean_loudness"]
    
    # If loudness is already good, don't adjust
    if current_loudness >= target_loudness:
        return 1.0
    
    # Calculate the gain in dB needed
    db_increase = target_loudness - current_loudness
    
    # Convert dB to amplitude ratio: ratio = 10^(dB/20)
    volume_factor = 10 ** (db_increase / 20)
    
    # Cap the maximum increase to prevent distortion
    return min(volume_factor, 3.0)

def process_video(input_path, output_path=None, target_loudness=-16.0):
    """Process the video file to adjust its audio volume if needed."""
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return False
    
    # Generate output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        output_dir = input_file.parent
        output_path = output_dir / f"louder_{input_file.name}"
    
    # Get audio statistics
    print(f"Analyzing audio in '{input_path}'...")
    audio_stats = get_audio_stats(input_path)
    
    # Calculate volume adjustment
    volume_factor = calculate_volume_adjustment(audio_stats, target_loudness)
    
    # Get video duration for progress bar
    duration = get_video_duration(input_path)
    
    if volume_factor <= 1.01:  # Small buffer for floating point comparison
        print("Audio volume is already good. No adjustment needed.")
        print(f"Creating output file '{output_path}'...")
        
        # Set up ffmpeg command to copy the file
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-c", "copy",
            "-y", str(output_path)
        ]
    else:
        print(f"Increasing audio volume by factor of {volume_factor:.2f}...")
        
        # Set up ffmpeg command to adjust volume
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-filter:a", f"volume={volume_factor}",
            "-c:v", "copy",  # Copy video stream without re-encoding
            "-y", str(output_path)
        ]
    
    # Add progress tracking
    cmd += ["-progress", "pipe:1"]
    
    try:
        # Start the ffmpeg process
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Set up progress bar
        pbar = tqdm(total=100, desc="Processing video", unit="%")
        
        # Track time and progress
        current_time = 0
        
        # Read output line by line for progress updates
        for line in process.stdout:
            if line.startswith("out_time_ms="):
                try:
                    # Extract time in milliseconds
                    time_ms = int(line.split("=")[1])
                    current_time = time_ms / 1000000  # Convert to seconds
                    
                    # Update progress bar
                    progress = min(int((current_time / duration) * 100), 100)
                    pbar.update(progress - pbar.n)  # Update to current position
                except:
                    pass
        
        # Close the progress bar
        pbar.close()
        
        # Wait for the process to complete
        process.wait()
        
        if process.returncode == 0:
            print(f"Processing complete. Output saved to '{output_path}'")
            return True
        else:
            print("Error processing video. Check ffmpeg installation.")
            return False
            
    except Exception as e:
        print(f"Error processing video: {e}")
        return False

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Automatically increase video audio volume if needed.")
    parser.add_argument("input_file", help="Path to the input MP4 video file")
    parser.add_argument("-o", "--output", help="Path to the output MP4 file (default: 'louder_' + input filename)")
    parser.add_argument("-t", "--target", type=float, default=-16.0, 
                        help="Target loudness in LUFS (default: -16.0)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if ffmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH.")
        print("Please install ffmpeg: https://ffmpeg.org/download.html")
        return
    
    # Install Python dependencies
    install_dependencies()
    
    # Process the video
    process_video(args.input_file, args.output, args.target)

if __name__ == "__main__":
    main()