#!/usr/bin/env python3
"""
MP4 Video Compressor

This script compresses MP4 videos while maintaining quality as much as possible.
It uses FFmpeg with optimized settings to reduce file size effectively.
Progress bars show compression status in real-time.

Usage:
    python video_compressor.py input.mp4 [output.mp4]
    
If no output filename is specified, it will create a file named "compressed_input.mp4"
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
import time
import re
import threading
from tqdm import tqdm

def get_video_info(video_path):
    """Get video information using FFmpeg."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error getting video info: {e}")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error parsing video info")
        sys.exit(1)

def get_video_duration(video_info):
    """Extract video duration from the video info."""
    try:
        return float(video_info['format']['duration'])
    except (KeyError, ValueError):
        print("Could not determine video duration")
        return 0

def get_video_bitrate(video_info):
    """Extract video bitrate from the video info."""
    try:
        for stream in video_info['streams']:
            if stream['codec_type'] == 'video':
                if 'bit_rate' in stream:
                    return int(stream['bit_rate'])
        # If bit_rate is not in stream, try format
        if 'bit_rate' in video_info['format']:
            return int(video_info['format']['bit_rate'])
    except (KeyError, ValueError):
        pass
    
    print("Could not determine video bitrate")
    return 0

def get_audio_bitrate(video_info):
    """Extract audio bitrate from the video info."""
    try:
        for stream in video_info['streams']:
            if stream['codec_type'] == 'audio':
                if 'bit_rate' in stream:
                    return int(stream['bit_rate'])
    except (KeyError, ValueError):
        pass
    
    print("Could not determine audio bitrate, will use default")
    return 128000  # Default to 128k

def get_video_resolution(video_info):
    """Extract video resolution from the video info."""
    try:
        for stream in video_info['streams']:
            if stream['codec_type'] == 'video':
                return (int(stream['width']), int(stream['height']))
    except (KeyError, ValueError):
        pass
    
    print("Could not determine video resolution")
    return (0, 0)

def monitor_ffmpeg_progress(process, duration, desc="Compressing video"):
    """
    Monitor FFmpeg progress and update a progress bar.
    
    Args:
        process: Subprocess Popen object
        duration: Video duration in seconds
        desc: Description for the progress bar
    """
    progress_bar = tqdm(total=100, desc=desc, unit="%")
    
    # Pattern to extract time from FFmpeg output
    time_pattern = re.compile(r"time=(\d+):(\d+):(\d+.\d+)")
    
    # Read from stderr line by line
    while process.poll() is None:
        if process.stderr:
            line = process.stderr.readline()
            if line:
                line = line.strip()
                # Match the time pattern
                match = time_pattern.search(line)
                if match:
                    hours, minutes, seconds = match.groups()
                    current_time = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
                    # Calculate progress percentage
                    progress = min(int((current_time / duration) * 100), 100)
                    # Update the progress bar
                    progress_bar.n = progress
                    progress_bar.refresh()
    
    # Ensure the progress bar completes
    progress_bar.n = 100
    progress_bar.refresh()
    progress_bar.close()

def compress_video_h265(input_path, output_path, crf=28, preset="medium"):
    """
    Compress video using H.265 codec with CRF (Constant Rate Factor).
    This method provides good quality with smaller file size.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output compressed video
        crf: Constant Rate Factor (18-28 recommended, lower = better quality)
        preset: Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    """
    # Get video information for progress monitoring
    video_info = get_video_info(input_path)
    duration = get_video_duration(video_info)
    
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx265',      # Use H.265 codec
        '-crf', str(crf),       # Constant Rate Factor 
        '-preset', preset,      # Preset for encoding speed/efficiency
        '-c:a', 'aac',          # Audio codec
        '-b:a', '128k',         # Audio bitrate
        '-tag:v', 'hvc1',       # Add tag for better compatibility
        '-y',                   # Overwrite output file if it exists
        '-progress', 'pipe:1',  # Output progress information
        output_path
    ]
    
    print(f"Compressing video with H.265 (CRF: {crf}, Preset: {preset})...")
    
    # Start the FFmpeg process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitor progress in a separate thread
    monitor_thread = threading.Thread(
        target=monitor_ffmpeg_progress,
        args=(process, duration, f"Compressing with H.265 (CRF: {crf})"),
        daemon=True
    )
    monitor_thread.start()
    
    # Wait for the process to complete
    process.wait()
    monitor_thread.join()
    
    # Check if the process exited successfully
    if process.returncode != 0:
        print(f"Error during compression: FFmpeg exited with code {process.returncode}")
        sys.exit(1)

def compress_video_h264(input_path, output_path, crf=23, preset="medium"):
    """
    Compress video using H.264 codec with CRF (Constant Rate Factor).
    This method provides good compatibility with most devices.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output compressed video
        crf: Constant Rate Factor (18-28 recommended, lower = better quality)
        preset: Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    """
    # Get video information for progress monitoring
    video_info = get_video_info(input_path)
    duration = get_video_duration(video_info)
    
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',      # Use H.264 codec
        '-crf', str(crf),       # Constant Rate Factor
        '-preset', preset,      # Preset for encoding speed/efficiency
        '-c:a', 'aac',          # Audio codec
        '-b:a', '128k',         # Audio bitrate
        '-movflags', '+faststart',  # Optimize for web streaming
        '-y',                   # Overwrite output file if it exists
        output_path
    ]
    
    print(f"Compressing video with H.264 (CRF: {crf}, Preset: {preset})...")
    
    # Start the FFmpeg process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitor progress in a separate thread
    monitor_thread = threading.Thread(
        target=monitor_ffmpeg_progress,
        args=(process, duration, f"Compressing with H.264 (CRF: {crf})"),
        daemon=True
    )
    monitor_thread.start()
    
    # Wait for the process to complete
    process.wait()
    monitor_thread.join()
    
    # Check if the process exited successfully
    if process.returncode != 0:
        print(f"Error during compression: FFmpeg exited with code {process.returncode}")
        sys.exit(1)

def compress_video_to_target_size(input_path, output_path, target_size_mb, two_pass=True):
    """
    Compress video to reach a target file size in megabytes.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output compressed video
        target_size_mb: Target size in megabytes
        two_pass: Whether to use two-pass encoding for better quality
    """
    # Get video information
    video_info = get_video_info(input_path)
    duration = get_video_duration(video_info)
    
    if duration == 0:
        print("Could not determine video duration, aborting target size compression")
        return False
    
    # Convert target size from MB to bits
    target_size_bits = target_size_mb * 8 * 1024 * 1024
    
    # Calculate target bitrate (90% for video, 10% for audio)
    # Subtract 5% for container overhead
    total_bitrate = int(target_size_bits / duration * 0.95)
    
    # Minimum audio bitrate (128 kbps)
    min_audio_bitrate = 128000
    max_audio_bitrate = 256000
    
    # Get original audio bitrate
    audio_bitrate = get_audio_bitrate(video_info)
    
    # Adjust audio bitrate
    if audio_bitrate > max_audio_bitrate:
        audio_bitrate = max_audio_bitrate
    elif audio_bitrate < min_audio_bitrate:
        audio_bitrate = min_audio_bitrate
    
    # Ensure audio bitrate doesn't take more than 10% of total
    if audio_bitrate > total_bitrate * 0.1:
        audio_bitrate = int(total_bitrate * 0.1)
        if audio_bitrate < min_audio_bitrate:
            audio_bitrate = min_audio_bitrate
            
    # Calculate video bitrate
    video_bitrate = total_bitrate - audio_bitrate
    
    # Ensure minimum video bitrate
    min_video_bitrate = 100000  # 100 kbps
    if video_bitrate < min_video_bitrate:
        print(f"Warning: Target size too small for acceptable quality. Video bitrate would be {video_bitrate/1000} kbps")
        video_bitrate = min_video_bitrate
    
    print(f"Target size: {target_size_mb} MB")
    print(f"Video duration: {duration:.2f} seconds")
    print(f"Total bitrate: {total_bitrate/1000:.2f} kbps")
    print(f"Video bitrate: {video_bitrate/1000:.2f} kbps")
    print(f"Audio bitrate: {audio_bitrate/1000:.2f} kbps")
    
    if two_pass:
        # First pass
        cmd_pass1 = [
            'ffmpeg',
            '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-b:v', f"{video_bitrate}",
            '-pass', '1',
            '-an',  # No audio in first pass
            '-f', 'mp4',
            '-movflags', '+faststart',
        ]
        
        # Use NUL on Windows, /dev/null on Unix
        if os.name == 'nt':
            cmd_pass1.append('NUL')
        else:
            cmd_pass1.append('/dev/null')
        
        # Second pass
        cmd_pass2 = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-b:v', f"{video_bitrate}",
            '-pass', '2',
            '-c:a', 'aac',
            '-b:a', f"{audio_bitrate}",
            '-movflags', '+faststart',
            '-y',
            output_path
        ]
        
        print("Performing two-pass encoding...")
        
        # First pass with progress bar
        print("Pass 1...")
        
        # Start the FFmpeg process for first pass
        process_pass1 = subprocess.Popen(
            cmd_pass1,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor progress in a separate thread
        monitor_thread_pass1 = threading.Thread(
            target=monitor_ffmpeg_progress,
            args=(process_pass1, duration, "Pass 1 (Analysis)"),
            daemon=True
        )
        monitor_thread_pass1.start()
        
        # Wait for the process to complete
        process_pass1.wait()
        monitor_thread_pass1.join()
        
        # Check if the process exited successfully
        if process_pass1.returncode != 0:
            print(f"Error during first pass: FFmpeg exited with code {process_pass1.returncode}")
            return False
        
        # Second pass with progress bar
        print("Pass 2...")
        
        # Start the FFmpeg process for second pass
        process_pass2 = subprocess.Popen(
            cmd_pass2,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor progress in a separate thread
        monitor_thread_pass2 = threading.Thread(
            target=monitor_ffmpeg_progress,
            args=(process_pass2, duration, "Pass 2 (Encoding)"),
            daemon=True
        )
        monitor_thread_pass2.start()
        
        # Wait for the process to complete
        process_pass2.wait()
        monitor_thread_pass2.join()
        
        # Check if the process exited successfully
        if process_pass2.returncode != 0:
            print(f"Error during second pass: FFmpeg exited with code {process_pass2.returncode}")
            return False
        
        # Clean up pass log files
        if os.path.exists("ffmpeg2pass-0.log"):
            os.remove("ffmpeg2pass-0.log")
        if os.path.exists("ffmpeg2pass-0.log.mbtree"):
            os.remove("ffmpeg2pass-0.log.mbtree")
    else:
        # Single pass
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-b:v', f"{video_bitrate}",
            '-c:a', 'aac',
            '-b:a', f"{audio_bitrate}",
            '-movflags', '+faststart',
            '-y',
            output_path
        ]
        
        print("Performing single-pass encoding...")
        
        # Start the FFmpeg process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor progress in a separate thread
        monitor_thread = threading.Thread(
            target=monitor_ffmpeg_progress,
            args=(process, duration, "Single-pass encoding"),
            daemon=True
        )
        monitor_thread.start()
        
        # Wait for the process to complete
        process.wait()
        monitor_thread.join()
        
        # Check if the process exited successfully
        if process.returncode != 0:
            print(f"Error during compression: FFmpeg exited with code {process.returncode}")
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Compress MP4 video files while maintaining quality')
    parser.add_argument('input', help='Input video file')
    parser.add_argument('output', nargs='?', help='Output video file (optional)')
    parser.add_argument('--codec', choices=['h264', 'h265'], default='h264', 
                        help='Video codec to use (default: h264)')
    parser.add_argument('--crf', type=int, default=23, 
                        help='Constant Rate Factor: lower = better quality, higher = smaller size (default: 23)')
    parser.add_argument('--preset', choices=['ultrafast', 'superfast', 'veryfast', 'faster', 
                                            'fast', 'medium', 'slow', 'slower', 'veryslow'], 
                        default='medium', help='Encoding preset (default: medium)')
    parser.add_argument('--target-size', type=float, help='Target size in MB')
    parser.add_argument('--two-pass', action='store_true', help='Use two-pass encoding for target size (better quality)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Check if FFmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: FFmpeg is not installed or not in PATH. Please install FFmpeg.")
        sys.exit(1)
    
    # Generate output filename if not provided
    if args.output is None:
        input_path = Path(args.input)
        output_filename = f"compressed_{input_path.name}"
        args.output = str(input_path.parent / output_filename)
    
    # Get file size before compression
    initial_size = os.path.getsize(args.input) / (1024 * 1024)  # Convert to MB
    print(f"Original file size: {initial_size:.2f} MB")
    
    # Get video information
    video_info = get_video_info(args.input)
    resolution = get_video_resolution(video_info)
    print(f"Resolution: {resolution[0]}x{resolution[1]}")
    
    # Display overall progress header
    print("\n" + "="*60)
    print(f"Starting video compression for {args.input}")
    print(f"Output will be saved to {args.output}")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    # Create an overall progress bar for the entire process
    with tqdm(total=100, desc="Overall progress", unit="%") as overall_pbar:
        # Initial analysis - 10% of overall progress
        overall_pbar.update(10)
        
        # Compress based on method - 80% of overall progress
        success = True
        if args.target_size:
            success = compress_video_to_target_size(args.input, args.output, args.target_size, args.two_pass)
            if not success:
                print("Using CRF method as fallback")
                if args.codec == 'h265':
                    compress_video_h265(args.input, args.output, args.crf, args.preset)
                else:
                    compress_video_h264(args.input, args.output, args.crf, args.preset)
        else:
            if args.codec == 'h265':
                compress_video_h265(args.input, args.output, args.crf, args.preset)
            else:
                compress_video_h264(args.input, args.output, args.crf, args.preset)
        
        # Update overall progress to 90%
        overall_pbar.update(80)
        
        # Final processing - 10% of overall progress
        time.sleep(0.5)  # Just for visual effect
        overall_pbar.update(10)
    
    # Calculate compression time
    elapsed_time = time.time() - start_time
    
    # Final results display
    print("\n" + "="*60)
    print(f"Compression completed in {elapsed_time:.2f} seconds")
    
    # Get file size after compression
    if os.path.exists(args.output):
        final_size = os.path.getsize(args.output) / (1024 * 1024)  # Convert to MB
        reduction = (1 - (final_size / initial_size)) * 100
        print(f"Original file size: {initial_size:.2f} MB")
        print(f"Compressed file size: {final_size:.2f} MB")
        print(f"Size reduction: {reduction:.2f}%")
        
        # Visual size comparison
        print("\nSize comparison:")
        bar_length = 40
        original_bar = "█" * bar_length
        compressed_bar_length = int((final_size / initial_size) * bar_length)
        compressed_bar = "█" * compressed_bar_length
        
        print(f"Original:   {original_bar} {initial_size:.2f} MB")
        print(f"Compressed: {compressed_bar} {final_size:.2f} MB")
    else:
        print("Compression failed - output file not found")
    
    print("="*60)

if __name__ == "__main__":
    main()