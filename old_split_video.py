#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def analyze_video_content(video_path, num_parts=2):
    """Analyze video content to find optimal splitting points using OpenCV."""
    print(f"Analyzing video content to find {num_parts-1} optimal splitting points...")
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if frame_count <= 0 or fps <= 0:
        raise ValueError("Invalid video properties detected")
    
    # Calculate frame differences to detect scenes
    prev_frame = None
    frame_diffs = []
    
    # Sample frames for efficiency (analyze every 5th frame)
    sample_rate = 5
    frame_positions = []
    
    # Create progress bar
    total_frames_to_analyze = frame_count // sample_rate
    progress_bar = tqdm(total=total_frames_to_analyze, desc="Analyzing frames", unit="frames")
    
    for i in range(0, frame_count, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # Calculate absolute difference between frames
            diff = cv2.absdiff(gray, prev_frame)
            diff_sum = np.sum(diff)
            frame_diffs.append(diff_sum)
            frame_positions.append(i)
        
        prev_frame = gray
        progress_bar.update(1)
    
    progress_bar.close()
    cap.release()
    
    # Get video duration as fallback
    duration = get_video_duration(video_path)
    
    # If we couldn't analyze frames, just divide the video into equal parts
    if not frame_diffs:
        split_times = [duration * i / num_parts for i in range(1, num_parts)]
        return split_times
    
    # For multiple splits, we'll divide the video into regions and find the best 
    # scene change in each region
    split_times = []
    
    if num_parts <= 1:
        return []
    
    # Get number of frames to analyze per segment
    section_size = len(frame_diffs) // num_parts
    
    # Find the best splitting point in each section
    for i in range(1, num_parts):
        # Define the section to look for a scene change
        start_idx = (section_size * i) - (section_size // 4)
        end_idx = (section_size * i) + (section_size // 4)
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(frame_diffs), end_idx)
        
        if start_idx >= end_idx:
            # Fallback to equal division
            split_time = duration * i / num_parts
        else:
            # Find frame with maximum difference in the section
            section = frame_diffs[start_idx:end_idx]
            max_diff_idx = start_idx + np.argmax(section)
            split_frame = frame_positions[max_diff_idx]
            
            # Convert frame number to timestamp
            split_time = split_frame / fps
        
        split_times.append(split_time)
        print(f"Split point {i} found at {split_time:.2f} seconds")
    
    return split_times


def get_video_duration(video_path):
    """Get video duration using ffprobe."""
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'json', 
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        return duration
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError) as e:
        print(f"Error getting video duration: {str(e)}")
        # Fallback to OpenCV if ffprobe fails
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count / fps


def run_ffmpeg_with_progress(cmd, desc):
    """Run FFmpeg command with a progress bar."""
    # Create process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Set up progress bar (indeterminate initially)
    pbar = tqdm(desc=desc, unit='%')
    
    # Track duration and start time to estimate progress
    duration = None
    start_time = time.time()
    
    # Read stderr line by line to update progress
    for line in process.stderr:
        # Try to find time information
        if "Duration" in line and duration is None:
            time_str = line.split("Duration: ")[1].split(",")[0].strip()
            h, m, s = map(float, time_str.split(':'))
            duration = h * 3600 + m * 60 + s
            pbar.total = 100  # Set total to 100%
        
        # Look for time progress
        if "time=" in line and duration:
            try:
                time_str = line.split("time=")[1].split(" ")[0].strip()
                if ":" in time_str:
                    h, m, s = map(float, time_str.split(':'))
                    current = h * 3600 + m * 60 + s
                else:
                    current = float(time_str)
                    
                progress = min(int((current / duration) * 100), 100)
                pbar.n = progress
                pbar.refresh()
            except (ValueError, IndexError):
                pass
    
    # Close progress bar
    pbar.close()
    
    # Get return code
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.SubprocessError(f"FFmpeg process failed with return code {return_code}")


def split_video(video_path, split_times, fast=False):
    """Split the video into multiple parts using ffmpeg."""
    try:
        print(f"Splitting video at {len(split_times)} points: {', '.join([f'{t:.2f}s' for t in split_times])}")
        
        # Get original video duration
        duration = get_video_duration(video_path)
        
        # Validate split times
        valid_split_times = []
        for t in split_times:
            if 0 < t < duration:
                valid_split_times.append(t)
            else:
                print(f"Invalid split time {t:.2f}s. Skipping.")
        
        if not valid_split_times and len(split_times) > 0:
            # Fallback to equal parts
            print("No valid split times. Using equal division.")
            num_parts = len(split_times) + 1
            valid_split_times = [duration * i / num_parts for i in range(1, num_parts)]
        
        # Sort split times in ascending order
        valid_split_times.sort()
        
        # Add start and end times to create a complete list of segments
        all_split_points = [0] + valid_split_times + [duration]
        
        # Generate output filenames
        path = Path(video_path)
        output_dir = path.parent
        base_name = path.stem
        extension = path.suffix
        
        output_files = []
        
        # Get video info to preserve quality settings
        video_info_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate',
            '-of', 'json',
            str(video_path)
        ]
        
        result = subprocess.run(video_info_cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(result.stdout)
        
        # Extract video properties
        width = video_info['streams'][0]['width']
        height = video_info['streams'][0]['height']
        fps_frac = video_info['streams'][0]['r_frame_rate'].split('/')
        fps = float(fps_frac[0]) / float(fps_frac[1])
        
        # Process each segment
        for i in range(len(all_split_points) - 1):
            part_num = i + 1
            start_time = all_split_points[i]
            end_time = all_split_points[i + 1]
            
            output_path = output_dir / f"Part{part_num}_{base_name}{extension}"
            output_files.append(output_path)
            
            # Determine ffmpeg command based on fast option
            if fast:
                # Faster option using stream copy (may cause sync issues)
                cmd = [
                    'ffmpeg',
                    '-i', str(video_path),
                    '-ss', str(start_time),
                    '-to', str(end_time),
                    '-c', 'copy',
                    '-y',
                    str(output_path)
                ]
            else:
                # Higher quality option with re-encoding
                cmd = [
                    'ffmpeg',
                    '-ss', str(start_time),
                    '-i', str(video_path),
                    '-to', str(end_time - start_time),
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '18',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-y',
                    str(output_path)
                ]
            
            # Run FFmpeg command with progress bar
            print(f"Creating part {part_num} of the video (from {start_time:.2f}s to {end_time:.2f}s)...")
            run_ffmpeg_with_progress(cmd, f"Creating part {part_num}")
        
        print(f"Successfully created {len(output_files)} video parts:")
        for i, file_path in enumerate(output_files, 1):
            print(f"{i}. {file_path}")
        
        return True
        
    except subprocess.SubprocessError as e:
        print(f"Error splitting video: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Split an MP4 video into multiple parts using AI scene detection.")
    parser.add_argument("video_path", type=str, help="Path to the input MP4 video file")
    parser.add_argument("--parts", type=int, default=2, help="Number of parts to split the video into (default: 2)")
    parser.add_argument("--fast", action="store_true", help="Use faster copy mode (may cause sync issues at split points)")
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    num_parts = max(1, args.parts)  # Ensure at least 1 part
    
    # Validate input
    if not video_path.exists():
        print(f"Error: File {video_path} does not exist")
        return 1
        
    if video_path.suffix.lower() != ".mp4":
        print(f"Warning: File {video_path} is not an MP4 file. Results may be unexpected.")
    
    # For single part, just copy the file
    if num_parts == 1:
        output_path = video_path.parent / f"Part1_{video_path.name}"
        print(f"Only one part requested. Copying to {output_path}")
        import shutil
        shutil.copy2(video_path, output_path)
        return 0
    
    # Analyze video to find the split points
    try:
        split_times = analyze_video_content(video_path, num_parts)
        success = split_video(video_path, split_times, args.fast)
        return 0 if success else 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())