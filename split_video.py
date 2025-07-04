# # Split mp4 videos without using mmpeg
# # Basic usage - much faster now!
# Install Python packages (already done)
# pip install opencv-python numpy tqdm
# # Install FFmpeg for audio support
# # Mac:
# brew install ffmpeg
# # Windows:
# # Download from https://ffmpeg.org/download.html
# # Linux:
# sudo apt update && sudo apt install ffmpeg
#
# # Check if everything is working
# python3 split_video.py "DevOps076 CI-CD with Kubernetes.mp4" --info
#
# # Split with audio (requires ffmpeg)
# python3 split_video.py "DevOps076 CI-CD with Kubernetes.mp4" --parts 5
#
# # Split without parallel processing
# python3 split_video.py "DevOps076 CI-CD with Kubernetes.mp4" --parts 5 --no-parallel


#!/usr/bin/env python3
import argparse
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import tempfile
import shutil
import uuid

import cv2
import numpy as np
from tqdm import tqdm


def check_ffmpeg():
    """Check if ffmpeg is available for audio processing."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


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
    
    # Adaptive sample rate based on video length
    if frame_count > 100000:  # ~55 minutes at 30fps
        sample_rate = 30
    elif frame_count > 50000:  # ~28 minutes at 30fps
        sample_rate = 15
    elif frame_count > 20000:  # ~11 minutes at 30fps
        sample_rate = 10
    else:
        sample_rate = 5
    
    # Calculate frame differences to detect scenes
    prev_frame = None
    frame_diffs = []
    frame_positions = []
    
    # Create progress bar
    total_frames_to_analyze = frame_count // sample_rate
    progress_bar = tqdm(total=total_frames_to_analyze, desc="Analyzing frames", unit="frames")
    
    # Use smaller frame size for faster processing
    target_width = 320
    
    for i in range(0, frame_count, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > target_width:
            target_height = int(height * target_width / width)
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # Use histogram comparison for more robust scene detection
            hist_curr = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist_prev = cv2.calcHist([prev_frame], [0], None, [64], [0, 256])
            
            # Calculate histogram correlation
            correlation = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_CORREL)
            # Convert to difference score
            diff_score = (1.0 - correlation) * 1000000
            
            frame_diffs.append(diff_score)
            frame_positions.append(i)
        
        prev_frame = gray
        progress_bar.update(1)
    
    progress_bar.close()
    cap.release()
    
    # Get video duration
    duration = frame_count / fps
    
    # If we couldn't analyze frames, just divide the video into equal parts
    if not frame_diffs:
        split_times = [duration * i / num_parts for i in range(1, num_parts)]
        return split_times
    
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


def get_video_info(video_path):
    """Get comprehensive video information using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get codec information
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration,
        'codec': codec
    }


def split_video_segment_ffmpeg_only(args):
    """Split video segment using FFmpeg only - more reliable for audio/video sync."""
    video_path, start_time, end_time, output_path, part_num = args
    
    try:
        duration = end_time - start_time
        
        # Use FFmpeg for the entire process to ensure perfect audio/video sync
        cmd = [
            'ffmpeg',
            '-i', str(video_path),           # Input file
            '-ss', str(start_time),          # Start time
            '-t', str(duration),             # Duration (not end time)
            '-c:v', 'libx264',               # Video codec
            '-preset', 'fast',               # Fast encoding
            '-crf', '23',                    # Good quality/speed balance
            '-c:a', 'aac',                   # Audio codec
            '-b:a', '128k',                  # Audio bitrate
            '-avoid_negative_ts', 'make_zero', # Handle timing issues
            '-y',                            # Overwrite output
            str(output_path)
        ]
        
        # Run FFmpeg with progress tracking
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Track progress
        progress_bar = tqdm(desc=f"Creating part {part_num}", unit="%", leave=False)
        
        stderr_lines = []
        for line in process.stderr:
            stderr_lines.append(line.strip())
            # Update progress when we see time information
            if "time=" in line:
                try:
                    # Extract current time
                    time_part = line.split("time=")[1].split(" ")[0]
                    if ":" in time_part:
                        time_parts = time_part.split(":")
                        current_seconds = float(time_parts[0]) * 3600 + float(time_parts[1]) * 60 + float(time_parts[2])
                        progress_percent = min(int((current_seconds / duration) * 100), 100)
                        progress_bar.n = progress_percent
                        progress_bar.refresh()
                except:
                    pass
        
        progress_bar.close()
        
        return_code = process.wait()
        
        if return_code != 0:
            error_msg = "\n".join(stderr_lines[-10:])  # Last 10 lines of error
            return False, f"FFmpeg failed for part {part_num}: {error_msg}"
        
        # Verify the output file was created and has reasonable size
        if not output_path.exists():
            return False, f"Output file not created for part {part_num}"
        
        file_size = output_path.stat().st_size
        if file_size < 1024:  # Less than 1KB indicates a problem
            return False, f"Output file too small for part {part_num} ({file_size} bytes)"
        
        return True, f"Successfully created part {part_num} ({file_size / 1024 / 1024:.1f} MB)"
        
    except Exception as e:
        return False, f"Error processing part {part_num}: {str(e)}"


def split_video_segment_fast_mode(args):
    """Split video segment using stream copy (fastest but may have sync issues)."""
    video_path, start_time, end_time, output_path, part_num = args
    
    try:
        duration = end_time - start_time
        
        # Use stream copy for maximum speed
        cmd = [
            'ffmpeg',
            '-i', str(video_path),           # Input file
            '-ss', str(start_time),          # Start time
            '-t', str(duration),             # Duration
            '-c', 'copy',                    # Copy streams without re-encoding
            '-avoid_negative_ts', 'make_zero', # Handle timing issues
            '-y',                            # Overwrite output
            str(output_path)
        ]
        
        # Run FFmpeg
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        progress_bar = tqdm(desc=f"Copying part {part_num}", unit="%", leave=False)
        
        stderr_lines = []
        for line in process.stderr:
            stderr_lines.append(line.strip())
            if "time=" in line:
                progress_bar.update(1)
        
        progress_bar.close()
        
        return_code = process.wait()
        
        if return_code != 0:
            error_msg = "\n".join(stderr_lines[-5:])
            return False, f"FFmpeg copy failed for part {part_num}: {error_msg}"
        
        # Verify output
        if not output_path.exists():
            return False, f"Output file not created for part {part_num}"
        
        file_size = output_path.stat().st_size
        if file_size < 1024:
            return False, f"Output file too small for part {part_num}"
        
        return True, f"Successfully copied part {part_num} ({file_size / 1024 / 1024:.1f} MB)"
        
    except Exception as e:
        return False, f"Error copying part {part_num}: {str(e)}"


def split_video(video_path, split_times, parallel=True, fast_mode=False):
    """Split the video into multiple parts using FFmpeg for reliability."""
    try:
        print(f"Splitting video at {len(split_times)} points: {', '.join([f'{t:.2f}s' for t in split_times])}")
        
        # Check if ffmpeg is available
        if not check_ffmpeg():
            print("❌ FFmpeg not found. Please install FFmpeg to use this tool.")
            print("   Installation: https://ffmpeg.org/download.html")
            return False
        
        print("✅ FFmpeg detected - videos will include audio and video")
        
        # Get video information
        video_info = get_video_info(video_path)
        duration = video_info['duration']
        
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
        
        # Add start and end times to create segments
        all_split_times = [0.0] + valid_split_times + [duration]
        
        # Generate output filenames
        path = Path(video_path)
        output_dir = path.parent
        base_name = path.stem
        extension = '.mp4'
        
        output_files = []
        
        # Prepare segment processing arguments
        segment_args = []
        for i in range(len(all_split_times) - 1):
            part_num = i + 1
            start_time = all_split_times[i]
            end_time = all_split_times[i + 1]
            
            output_path = output_dir / f"Part{part_num}_{base_name}{extension}"
            output_files.append(output_path)
            
            segment_args.append((video_path, start_time, end_time, output_path, part_num))
        
        # Choose processing function
        process_func = split_video_segment_fast_mode if fast_mode else split_video_segment_ffmpeg_only
        mode_name = "fast copy" if fast_mode else "high quality"
        
        # Process segments
        if parallel and len(segment_args) > 1:
            # Use parallel processing for multiple segments
            max_workers = min(multiprocessing.cpu_count() // 2, len(segment_args), 3)  # Conservative worker count
            print(f"Processing {len(segment_args)} segments in parallel ({mode_name} mode) with {max_workers} workers...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_part = {executor.submit(process_func, args): args[4] for args in segment_args}
                
                # Wait for completion
                success_count = 0
                results = []
                for future in as_completed(future_to_part):
                    part_num = future_to_part[future]
                    try:
                        success, message = future.result()
                        results.append((part_num, success, message))
                        if success:
                            success_count += 1
                            print(f"✅ {message}")
                        else:
                            print(f"❌ {message}")
                    except Exception as e:
                        print(f"❌ Part {part_num} error: {str(e)}")
                        results.append((part_num, False, str(e)))
                
                # Sort results by part number for clean output
                results.sort(key=lambda x: x[0])
                
                if success_count == len(segment_args):
                    print(f"\n🎉 Successfully created {len(output_files)} video parts:")
                    for i, file_path in enumerate(output_files, 1):
                        if file_path.exists():
                            file_size = file_path.stat().st_size / (1024 * 1024)
                            print(f"{i}. {file_path.name} ({file_size:.1f} MB)")
                        else:
                            print(f"{i}. {file_path.name} (❌ File not found)")
                    return True
                else:
                    print(f"⚠️  Only {success_count}/{len(segment_args)} parts were created successfully")
                    # List which parts failed
                    for part_num, success, message in results:
                        if not success:
                            print(f"   Part {part_num}: {message}")
                    return False
        else:
            # Sequential processing
            print(f"Processing segments sequentially ({mode_name} mode)...")
            success_count = 0
            for args in segment_args:
                success, message = process_func(args)
                if success:
                    success_count += 1
                    print(f"✅ {message}")
                else:
                    print(f"❌ {message}")
            
            if success_count == len(segment_args):
                print(f"\n🎉 Successfully created {len(output_files)} video parts:")
                for i, file_path in enumerate(output_files, 1):
                    if file_path.exists():
                        file_size = file_path.stat().st_size / (1024 * 1024)
                        print(f"{i}. {file_path.name} ({file_size:.1f} MB)")
                    else:
                        print(f"{i}. {file_path.name} (❌ File not found)")
                return True
            else:
                print(f"⚠️  Only {success_count}/{len(segment_args)} parts were created successfully")
                return False
        
    except Exception as e:
        print(f"Error splitting video: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Split a video into multiple parts using AI scene detection with audio preservation.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--parts", type=int, default=2, help="Number of parts to split the video into (default: 2)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing of video segments")
    parser.add_argument("--fast", action="store_true", help="Use fast copy mode (faster but may have sync issues)")
    parser.add_argument("--info", action="store_true", help="Show video information and exit")
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    num_parts = max(1, args.parts)
    
    # Validate input
    if not video_path.exists():
        print(f"Error: File {video_path} does not exist")
        return 1
        
    if video_path.suffix.lower() not in [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"]:
        print(f"Warning: File {video_path} may not be a supported video format. Results may be unexpected.")
    
    # Show video info if requested
    if args.info:
        try:
            info = get_video_info(video_path)
            ffmpeg_status = "✅ Available" if check_ffmpeg() else "❌ Not found"
            print(f"\nVideo Information:")
            print(f"Duration: {info['duration']:.2f} seconds ({info['duration']/60:.1f} minutes)")
            print(f"Resolution: {info['width']}x{info['height']}")
            print(f"Frame Rate: {info['fps']:.2f} fps")
            print(f"Total Frames: {info['frame_count']}")
            print(f"Codec: {info['codec']}")
            print(f"FFmpeg: {ffmpeg_status}")
            return 0
        except Exception as e:
            print(f"Error getting video info: {str(e)}")
            return 1
    
    # Check FFmpeg availability
    if not check_ffmpeg():
        print("❌ Error: FFmpeg is required but not found.")
        print("Please install FFmpeg from: https://ffmpeg.org/download.html")
        return 1
    
    # For single part, just copy the file
    if num_parts == 1:
        output_path = video_path.parent / f"Part1_{video_path.stem}.mp4"
        print(f"Only one part requested. Copying to {output_path}")
        shutil.copy2(video_path, output_path)
        return 0
    
    # Analyze video to find the split points
    try:
        print(f"\n🎬 Starting video analysis and splitting...")
        print(f"📁 Input: {video_path}")
        print(f"✂️  Splitting into: {num_parts} parts")
        print(f"⚡ Parallel processing: {'Enabled' if not args.no_parallel else 'Disabled'}")
        print(f"🚀 Mode: {'Fast copy' if args.fast else 'High quality'}")
        print("-" * 60)
        
        start_time = time.time()
        
        split_times = analyze_video_content(video_path, num_parts)
        success = split_video(video_path, split_times, not args.no_parallel, args.fast)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("-" * 60)
        print(f"✅ Processing completed in {processing_time:.1f} seconds")
        
        return 0 if success else 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())