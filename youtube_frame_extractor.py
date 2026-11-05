"""
YouTube Frame Extractor for Object Detection Dataset
Extracts frames from YouTube videos at specified intervals
"""

import os
import subprocess
import cv2
from pathlib import Path

class YouTubeFrameExtractor:
    def __init__(self, output_dir="extracted_frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def download_video(self, url, video_name):
        """Download YouTube video using yt-dlp"""
        video_path = self.output_dir / f"{video_name}.mp4"
        
        try:
            # Download video with yt-dlp
            cmd = [
                'yt-dlp',
                '-f', 'best[height<=720]',  # 720p max to save space
                '-o', str(video_path),
                url
            ]
            print(f"Downloading: {url}")
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Downloaded: {video_name}")
            return video_path
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download {url}: {e}")
            return None
    
    def extract_frames(self, video_path, video_name, frame_interval=30, min_resolution=(640, 480)):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            video_name: Name for organizing frames
            frame_interval: Extract 1 frame every N frames (30 = ~1 per second at 30fps)
            min_resolution: Minimum frame resolution (width, height)
        """
        frames_dir = self.output_dir / f"frames_{video_name}"
        frames_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"✗ Failed to open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {fps:.1f} fps, {total_frames} frames")
        print(f"Extracting 1 frame every {frame_interval} frames...")
        
        frame_count = 0
        saved_count = 0
        saved_paths = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at intervals
            if frame_count % frame_interval == 0:
                # Check resolution
                h, w = frame.shape[:2]
                if w >= min_resolution[0] and h >= min_resolution[1]:
                    frame_filename = frames_dir / f"{video_name}_frame_{saved_count:04d}.jpg"
                    cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_paths.append(frame_filename)
                    saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"✓ Extracted {saved_count} frames from {video_name}")
        
        # Clean up video file to save space
        try:
            os.remove(video_path)
            print(f"✓ Cleaned up video file")
        except:
            pass
        
        return saved_paths
    
    def process_url(self, url, video_name, frame_interval=30):
        """Download video and extract frames"""
        print(f"\n{'='*60}")
        print(f"Processing: {video_name}")
        print(f"{'='*60}")
        
        video_path = self.download_video(url, video_name)
        if video_path and os.path.exists(video_path):
            frames = self.extract_frames(video_path, video_name, frame_interval)
            return frames
        return []


def main():
    # Video URLs
    videos = [
        ("https://www.youtube.com/watch?v=BdPGO6sfc3c", "ewaste_1"),
        ("https://www.youtube.com/watch?v=aDjDGrrDD7o", "ewaste_2"),
        ("https://www.youtube.com/shorts/NaiKCNLxFUY", "ewaste_3"),
        ("https://www.youtube.com/watch?v=eT34ypRodB0", "ewaste_4"),
        ("https://www.youtube.com/shorts/IUrdgZblgKs", "ewaste_5"),
    ]
    
    extractor = YouTubeFrameExtractor(output_dir="raw_extracted")
    
    all_frames = []
    
    # Process each video
    for url, name in videos:
        frames = extractor.process_url(url, name, frame_interval=30)  # 1 frame per second
        all_frames.extend(frames)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total frames extracted: {len(all_frames)}")
    print(f"Next step: Run deduplication script to remove similar frames")
    

if __name__ == "__main__":
    main()
