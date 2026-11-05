"""
Complete Pipeline: YouTube Videos → YOLO Dataset
Runs extraction, deduplication, and organization automatically
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    
    try:
        subprocess.run(cmd, check=True, shell=isinstance(cmd, str))
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    try:
        import cv2
        import imagehash
        import yaml
        print("✓ All Python packages installed")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("\nInstalling required packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # Check yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        print("✓ yt-dlp installed")
    except:
        print("Installing yt-dlp...")
        subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], check=True)

def main():
    """Run complete pipeline"""
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║     E-WASTE DETECTION DATASET CREATION PIPELINE           ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Step 0: Check dependencies
    check_dependencies()
    
    # Step 1: Extract frames from YouTube
    print("\n" + "="*60)
    user_input = input("Step 1: Extract frames from YouTube videos? (y/n): ")
    if user_input.lower() == 'y':
        if not run_command([sys.executable, "youtube_frame_extractor.py"], 
                          "Extracting frames from YouTube"):
            return
    else:
        print("Skipping frame extraction")
    
    # Step 2: Deduplicate images
    print("\n" + "="*60)
    user_input = input("Step 2: Remove duplicate/similar images? (y/n): ")
    if user_input.lower() == 'y':
        # First dry run to show what will be removed
        print("\nRunning dry-run to preview duplicates...")
        run_command([sys.executable, "image_deduplicator.py", "raw_extracted", "--dry-run"],
                   "Preview duplicate detection")
        
        proceed = input("\nProceed with deduplication? (y/n): ")
        if proceed.lower() == 'y':
            if not run_command([sys.executable, "image_deduplicator.py", "raw_extracted", 
                              "--output-dir", "unique_images"],
                              "Removing duplicates"):
                return
    else:
        print("Skipping deduplication")
    
    # Step 3: Organize into YOLO structure
    print("\n" + "="*60)
    print("Step 3: Organize into YOLO dataset structure")
    print("\nDefine your object classes (3-4 classes recommended)")
    print("Examples for e-waste project: e-waste, circuit-board, battery, cable")
    print("Or: e-waste, landscape, vehicle, building")
    
    classes_input = input("\nEnter class names separated by spaces: ")
    classes = classes_input.strip().split()
    
    if len(classes) < 2:
        print("Please provide at least 2 classes")
        return
    
    print(f"\nClasses: {', '.join(classes)}")
    
    # Determine input directory
    input_dir = "unique_images" if Path("unique_images").exists() else "raw_extracted"
    
    if not run_command([sys.executable, "dataset_organizer.py", input_dir, 
                       "--classes"] + classes + ["--output-dir", "yolo_dataset"],
                      "Organizing YOLO dataset"):
        return
    
    # Final summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE! 🎉")
    print(f"{'='*60}")
    print("\nYour dataset is ready for annotation!")
    print(f"\nDataset location: {Path('yolo_dataset').absolute()}")
    print(f"Classes: {', '.join(classes)}")
    
    print("\n📋 NEXT STEPS:")
    print("\n1. ANNOTATE YOUR IMAGES:")
    print("   Option A: LabelImg (Desktop tool)")
    print("     - Install: pip install labelImg")
    print("     - Run: labelImg yolo_dataset/images/train yolo_dataset/classes.txt")
    print("     - Set format to YOLO in the left sidebar")
    
    print("\n   Option B: Roboflow (Web-based)")
    print("     - Upload images to roboflow.com")
    print("     - Annotate online")
    print("     - Export in YOLO format")
    
    print("\n   Option C: CVAT (Web-based, advanced)")
    print("     - Self-hosted or cloud: cvat.ai")
    
    print("\n2. ANNOTATION FORMAT:")
    print("   Each label file should contain lines like:")
    print("   <class_id> <x_center> <y_center> <width> <height>")
    print("   Example: 0 0.5 0.5 0.3 0.4")
    print("   (All values normalized 0-1)")
    
    print("\n3. TRAINING:")
    print("   Once annotated, train with:")
    print("   - YOLOv8: ultralytics (recommended)")
    print("   - YOLOv5: PyTorch")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
