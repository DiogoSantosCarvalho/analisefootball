
import sys
from pathlib import Path

# Project root directory
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Path to the trained YOLO model
# NOTE: Model files are NOT stored in GitHub (too large for LFS without cost).
# Download or place your model files in the paths below:
model_path = PROJECT_DIR / "Models" / "Trained" / "yolov11" / "weights" / "best.pt"

# Alternative model paths (uncomment if using different models)
# model_path = PROJECT_DIR / "Models" / "Trained" / "yolov11_keypoints_" / "weights" / "best.pt"
# model_path = PROJECT_DIR / "Models" / "Pretrained" / "yolo11n.pt"  # Base YOLO model
# model_path = PROJECT_DIR / "path/to/your/custom/model.pt"   # Custom model

# =============================================================================
# VIDEO CONFIGURATION
# =============================================================================

# Input test video path (project-relative, portable across machines)
test_video = PROJECT_DIR / "inputvideo" / "40SEC.mp4"

# Output video path (project-relative)
test_video_output = PROJECT_DIR / "output" / "output.mp4"

# Alternative video paths (examples)
# test_video = PROJECT_DIR / "test_videos/sample.mp4"
# test_video_output = PROJECT_DIR / "output/tracked_sample.mp4"

# =============================================================================
# REAL-TIME / LIVE CAMERA SOURCE
# =============================================================================
# Pick ONE of the options below and assign it to `live_source`.
#
# Option A — Webcam or USB (DroidCam USB mode):
#   live_source = 0          # default webcam / DroidCam as virtual camera index 0
#   live_source = 1          # if your phone cam shows up as index 1
#
# Option B — IP Webcam app (Android):
#   Install "IP Webcam" on your phone, start the server.
#   live_source = "http://192.168.1.X:8080/video"   # replace X with your phone IP
#
# Option C — OBS Virtual Camera:
#   In OBS: Start Virtual Camera. The phone stream goes into OBS via USB/Wi-Fi.
#   live_source = 0  (or 1/2 depending on which index OBS Virtual Camera appears)
#
# Option D — RTSP stream (e.g. DroidCam Pro, some IP cameras):
#   live_source = "rtsp://192.168.1.X:4747/video"
#
live_source = 1   # 0=webcam portátil, 1=Camo Studio (telemóvel), 2=outra câmara

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Team assignment training parameters
TRAINING_FRAME_STRIDE = 12        # Skip frames during training data collection
TRAINING_FRAME_LIMIT = 120 * 24   # Maximum frames for training (120*24 = ~2 mins at 24fps)

# Clustering parameters  
EMBEDDING_BATCH_SIZE = 24         # Batch size for SigLIP embedding extraction
UMAP_COMPONENTS = 3               # UMAP dimensionality reduction components
N_TEAMS = 2                       # Number of teams to cluster (usually 2)

# Tracking parameters
TRACKER_MATCH_THRESH = 0.5        # ByteTrack matching threshold
TRACKER_BUFFER_SIZE = 120         # Number of frames to keep in tracking buffer

# Ball interpolation
BALL_INTERPOLATION_LIMIT = 30     # Max frames to interpolate missing ball detections

# =============================================================================
# DISTANCE TRACKING
# =============================================================================
# Approximate meters per pixel for a standard broadcast football camera.
# Based on: ~1920px width showing ~60m of pitch at midfield = 0.031 m/px.
# A factor of ~0.1 accounts for perspective (players appear smaller far away).
# For precise distances, use homography with keypoint detection.
# Tune this value for your specific video/camera setup.
METERS_PER_PIXEL = 0.065

# =============================================================================
# DETECTION CLASSES
# =============================================================================

CLASS_NAMES = {
    0: "Player",
    1: "Ball", 
    2: "Referee"
}

# Class colors for visualization (BGR format)
CLASS_COLORS = {
    0: (0, 255, 0),    # Green for players
    1: (0, 0, 255),    # Red for ball
    2: (255, 0, 0)     # Blue for referees
}

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# GPU settings
USE_GPU = True                    # Set to False to force CPU usage
GPU_DEVICE = 0                    # GPU device index (if multiple GPUs)

# Processing settings
MAX_VIDEO_FRAMES = -1             # Max frames to process (-1 for all frames)
OUTPUT_FPS = 30                   # Output video FPS
OUTPUT_MAX_WIDTH = 1920           # Set None to keep original resolution (e.g., 4K)

# Memory optimization
ENABLE_SAHI = False               # Enable SAHI for large image inference
SAHI_SLICE_HEIGHT = 640           # SAHI slice height
SAHI_SLICE_WIDTH = 640            # SAHI slice width
SAHI_OVERLAP_HEIGHT = 0.2         # SAHI overlap ratio
SAHI_OVERLAP_WIDTH = 0.2          # SAHI overlap ratio

# =============================================================================
# VALIDATION & DEBUGGING
# =============================================================================

# Print configuration status
if __name__ == "__main__":
    print("=== Soccer Analysis Configuration ===")
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"Model Path: {model_path}")
    print(f"Test Video: {test_video}")
    print(f"Output Path: {test_video_output}")
    print()
    
    issues = validate_config()
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease fix these issues before running the system.")
    else:
        print("✅ Configuration looks good!")
        print("\nRun 'python main.py' to start the complete pipeline.")