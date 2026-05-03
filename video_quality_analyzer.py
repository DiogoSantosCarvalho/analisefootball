"""
Video Quality Analyzer with YOLO Detection
============================================

Analyzes multiple MP4 videos for:
- Metadata extraction (resolution, FPS, duration, etc.)
- Downscaled variants generation (1080p, 720p, 480p)
- Per-frame quality metrics (sharpness, SNR, blur detection)
- YOLO object detection (frame-by-frame)
- Aggregated statistics and reporting

Dependencies: opencv-python, numpy, pandas, ultralytics, ffmpeg
"""

import os
import json
import csv
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO
import time

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# METADATA EXTRACTION
# ============================================================================

def get_video_metadata(video_path: str) -> Dict:
    """
    Extract video metadata using OpenCV.
    
    Args:
        video_path: Path to MP4 file
        
    Returns:
        Dict with keys: name, resolution, fps, duration_s, total_frames, filesize_mb
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        filesize_mb = os.path.getsize(video_path) / (1024 * 1024)
        bitrate_mbps = (filesize_mb * 8) / duration_s if duration_s > 0 else 0
        
        return {
            'name': Path(video_path).stem,
            'resolution': f"{width}x{height}",
            'fps': fps,
            'duration_s': duration_s,
            'total_frames': total_frames,
            'filesize_mb': round(filesize_mb, 2),
            'bitrate_mbps': round(bitrate_mbps, 2),
            'width': width,
            'height': height
        }
    except Exception as e:
        logger.error(f"Error extracting metadata from {video_path}: {e}")
        return None


# ============================================================================
# VIDEO DOWNSCALING WITH FFMPEG
# ============================================================================

def generate_variant(input_path: str, output_path: str, target_height: int = 1080) -> bool:
    """
    Generate downscaled video variant using ffmpeg.
    
    Args:
        input_path: Path to original video
        output_path: Path to save variant
        target_height: Target height (maintains aspect ratio)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Calculate width to maintain aspect ratio
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vf', f'scale=-1:{target_height}',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-y',  # Overwrite without asking
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"Generated variant: {output_path}")
            return True
        else:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timeout for {output_path}")
        return False
    except Exception as e:
        logger.error(f"Error generating variant {output_path}: {e}")
        return False


def ensure_variants(video_path: str, variants_dir: str = "variants") -> Dict[str, str]:
    """
    Ensure downscaled variants exist. Generate if not present.
    
    Args:
        video_path: Path to original video
        variants_dir: Base directory for variants
        
    Returns:
        Dict mapping variant names to file paths {
            'original': path_to_original,
            '1080p': path_to_1080p,
            '720p': path_to_720p,
            '480p': path_to_480p
        }
    """
    video_name = Path(video_path).stem
    variants = {}
    
    # Original
    variants['original'] = video_path
    
    # 1080p, 720p, 480p
    for height, label in [(1080, '1080p'), (720, '720p'), (480, '480p')]:
        variant_path = os.path.join(variants_dir, video_name, f"{label}.mp4")
        
        if not os.path.exists(variant_path):
            logger.info(f"Generating {label} variant for {video_name}...")
            if not generate_variant(video_path, variant_path, target_height=height):
                logger.warning(f"Failed to generate {label}")
                continue
        
        variants[label] = variant_path
    
    return variants


# ============================================================================
# FRAME QUALITY METRICS
# ============================================================================

def calculate_sharpness(frame: np.ndarray) -> float:
    """
    Calculate sharpness using Laplacian variance (Lipschitz criterion).
    
    Args:
        frame: Input frame (BGR or grayscale)
        
    Returns:
        Laplacian variance (higher = sharper)
    """
    if frame is None or frame.size == 0:
        return 0.0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return float(variance)


def calculate_snr(frame: np.ndarray) -> float:
    """
    Estimate Signal-to-Noise Ratio (simple approach).
    Uses std of Gaussian blurred - std of original as proxy.
    
    Args:
        frame: Input frame
        
    Returns:
        Estimated SNR value
    """
    if frame is None or frame.size == 0:
        return 0.0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    gray = gray.astype(np.float32)
    
    signal_var = np.var(gray)
    if signal_var == 0:
        return 0.0
    
    # Simple noise estimate: variance of high-pass filtered image
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8.0
    noise_img = cv2.filter2D(gray, -1, kernel)
    noise_var = np.var(noise_img)
    
    snr = 10 * np.log10(signal_var / (noise_var + 1e-5)) if signal_var > 0 else 0
    return float(snr)


def detect_blur(frame: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Detect if frame is blurry using Laplacian variance threshold.
    
    Args:
        frame: Input frame
        threshold: Laplacian variance threshold (below = blurry)
        
    Returns:
        True if frame is blurry, False otherwise
    """
    sharpness = calculate_sharpness(frame)
    return sharpness < threshold


def analyze_frame_quality(frame: np.ndarray, blur_threshold: float = 100.0) -> Dict:
    """
    Analyze frame quality metrics.
    
    Args:
        frame: Input frame
        blur_threshold: Threshold for blur detection
        
    Returns:
        Dict with sharpness, snr, is_blurry
    """
    return {
        'sharpness': calculate_sharpness(frame),
        'snr': calculate_snr(frame),
        'is_blurry': detect_blur(frame, blur_threshold)
    }


# ============================================================================
# YOLO DETECTION
# ============================================================================

def run_yolo_detection(frame: np.ndarray, model: YOLO, conf_threshold: float = 0.5) -> Tuple[int, float, float]:
    """
    Run YOLO detection on a frame and return statistics.
    
    Args:
        frame: Input frame
        model: Loaded YOLO model
        conf_threshold: Confidence threshold for detections
        
    Returns:
        Tuple of (num_detections, mean_confidence, processing_time_ms)
    """
    start_time = time.time()
    
    try:
        results = model(frame, conf=conf_threshold, verbose=False)
        processing_time_ms = (time.time() - start_time) * 1000
        
        if len(results) == 0 or results[0].boxes is None:
            return 0, 0.0, processing_time_ms
        
        boxes = results[0].boxes
        num_detections = len(boxes)
        
        if num_detections == 0:
            return 0, 0.0, processing_time_ms
        
        confidences = boxes.conf.cpu().numpy()
        mean_confidence = float(np.mean(confidences))
        
        return num_detections, mean_confidence, processing_time_ms
        
    except Exception as e:
        logger.error(f"YOLO detection error: {e}")
        return 0, 0.0, (time.time() - start_time) * 1000


# ============================================================================
# VIDEO ANALYSIS
# ============================================================================

def analyze_video_variant(video_path: str, variant_name: str, metadata: Dict, 
                         model: YOLO, frame_skip: int = 10, 
                         blur_threshold: float = 100.0) -> Dict:
    """
    Analyze single video variant: quality metrics + YOLO detection.
    
    Args:
        video_path: Path to video file
        variant_name: Name of variant (e.g., '1080p', 'original')
        metadata: Original video metadata
        model: Loaded YOLO model
        frame_skip: Analyze every Nth frame
        blur_threshold: Laplacian threshold for blur detection
        
    Returns:
        Dict with frame-level and aggregated statistics
    """
    logger.info(f"Analyzing {variant_name}: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None
    
    frame_data = []
    frame_idx = 0
    frames_analyzed = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Quality analysis
            quality = analyze_frame_quality(frame, blur_threshold)
            
            # YOLO detection
            num_dets, mean_conf, proc_time = run_yolo_detection(frame, model)
            
            frame_data.append({
                'frame_idx': frame_idx,
                'sharpness': quality['sharpness'],
                'snr': quality['snr'],
                'is_blurry': quality['is_blurry'],
                'num_detections': num_dets,
                'mean_confidence': mean_conf,
                'detection_time_ms': proc_time
            })
            
            frames_analyzed += 1
            if frames_analyzed % 50 == 0:
                logger.info(f"  Analyzed {frames_analyzed} frames from {variant_name}")
            
            frame_idx += 1
    
    finally:
        cap.release()
    
    if not frame_data:
        logger.warning(f"No frames analyzed for {variant_name}")
        return None
    
    # Aggregate statistics
    df = pd.DataFrame(frame_data)
    
    aggregated = {
        'variant': variant_name,
        'frames_analyzed': len(df),
        'sharpness': {
            'mean': float(df['sharpness'].mean()),
            'median': float(df['sharpness'].median()),
            'std': float(df['sharpness'].std())
        },
        'snr': {
            'mean': float(df['snr'].mean()),
        },
        'blur': {
            'percent': float(100 * df['is_blurry'].sum() / len(df))
        },
        'detection': {
            'avg_detections_per_frame': float(df['num_detections'].mean()),
            'avg_confidence': float(df['mean_confidence'].mean()),
            'frames_no_detection': int((df['num_detections'] == 0).sum()),
            'avg_processing_time_ms': float(df['detection_time_ms'].mean())
        }
    }
    
    return {
        'frame_level': frame_data,
        'aggregated': aggregated
    }


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(video_name: str, analysis: Dict, output_dir: str = "analysis_output") -> str:
    """
    Save analysis results to JSON file.
    
    Args:
        video_name: Name of video
        analysis: Complete analysis dict
        output_dir: Output directory
        
    Returns:
        Path to saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_name}_analysis.json")
    
    try:
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved results: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None


def create_summary_csv(all_results: List[Dict], output_path: str = "analysis_summary.csv"):
    """
    Create summary CSV with results from all videos.
    
    Args:
        all_results: List of analysis dicts
        output_path: Path to save CSV
    """
    rows = []
    
    for result in all_results:
        video_name = result['video_name']
        metadata = result['metadata']
        
        for variant_analysis in result['variants']:
            agg = variant_analysis['aggregated']
            
            row = {
                'video': video_name,
                'variant': agg['variant'],
                'resolution': metadata['resolution'],
                'fps': metadata['fps'],
                'duration_s': metadata['duration_s'],
                'filesize_mb': metadata['filesize_mb'],
                'bitrate_mbps': metadata['bitrate_mbps'],
                'avg_sharpness': agg['sharpness']['mean'],
                'snr_avg': agg['snr']['mean'],
                'blur_pct': agg['blur']['percent'],
                'avg_detections': agg['detection']['avg_detections_per_frame'],
                'avg_confidence': agg['detection']['avg_confidence'],
                'frames_no_detection': agg['detection']['frames_no_detection'],
                'processing_time_avg_ms': agg['detection']['avg_processing_time_ms']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Summary CSV saved: {output_path}")
    return output_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def analyze_videos(
    videos_dir: str = "videos",
    variants_dir: str = "variants",
    output_dir: str = "analysis_output",
    model_path: str = "yolov11n.pt",
    frame_skip: int = 10,
    blur_threshold: float = 100.0
):
    """
    Main pipeline: analyze all videos in directory.
    
    Args:
        videos_dir: Directory containing MP4 files
        variants_dir: Directory for downscaled variants
        output_dir: Directory for results
        model_path: Path to YOLO model
        frame_skip: Analyze every Nth frame
        blur_threshold: Laplacian threshold for blur detection
    """
    logger.info("=== Starting Video Quality Analysis ===")
    
    # Load YOLO model
    logger.info(f"Loading YOLO model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return
    
    # Find all MP4 files
    video_files = list(Path(videos_dir).glob("*.mp4"))
    if not video_files:
        logger.warning(f"No MP4 files found in {videos_dir}")
        return
    
    logger.info(f"Found {len(video_files)} videos")
    
    all_results = []
    
    # Process each video
    for video_path in video_files:
        video_path_str = str(video_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {video_path.name}")
        logger.info(f"{'='*60}")
        
        # Extract metadata
        metadata = get_video_metadata(video_path_str)
        if metadata is None:
            logger.error(f"Skipping {video_path.name}")
            continue
        
        logger.info(f"  Resolution: {metadata['resolution']}")
        logger.info(f"  FPS: {metadata['fps']}")
        logger.info(f"  Duration: {metadata['duration_s']:.2f}s")
        logger.info(f"  Filesize: {metadata['filesize_mb']}MB")
        
        # Generate variants
        variants = ensure_variants(video_path_str, variants_dir)
        logger.info(f"Variants available: {list(variants.keys())}")
        
        # Analyze each variant
        variants_analysis = []
        for variant_name, variant_path in variants.items():
            if not os.path.exists(variant_path):
                logger.warning(f"Variant not available: {variant_name}")
                continue
            
            analysis = analyze_video_variant(
                variant_path, variant_name, metadata, model, 
                frame_skip, blur_threshold
            )
            if analysis:
                variants_analysis.append(analysis)
        
        # Save results
        result = {
            'video_name': metadata['name'],
            'metadata': metadata,
            'variants': variants_analysis
        }
        
        save_results(metadata['name'], result, output_dir)
        all_results.append(result)
    
    # Create summary CSV
    if all_results:
        csv_path = os.path.join(output_dir, "analysis_summary.csv")
        create_summary_csv(all_results, csv_path)
        logger.info(f"\nSummary: {csv_path}")
    
    logger.info("\n=== Analysis Complete ===")
    return all_results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create test directory structure
    os.makedirs("videos", exist_ok=True)
    os.makedirs("variants", exist_ok=True)
    os.makedirs("analysis_output", exist_ok=True)
    
    logger.info("Video Quality Analyzer - Example Usage")
    logger.info("========================================")
    logger.info("")
    logger.info("To use this analyzer:")
    logger.info("1. Place your MP4 files in the 'videos/' directory")
    logger.info("2. Run: python video_quality_analyzer.py")
    logger.info("")
    logger.info("Output files:")
    logger.info("- analysis_output/<video_name>_analysis.json - Per-frame and aggregated stats")
    logger.info("- analysis_output/analysis_summary.csv - Summary table across all videos")
    logger.info("")
    logger.info("Configuration:")
    logger.info("- model_path: YOLO model to use (default: yolov11n.pt)")
    logger.info("- frame_skip: Analyze every Nth frame (default: 10)")
    logger.info("- blur_threshold: Laplacian variance threshold (default: 100.0)")
    logger.info("")
    
    # Uncomment to run analysis
    # results = analyze_videos(
    #     videos_dir="videos",
    #     variants_dir="variants",
    #     output_dir="analysis_output",
    #     model_path="yolov11n.pt",
    #     frame_skip=10,
    #     blur_threshold=100.0
    # )
