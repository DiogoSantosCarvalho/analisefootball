import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

from pipelines import TrackingPipeline, ProcessingPipeline, DetectionPipeline, KeypointPipeline, TacticalPipeline
from constants import model_path, test_video, METERS_PER_PIXEL, live_source, OUTPUT_MAX_WIDTH
from keypoint_detection.keypoint_constants import keypoint_model_path
from utils import BallKalmanTracker, get_video_fps
from collections import defaultdict
import numpy as np
import time
from tqdm import tqdm
import supervision as sv
import cv2
import math
import json


class CompleteSoccerAnalysisPipeline:
    """Complete end-to-end soccer analysis pipeline integrating all functionalities."""
    
    def __init__(self, detection_model_path: str, keypoint_model_path: str):
        """Initialize all pipeline components.
        
        Args:
            detection_model_path: Path to YOLO detection model
            keypoint_model_path: Path to YOLO keypoint detection model
        """
        self.detection_pipeline = DetectionPipeline(detection_model_path)
        self.keypoint_pipeline = KeypointPipeline(keypoint_model_path)
        self.tracking_pipeline = TrackingPipeline(detection_model_path, use_color_kmeans=True)
        self.tactical_pipeline = TacticalPipeline(keypoint_model_path, detection_model_path) if TacticalPipeline is not None else None
        if self.tactical_pipeline is None:
            print("Warning: Tactical minimap disabled (missing optional 'sports' dependency).")
        self.processing_pipeline = ProcessingPipeline()
        
    def initialize_models(self):
        """Initialize all models required for complete analysis."""
        
        print("Initializing all pipeline models...")
        start_time = time.time()
        
        # Initialize all pipeline models
        self.detection_pipeline.initialize_model()
        self.keypoint_pipeline.initialize_model()
        self.tracking_pipeline.initialize_models()
        # Note: tactical_pipeline models are NOT loaded — generate_minimap_overlay
        # uses only the stateless homography_transformer and pitch_config; keypoints
        # are detected externally via self.keypoint_pipeline to avoid duplicate GPU loads.
        
        init_time = time.time() - start_time
        print(f"All models initialized in {init_time:.2f}s")
        
    def _build_player_gallery(self, video_path: str, all_tracks: dict, locked_teams: dict,
                               frame_count: int, output_dir: str) -> str:
        """Produce a gallery image showing one crop per player tracker_id.

        For every tracked player we pick the frame where the bbox is largest
        (most likely a clean, close-up view) and crop the top 2/3 (torso+head).
        The gallery is saved as <output_dir>/player_gallery.jpg and the path is returned.
        """
        player_tracks = all_tracks.get('player', {})
        if not player_tracks:
            return None

        # For each tracker_id find the frame with the largest bbox area
        best: dict = {}   # {tid: (frame_idx, area)}
        for fi, tdict in player_tracks.items():
            for tid, bbox in tdict.items():
                if bbox[0] is None or np.isnan(float(bbox[0])):
                    continue
                area = (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1]))
                if tid not in best or area > best[tid][1]:
                    best[tid] = (fi, area)

        if not best:
            return None

        # Gather the frame indices we need (sorted, unique)
        needed_frames = sorted(set(fi for fi, _ in best.values()))
        frame_to_image: dict = {}

        gen = sv.get_video_frames_generator(video_path, end=frame_count if frame_count != -1 else None)
        for fi, frame in enumerate(gen):
            if fi in needed_frames:
                frame_to_image[fi] = frame.copy()
            if fi > max(needed_frames):
                break

        THUMB_W, THUMB_H = 80, 120
        GAL_COLS = 10
        tids_sorted = sorted(best.keys())

        crops = []
        for tid in tids_sorted:
            fi, _ = best[tid]
            frame = frame_to_image.get(fi)
            if frame is None:
                crop = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
            else:
                bbox = player_tracks[fi][tid]
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                # Clamp to frame boundaries
                fh, fw = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fw, x2), min(fh, y2)
                if x2 <= x1 or y2 <= y1:
                    crop = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
                else:
                    crop = frame[y1:y2, x1:x2]
                    # Use top 70% of the crop (torso/jersey area)
                    crop = crop[:int(crop.shape[0] * 0.7), :]
                    crop = cv2.resize(crop, (THUMB_W, THUMB_H))

            # Colour border by team
            team = locked_teams.get(tid, None)
            border_color = (34, 34, 255) if team == 0 else (255, 102, 34) if team == 1 else (200, 200, 200)
            crop = cv2.copyMakeBorder(crop, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=border_color)

            # Label: tracker_id + team
            team_str = f"T{team}" if team is not None else "?"
            label = f"#{tid} {team_str}"
            cv2.putText(crop, label, (2, crop.shape[0] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            crops.append(crop)

        # Build grid
        thumb_total_w = THUMB_W + 8  # + 2*border
        thumb_total_h = THUMB_H + 8
        n_cols = min(GAL_COLS, len(crops))
        n_rows = math.ceil(len(crops) / n_cols)
        gallery = np.zeros((n_rows * thumb_total_h, n_cols * thumb_total_w, 3), dtype=np.uint8)
        for idx, crop in enumerate(crops):
            row, col = divmod(idx, n_cols)
            y0 = row * thumb_total_h
            x0 = col * thumb_total_w
            gallery[y0:y0 + thumb_total_h, x0:x0 + thumb_total_w] = crop

        gallery_path = str(Path(output_dir) / "player_gallery.jpg")
        cv2.imwrite(gallery_path, gallery)
        print(f"  Player gallery saved → {gallery_path}")
        return gallery_path

    def _build_requested_metrics(self,
                                 frames_processed: int,
                                 total_time_s: float,
                                 source_fps: float,
                                 detection_count_sums: dict,
                                 detection_conf_sums: dict,
                                 detection_conf_counts: dict) -> dict:
        """Build requested quality/performance metrics for JSON output.

        Note:
            Precision/Recall/mAP and MOTA/IDF1 require frame-level ground truth
            annotations, which are not produced by this pipeline.
        """
        processing_fps = (float(frames_processed) / float(total_time_s)) if total_time_s > 0 and frames_processed > 0 else None

        def _mean_conf(obj_name: str):
            count = int(detection_conf_counts.get(obj_name, 0))
            if count <= 0:
                return None
            return float(detection_conf_sums.get(obj_name, 0.0) / count)

        return {
            'fps': {
                'source_video_fps': float(source_fps) if source_fps is not None else None,
                'processing_fps': float(processing_fps) if processing_fps is not None else None,
                'real_time_capable': bool(processing_fps >= source_fps) if processing_fps is not None and source_fps is not None else None,
            },
            'detection_metrics': {
                'precision': None,
                'recall': None,
                'mAP': None,
                'status': 'ground_truth_required',
                'reason': 'Precision/Recall/mAP require labeled ground-truth boxes per frame.',
                'proxy_stats': {
                    'detections_total': {
                        'players': int(detection_count_sums.get('player', 0)),
                        'ball': int(detection_count_sums.get('ball', 0)),
                        'referees': int(detection_count_sums.get('referee', 0)),
                    },
                    'mean_confidence': {
                        'players': _mean_conf('player'),
                        'ball': _mean_conf('ball'),
                        'referees': _mean_conf('referee'),
                    }
                }
            },
            'tracking_metrics': {
                'MOTA': None,
                'IDF1': None,
                'status': 'ground_truth_required',
                'reason': 'MOTA/IDF1 require labeled ground-truth trajectories and identities per frame.'
            }
        }

    def analyze_video(self, video_path: str, frame_count: int = -1, output_suffix: str = "_complete_analysis",
                      target_player_id: int = None):
        """Run complete end-to-end soccer analysis using streaming to minimize RAM usage.
        
        Flow:
        1. Initialize models
        2. Train team assignment models
        3. Stream video to extract detections & tracks (no frames kept in memory)
        4. Interpolate ball tracks
        5. Stream video again: annotate each frame and write directly to output
        
        Args:
            video_path: Path to input video
            frame_count: Number of frames to process (-1 for all)
            output_suffix: Suffix for output video file
            target_player_id: If set, this tracker_id is highlighted in the video with a gold
                              bounding box, a stats panel (bottom-left) and a movement trail
                              on the minimap.  Find the right id by checking player_gallery.jpg.
            
        Returns:
            Path to output video
        """
        print("=== Starting Complete Soccer Analysis Pipeline ===")
        total_start_time = time.time()
        
        # Step 1: Initialize all models
        self.initialize_models()
        
        # Step 2: Train team assignment models
        print("\n[Step 2/7] Training team assignment models...")
        self.tracking_pipeline.train_team_assignment_models(video_path)

        # Step 3: Stream video for detection & tracking (no frames kept in RAM)
        print("\n[Step 3/7] Streaming video for detections and tracking...")
        video_info = sv.VideoInfo.from_video_path(video_path)
        total_frames = video_info.total_frames if frame_count == -1 else frame_count
        
        all_tracks = {'player': {}, 'ball': {}, 'referee': {}, 'player_classids': {}}
        frame_generator = sv.get_video_frames_generator(video_path, end=frame_count if frame_count != -1 else None)

        # Aggregated detection statistics for output metrics.
        detection_count_sums = {'player': 0, 'ball': 0, 'referee': 0}
        detection_conf_sums = {'player': 0.0, 'ball': 0.0, 'referee': 0.0}
        detection_conf_counts = {'player': 0, 'ball': 0, 'referee': 0}

        # Kalman filter for ball — replaces post-processing interpolation
        ball_kalman = BallKalmanTracker()
        # Dedicated ByteTracker for referees so their IDs are stable across frames
        ref_tracker = sv.ByteTrack(lost_track_buffer=120)

        i = -1
        for i, frame in enumerate(tqdm(frame_generator, total=total_frames, desc="Detecting & tracking")):
            player_detections, ball_detections, referee_detections = self.detection_pipeline.detect_frame_objects(frame)

            # Aggregate detection count and confidence stats.
            detection_count_sums['player'] += int(len(player_detections.xyxy))
            detection_count_sums['ball'] += int(len(ball_detections.xyxy))
            detection_count_sums['referee'] += int(len(referee_detections.xyxy))

            for obj_name, detections in (
                ('player', player_detections),
                ('ball', ball_detections),
                ('referee', referee_detections),
            ):
                confidences = getattr(detections, 'confidence', None)
                if confidences is not None and len(confidences) > 0:
                    detection_conf_sums[obj_name] += float(np.sum(confidences))
                    detection_conf_counts[obj_name] += int(len(confidences))

            player_detections = self.tracking_pipeline.tracking_callback(player_detections)
            player_detections, _ = self.tracking_pipeline.clustering_callback(frame, player_detections)
            # Track referees with ByteTrack so IDs are consistent (not sequential per frame)
            if len(referee_detections.xyxy) > 0:
                referee_detections = ref_tracker.update_with_detections(referee_detections)

            all_tracks = self.tracking_pipeline.convert_detection_to_tracks(
                player_detections, ball_detections, referee_detections, all_tracks, i
            )

            # Kalman-filter the ball: smooth detections + predict through misses
            # Done AFTER convert_detection_to_tracks so Kalman overwrites raw detection
            raw_ball = ball_detections.xyxy[0].tolist() if len(ball_detections.xyxy) > 0 else None
            kalman_ball = ball_kalman.update(raw_ball)
            all_tracks['ball'][i] = kalman_ball if kalman_ball is not None else [None] * 4

        frames_processed = i + 1

        # Step 3b: Lock team assignments via majority vote over all frames.
        # Per-frame color prediction can flip (noisy crops) — voting fixes this:
        # each tracker_id is permanently assigned to whichever team it was predicted
        # as most often throughout the entire clip.
        print("\n[Step 3b/7] Locking team assignments (majority vote)...")
        team_votes = defaultdict(lambda: defaultdict(int))  # {tracker_id: {team: count}}
        for fi, classids in all_tracks['player_classids'].items():
            for tid, cid in classids.items():
                if tid == -1 or cid is None:
                    continue
                team_votes[tid][int(cid)] += 1
        locked_teams = {}
        for tid, votes in team_votes.items():
            locked_teams[tid] = max(votes, key=votes.get)  # team with most votes wins
        # Override all stored class_ids with the locked value
        for fi in all_tracks['player_classids']:
            for tid in list(all_tracks['player_classids'][fi].keys()):
                if tid in locked_teams:
                    all_tracks['player_classids'][fi][tid] = locked_teams[tid]
        n_t0 = sum(1 for t in locked_teams.values() if t == 0)
        n_t1 = sum(1 for t in locked_teams.values() if t == 1)
        print(f"  Locked: {n_t0} players → Team 0 (red), {n_t1} players → Team 1 (blue)")

        # Step 3c: Build player gallery so user can identify tracker_ids visually
        print("\n[Step 3c/7] Building player identification gallery...")
        output_dir = str(Path(video_path).parent)
        gallery_path = self._build_player_gallery(video_path, all_tracks, locked_teams, frame_count, output_dir)
        if gallery_path:
            print(f"  Open '{gallery_path}' to identify who is which tracker_id.")
            print(f"  Then re-run with target_player_id=<id> to spotlight that player.")

        # Step 4: Ball track interpolation (light pass to fill any remaining gaps)
        print("\n[Step 4/7] Smoothing ball tracks...")
        all_tracks = self.processing_pipeline.interpolate_ball_tracks(all_tracks)

        # Step 4b: Smooth player & referee bbox coordinates over time (EMA per tracker_id).
        # YOLO bboxes jitter 2-8 pixels frame-to-frame even for still players.
        # Smoothing here fixes stutter in both the video overlay AND the minimap.
        print("[Step 4b/7] Smoothing player/referee tracks...")
        BBOX_ALPHA = 0.35   # EMA weight for current frame; lower = smoother but more lag
        for track_key in ('player', 'referee'):
            raw = all_tracks[track_key]   # {frame_idx: {tracker_id: [x1,y1,x2,y2]}}
            if not raw:
                continue
            # Collect frame indices sorted, build per-tracker smoothed coords
            sorted_frames = sorted(raw.keys())
            smoothed = {}   # {tracker_id: [x1,y1,x2,y2]}  last EMA state
            smooth_tracks = {}
            for fi in sorted_frames:
                frame_data = raw[fi]
                new_frame_data = {}
                for tid, bbox in frame_data.items():
                    if bbox[0] is None or (hasattr(bbox[0], '__float__') and np.isnan(float(bbox[0]))):
                        new_frame_data[tid] = bbox
                        continue
                    arr = np.array(bbox, dtype=np.float32)
                    if tid in smoothed:
                        arr = BBOX_ALPHA * arr + (1.0 - BBOX_ALPHA) * smoothed[tid]
                    smoothed[tid] = arr
                    new_frame_data[tid] = arr.tolist()
                smooth_tracks[fi] = new_frame_data
            all_tracks[track_key] = smooth_tracks

        # Steps 5+6: Stream video again, annotate and write each frame directly (no frames list in RAM)
        print("\n[Step 5/7] Annotating and writing output video (streaming)...")
        output_path = self.processing_pipeline.generate_output_path(video_path, output_suffix)
        
        output_fps = get_video_fps(video_path)
        fps = output_fps
        frame_generator2 = sv.get_video_frames_generator(video_path, end=frame_count if frame_count != -1 else None)

        # Distance tracking: accumulated meters per tracker_id
        player_distances = {}       # {tracker_id: total_meters}
        player_prev_pos = {}        # {tracker_id: (ground_x, ground_y) in pixels}
        # Speed tracking: instantaneous km/h per tracker_id (last valid frame)
        player_speeds = {}          # {tracker_id: km/h}

        # Target player trail: last N pitch-coord positions for the minimap trail
        TARGET_TRAIL_LEN = 60
        target_trail: list = []     # list of (pitch_x, pitch_y)

        video_info_out = sv.VideoInfo.from_video_path(video_path)
        src_w, src_h = int(video_info_out.width), int(video_info_out.height)

        # Optional output downscale for smoother playback (e.g., 4K -> 1080p)
        if OUTPUT_MAX_WIDTH is not None and src_w > int(OUTPUT_MAX_WIDTH):
            scale = float(OUTPUT_MAX_WIDTH) / float(src_w)
            frame_w = int(round(src_w * scale))
            frame_h = int(round(src_h * scale))
            print(f"Downscaling output: {src_w}x{src_h} -> {frame_w}x{frame_h}")
        else:
            frame_w, frame_h = src_w, src_h

        writer = None
        for codec in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            candidate = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_w, frame_h))
            if candidate.isOpened():
                writer = candidate
                print(f"Writing with codec: {codec} | FPS: {output_fps} | Size: {frame_w}x{frame_h}")
                break
            candidate.release()

        if writer is None:
            raise RuntimeError("Could not open VideoWriter with avc1/mp4v codecs.")

        for index, frame in enumerate(tqdm(frame_generator2, total=frames_processed, desc="Annotating & writing")):
            player_tracks = all_tracks['player'].get(index, {-1: [None]*4})
            ball_tracks = all_tracks['ball'].get(index, [None]*4)
            referee_tracks = all_tracks['referee'].get(index, {-1: [None]*4})
            player_classids = all_tracks.get('player_classids', {}).get(index, None)

            if -1 in player_tracks:
                player_tracks = None
                player_classids = None
            if -1 in referee_tracks:
                referee_tracks = None
            if ball_tracks is None or not all(ball_tracks) or np.isnan(np.array(ball_tracks, dtype=float)).all():
                ball_tracks = None

            # Accumulate distance for each tracked player this frame
            MIN_MOVEMENT_PX = 10.0
            if player_tracks is not None:
                for tid, bbox in player_tracks.items():
                    if bbox[0] is not None and not np.isnan(float(bbox[0])):
                        gx = (float(bbox[0]) + float(bbox[2])) / 2.0
                        gy = float(bbox[3])
                        tid_int = int(tid)
                        if tid_int in player_prev_pos:
                            px, py = player_prev_pos[tid_int]
                            dist_px = np.sqrt((gx - px) ** 2 + (gy - py) ** 2)
                            if dist_px >= MIN_MOVEMENT_PX:
                                dist_m = dist_px * METERS_PER_PIXEL
                                if dist_m < 5.0:
                                    player_distances[tid_int] = player_distances.get(tid_int, 0.0) + dist_m
                                    # Speed = dist this frame / time for this frame → km/h
                                    speed_ms = dist_m * fps
                                    player_speeds[tid_int] = speed_ms * 3.6
                                player_prev_pos[tid_int] = (gx, gy)
                        else:
                            player_prev_pos[tid_int] = (gx, gy)

            player_detections, ball_detections, referee_detections = \
                self.tracking_pipeline.annotator_manager.convert_tracks_to_detections(
                    player_tracks, ball_tracks, referee_tracks, player_classids
                )
            annotated_frame = self.tracking_pipeline.annotator_manager.annotate_all(
                frame, player_detections, ball_detections, referee_detections,
                distance_dict=player_distances
            )

            # ── Target-player spotlight ───────────────────────────────────
            if target_player_id is not None and player_tracks is not None:
                tid_key = None
                for k in player_tracks.keys():
                    if int(k) == target_player_id:
                        tid_key = k
                        break
                if tid_key is not None:
                    bbox = player_tracks[tid_key]
                    if bbox[0] is not None and not np.isnan(float(bbox[0])):
                        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                        # Gold pulsing rectangle — thickness 3
                        cv2.rectangle(annotated_frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4),
                                      (0, 215, 255), 3)
                        # "TARGET" label above bbox
                        lbl = f"TARGET #{target_player_id}"
                        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        tx = max(0, x1 - 4)
                        ty = max(th + 4, y1 - 8)
                        cv2.rectangle(annotated_frame, (tx, ty - th - 4), (tx + tw + 4, ty + 2),
                                      (0, 0, 0), -1)
                        cv2.putText(annotated_frame, lbl, (tx + 2, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)

            # --- Tactical minimap overlay ---
            try:
                keypoints, _ = self.keypoint_pipeline.detect_keypoints_in_frame(frame)
                annotated_frame = self.tactical_pipeline.generate_minimap_overlay(
                    annotated_frame, player_detections, ball_detections, referee_detections,
                    keypoints, overlay_size=(420, 270), position="bottom-right",
                    target_player_id=target_player_id,
                    player_tracks=player_tracks,
                    target_trail=target_trail,
                    trail_max_len=TARGET_TRAIL_LEN,
                )
            except Exception:
                pass

            # ── Individual stats panel (bottom-left) ─────────────────────
            if target_player_id is not None:
                annotated_frame = self._draw_player_stats_panel(
                    annotated_frame, target_player_id, player_distances, player_speeds, locked_teams
                )

            if annotated_frame.shape[1] != frame_w or annotated_frame.shape[0] != frame_h:
                annotated_to_write = cv2.resize(annotated_frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
            else:
                annotated_to_write = annotated_frame
            writer.write(annotated_to_write)

        writer.release()

        # Summary
        total_time = time.time() - total_start_time
        requested_metrics = self._build_requested_metrics(
            frames_processed=frames_processed,
            total_time_s=total_time,
            source_fps=output_fps,
            detection_count_sums=detection_count_sums,
            detection_conf_sums=detection_conf_sums,
            detection_conf_counts=detection_conf_counts,
        )

        proc_fps = requested_metrics['fps']['processing_fps']
        source_fps = requested_metrics['fps']['source_video_fps']
        real_time = requested_metrics['fps']['real_time_capable']
        det_stats = requested_metrics['detection_metrics']['proxy_stats']
        
        print(f"\n{'='*70}")
        print(f"  COMPLETE SOCCER ANALYSIS PIPELINE - FINAL REPORT")
        print(f"{'='*70}")
        
        print(f"\n[PROCESSING PERFORMANCE]")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Frames processed: {frames_processed}")
        print(f"  Avg time/frame: {total_time/frames_processed:.3f}s")
        
        print(f"\n[FPS METRICS]")
        print(f"  Source video FPS: {source_fps}")
        print(f"  Processing FPS: {proc_fps:.2f}" if proc_fps is not None else "  Processing FPS: N/A")
        print(f"  Real-time capable: {real_time}")
        
        print(f"\n[DETECTION METRICS]")
        print(f"  Status: Ground truth required (no labeled annotations in pipeline)")
        print(f"  Precision: N/A (requires ground-truth boxes per frame)")
        print(f"  Recall: N/A (requires ground-truth boxes per frame)")
        print(f"  mAP: N/A (requires labeled ground-truth boxes per frame)")
        
        print(f"\n[DETECTION STATISTICS (Proxy Metrics)]")
        print(f"  Total detections:")
        print(f"    - Players: {det_stats['detections_total'].get('players', 0)}")
        print(f"    - Ball: {det_stats['detections_total'].get('ball', 0)}")
        print(f"    - Referees: {det_stats['detections_total'].get('referee', 0)}")
        print(f"  Mean confidence scores:")
        print(f"    - Players: {det_stats['mean_confidence'].get('players', 'N/A')}")
        print(f"    - Ball: {det_stats['mean_confidence'].get('ball', 'N/A')}")
        print(f"    - Referees: {det_stats['mean_confidence'].get('referees', 'N/A')}")
        
        print(f"\n[TRACKING METRICS]")
        print(f"  Status: Ground truth required (no labeled trajectories in pipeline)")
        print(f"  MOTA: N/A (requires labeled ground-truth trajectories & identities)")
        print(f"  IDF1: N/A (requires labeled ground-truth trajectories & identities)")
        
        print(f"\n[OUTPUT FILES]")
        print(f"  Video: {output_path}")
        print(f"  JSON data: {str(Path(output_dir) / (Path(output_path).stem + '_data.json'))}")
        print(f"  Player gallery: {gallery_path if gallery_path else 'N/A'}")
        
        print(f"\n{'='*70}")

        # Prepare exportable summary and raw tracks for JSON output
        def _to_serializable(o):
            # Convert numpy types and nested structures into plain Python types
            if o is None:
                return None
            if isinstance(o, dict):
                return {str(k): _to_serializable(v) for k, v in o.items()}
            if isinstance(o, (list, tuple, np.ndarray)):
                return [_to_serializable(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            if isinstance(o, (float, int)):
                return o
            try:
                return str(o)
            except Exception:
                return None

        export = {
            'video': str(video_path),
            'total_time_s': float(total_time),
            'frames_processed': int(frames_processed),
            'avg_time_per_frame_s': float(total_time / frames_processed) if frames_processed else None,
            'requested_metrics': _to_serializable(requested_metrics),
            'locked_teams': _to_serializable(locked_teams),
            'player_distances_m': _to_serializable(player_distances),
            'player_speeds_kmh': _to_serializable(player_speeds),
            'tracks': _to_serializable(all_tracks),
        }

        # Write JSON summary next to the output video
        try:
            json_name = Path(output_path).stem + "_data.json"
            json_path = str(Path(output_dir) / json_name)
            with open(json_path, 'w', encoding='utf-8') as fj:
                json.dump(export, fj, ensure_ascii=False, indent=2)
            print(f"Data exported to: {json_path}")
        except Exception as e:
            print(f"Warning: could not write JSON data file: {e}")

        # Write metrics to a separate clean JSON file
        try:
            metrics_file = {
                "precision": None,
                "recall": None,
                "mAP": None,
                "MOTA": None,
                "IDF1": None,
                "fps": {
                    "source_fps": requested_metrics['fps']['source_video_fps'],
                    "processing_fps": requested_metrics['fps']['processing_fps'],
                    "real_time_capable": requested_metrics['fps']['real_time_capable']
                },
                "detection_stats": {
                    "total_detections": det_stats['detections_total'],
                    "mean_confidence": det_stats['mean_confidence']
                },
                "notes": {
                    "precision": "Requires ground-truth labeled boxes per frame",
                    "recall": "Requires ground-truth labeled boxes per frame",
                    "mAP": "Requires ground-truth labeled boxes per frame",
                    "MOTA": "Requires ground-truth trajectories and identities",
                    "IDF1": "Requires ground-truth trajectories and identities"
                }
            }
            metrics_json_path = str(Path(output_dir) / "metrics.json")
            with open(metrics_json_path, 'w', encoding='utf-8') as fm:
                json.dump(metrics_file, fm, ensure_ascii=False, indent=2)
            print(f"Metrics exported to: {metrics_json_path}")
        except Exception as e:
            print(f"Warning: could not write metrics file: {e}")

        return output_path

    def _draw_player_stats_panel(self, frame: np.ndarray, target_id: int,
                                  distances: dict, speeds: dict, locked_teams: dict) -> np.ndarray:
        """Overlay a small stats panel in the bottom-left for the target player."""
        dist_m  = distances.get(target_id, 0.0)
        speed   = speeds.get(target_id, 0.0)
        team    = locked_teams.get(target_id, None)
        team_str = f"Team {'Red' if team == 0 else 'Blue' if team == 1 else '?'}"

        lines = [
            f"Player #{target_id}  ({team_str})",
            f"Distance : {dist_m:.0f} m",
            f"Speed    : {speed:.1f} km/h",
        ]

        PAD, LINE_H = 8, 22
        panel_w = 200
        panel_h = len(lines) * LINE_H + PAD * 2

        fh, fw = frame.shape[:2]
        x0, y0 = 10, fh - panel_h - 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Gold border
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 215, 255), 1)

        for i, line in enumerate(lines):
            color = (0, 215, 255) if i == 0 else (220, 220, 220)
            cv2.putText(frame, line, (x0 + PAD, y0 + PAD + (i + 1) * LINE_H - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return frame

    def analyze_realtime(self, source):
        """Display full tracking + team-color analysis in a live window.

        Workflow:
          1. Initialize models.
          2. Train team-color K-Means — uses test_video when source is a live
             camera (int index or RTSP/HTTP URL) since supervision can't sample
             frames from a live stream via get_video_frames_generator.
          3. Open the actual source with cv2.VideoCapture and process
             frame-by-frame: detect → ByteTrack → team assignment → cv2.imshow.
          Press 'q' to stop at any time.

        Args:
            source: Video file path, webcam index (int), or stream URL (str).
        """
        from constants import test_video as _train_video
        self.initialize_models()

        # Determine if source is a live stream (int index or non-file URL)
        source_is_live = isinstance(source, int) or (
            isinstance(source, str) and not Path(source).exists()
        )

        if source_is_live:
            print("Live source detected — training team colors from test_video...")
            self.tracking_pipeline.train_team_assignment_models(_train_video)
            # Run realtime loop without retraining
            self.tracking_pipeline.track_realtime(source, retrain=False)
        else:
            # File path — train + play from the same source
            self.tracking_pipeline.track_realtime(source, retrain=True)


if __name__ == "__main__":
    # ── Mode selection ────────────────────────────────────────────────────────
    # REALTIME = True  → live window (cv2.imshow), press 'q' to quit
    # REALTIME = False → full 2-pass analysis saved to a video file
    # ─────────────────────────────────────────────────────────────────────────
    REALTIME = False

    # ── How to spotlight a player (only used when REALTIME=False) ────────────
    # Step 1: Run once with TARGET_PLAYER=None to generate player_gallery.jpg.
    # Step 2: Re-run with TARGET_PLAYER=<the id> to spotlight that player.
    # ─────────────────────────────────────────────────────────────────────────
    TARGET_PLAYER = None   # ← change to an int (e.g. 5) after checking player_gallery.jpg

    print("Starting Soccer Analysis...")
    pipeline = CompleteSoccerAnalysisPipeline(model_path, keypoint_model_path)

    if REALTIME:
        # live_source can be 0 (webcam), a URL (IP Webcam / RTSP), or a video path.
        # Configure it in constants.py.
        pipeline.analyze_realtime(live_source)
    else:
        output_video = pipeline.analyze_video(test_video, frame_count=-1, target_player_id=TARGET_PLAYER)
        print(f"\nAnalysis finished! Output video: {output_video}")