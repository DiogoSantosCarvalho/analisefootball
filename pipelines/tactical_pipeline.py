"""Tactical Analysis Pipeline for Soccer Analysis.

This pipeline coordinates tactical analysis functionality by combining
keypoint detection, player/ball/referee detection, and field coordinate transformations.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from pipelines.detection_pipeline import DetectionPipeline
from pipelines.keypoint_pipeline import KeypointPipeline
from pipelines.processing_pipeline import ProcessingPipeline
from tactical_analysis.homography import HomographyTransformer
from sports.common.view import ViewTransformer
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv


class TacticalPipeline:
    """Complete tactical analysis pipeline for soccer field coordinate transformations."""
    
    def __init__(self, keypoint_model_path: str, detection_model_path: str):
        """Initialize the tactical analysis pipeline.
        
        Args:
            keypoint_model_path: Path to the YOLO keypoint detection model
            detection_model_path: Path to the YOLO detection model
        """
        self.keypoint_pipeline = KeypointPipeline(keypoint_model_path)
        self.detection_pipeline = DetectionPipeline(detection_model_path)
        self.processing_pipeline = ProcessingPipeline()
        self.homography_transformer = HomographyTransformer()
        self.pitch_config = SoccerPitchConfiguration()

        # Temporal smoothing for minimap stability
        # We EMA the raw 3×3 homography matrix (ViewTransformer.m) directly.
        # This is more stable than averaging keypoints because it doesn't suffer
        # from different subsets of keypoints being visible each frame.
        self._H_smooth: np.ndarray = None    # smoothed homography matrix (3,3)
        self._H_ALPHA: float = 0.15          # EMA weight for new frame (lower = smoother)
        
    def initialize_models(self):
        """Initialize keypoint and detection models."""
        print("Initializing tactical analysis models...")
        self.keypoint_pipeline.initialize_model()
        self.detection_pipeline.initialize_model()
        print("Models initialized successfully")
    
    def detect_frame_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """Detect keypoints in a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Detected keypoints array with shape (N, 29, 3)
        """
        keypoints, _ = self.keypoint_pipeline.detect_keypoints_in_frame(frame)
        return keypoints
    
    def detect_frame_objects(self, frame: np.ndarray) -> Tuple[sv.Detections, sv.Detections, sv.Detections]:
        """Detect players, ball, and referees in a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (player_detections, ball_detections, referee_detections)
        """
        return self.detection_pipeline.detect_frame_objects(frame)
    
    def transform_keypoints_to_pitch(self, detected_keypoints: np.ndarray) -> ViewTransformer:
        """Transform frame keypoints to pitch coordinate system.
        
        Args:
            detected_keypoints: Array of shape (1, 29, 3) with [x, y, confidence]
            
        Returns:
            ViewTransformer object for frame-to-pitch transformation
        """
        return self.homography_transformer.transform_to_pitch_keypoints(detected_keypoints)
    
    def transform_detections_to_pitch(self, detections: sv.Detections, 
                                    view_transformer: ViewTransformer) -> np.ndarray:
        """Transform detection bounding boxes to pitch coordinates.
        
        Args:
            detections: Detection results from YOLO
            view_transformer: ViewTransformer for frame-to-pitch conversion
            
        Returns:
            Array of transformed center points in pitch coordinates (N, 2)
        """
        if detections is None or len(detections.xyxy) == 0 or view_transformer is None:
            return np.array([]).reshape(0, 2)
        
        # Get center points of bounding boxes
        bboxes = detections.xyxy
        center_points = np.array([
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] 
            for bbox in bboxes
        ])
        
        # Transform to pitch coordinates
        pitch_points = self.homography_transformer.transform_points_to_pitch(
            center_points, view_transformer
        )
        
        return pitch_points if pitch_points is not None else np.array([]).reshape(0, 2)
    
    def create_tactical_frame(self, player_points: np.ndarray, ball_points: np.ndarray, 
                            referee_points: np.ndarray, team2_points: np.ndarray = None, 
                            frame_size: Tuple[int, int] = (1050, 680)) -> np.ndarray:
        """Create a tactical view frame showing positions on the pitch.
        
        Args:
            player_points: Team 1 player positions in pitch coordinates (N, 2)
            ball_points: Ball positions in pitch coordinates (N, 2) 
            referee_points: Referee positions in pitch coordinates (N, 2)
            team2_points: Team 2 player positions in pitch coordinates (N, 2)
            frame_size: Size of output tactical frame (width, height)
            
        Returns:
            Tactical view frame as numpy array
        """
        # Create pitch visualization
        pitch_frame = draw_pitch(self.pitch_config)
        
        # Draw team 1 player positions (class_id 0)
        if len(player_points) > 0:
            for point in player_points:
                if not np.isnan(point).any():
                    # Convert pitch coordinates to frame coordinates
                    x_ratio = point[0] / 12000  # Pitch width is 12000 units
                    y_ratio = point[1] / 7000   # Pitch height is 7000 units
                    
                    frame_x = int(x_ratio * frame_size[0])
                    frame_y = int(y_ratio * frame_size[1])
                    
                    # Draw team 1 player as circle
                    cv2.circle(pitch_frame, (frame_x, frame_y), 8, (128, 0, 128), -1)  # Purple for team 1
        
        # Draw team 2 player positions (class_id 1)
        if team2_points is not None and len(team2_points) > 0:
            for point in team2_points:
                if not np.isnan(point).any():
                    # Convert pitch coordinates to frame coordinates
                    x_ratio = point[0] / 12000  # Pitch width is 12000 units
                    y_ratio = point[1] / 7000   # Pitch height is 7000 units
                    
                    frame_x = int(x_ratio * frame_size[0])
                    frame_y = int(y_ratio * frame_size[1])
                    
                    # Draw team 2 player as circle
                    cv2.circle(pitch_frame, (frame_x, frame_y), 8, (0, 0, 255), -1)  # Red for team 2
        
        # Draw ball positions
        if len(ball_points) > 0:
            for point in ball_points:
                if not np.isnan(point).any():
                    x_ratio = point[0] / 12000
                    y_ratio = point[1] / 7000
                    
                    frame_x = int(x_ratio * frame_size[0])
                    frame_y = int(y_ratio * frame_size[1])
                    
                    # Draw ball as circle
                    cv2.circle(pitch_frame, (frame_x, frame_y), 6, (255, 255, 255), -1)  # White for ball
        
        # Draw referee positions
        if len(referee_points) > 0:
            for point in referee_points:
                if not np.isnan(point).any():
                    x_ratio = point[0] / 12000
                    y_ratio = point[1] / 7000
                    
                    frame_x = int(x_ratio * frame_size[0])
                    frame_y = int(y_ratio * frame_size[1])
                    
                    # Draw referee as square
                    cv2.rectangle(pitch_frame, (frame_x-6, frame_y-6), 
                                (frame_x+6, frame_y+6), (0, 0, 0), -1)  # Black for referees
        
        return pitch_frame

    def process_detections_for_tactical_analysis(self, player_detections, ball_detections, referee_detections, keypoints):
        """Process Detections and Keypoints for tactical analysis.
        
        Args:
            player_detections : Detections for players
            ball_detections : Detections for ball
            referee_detections : Detections for referee
            keypoints : Keypoints of the soccer pitch
            
        Returns:
            Tuple of (tactical_frame, metadata_dict)
        """
        # Get transformation matrix
        view_transformer = self.transform_keypoints_to_pitch(keypoints)

        # Separate player detections
        team_1_detections = player_detections[player_detections.class_id == 0]
        team_2_detections = player_detections[player_detections.class_id == 1]

        # Transform detection
        team1_points = self.transform_detections_to_pitch(team_1_detections, view_transformer)
        team2_points = self.transform_detections_to_pitch(team_2_detections, view_transformer)
        ball_pitch_points = self.transform_detections_to_pitch(ball_detections, view_transformer)
        referee_pitch_points = self.transform_detections_to_pitch(referee_detections, view_transformer)
        
        # Create tactical frame with team separation
        tactical_frame = self.create_tactical_frame(
            team1_points, ball_pitch_points, referee_pitch_points, team2_points
        )
        
        # Prepare metadata
        metadata = {
            'num_players': len(team1_points) + len(team2_points),
            'num_team1_players': len(team1_points),
            'num_team2_players': len(team2_points),
            'num_balls': len(ball_pitch_points),
            'num_referees': len(referee_pitch_points),
            'transformation_valid': view_transformer is not None,
            'team1_positions': team1_points.tolist() if len(team1_points) > 0 else [],
            'team2_positions': team2_points.tolist() if len(team2_points) > 0 else [],
            'ball_positions': ball_pitch_points.tolist() if len(ball_pitch_points) > 0 else [],
            'referee_positions': referee_pitch_points.tolist() if len(referee_pitch_points) > 0 else []
        }
        
        return tactical_frame, metadata
    
    def process_frame_for_tactical_analysis(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process a single frame for tactical analysis.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (tactical_frame, metadata_dict)
        """
        # Detect keypoints and objects
        keypoints = self.detect_frame_keypoints(frame)
        player_detections, ball_detections, referee_detections = self.detect_frame_objects(frame)

        # Process the Detections
        tactical_frame, metadata = self.process_detections_for_tactical_analysis(player_detections, ball_detections, referee_detections, keypoints)

        return tactical_frame, metadata
    
    def create_overlay_frame(self, original_frame: np.ndarray, tactical_frame: np.ndarray, 
                           overlay_size: Tuple[int, int] = (300, 200), position: str = "top-right") -> np.ndarray:
        """Create a frame with tactical overlay on original video.
        
        Args:
            original_frame: Original video frame
            tactical_frame: Tactical analysis frame
            overlay_size: Size of the overlay (width, height)
            position: Position of overlay ("top-right", "top-left", "bottom-right", "bottom-left")
            
        Returns:
            Combined frame with tactical overlay
        """
        # Resize tactical frame for overlay
        resized_tactical = cv2.resize(tactical_frame, overlay_size)
        
        # Copy original frame
        combined_frame = original_frame.copy()
        h, w = combined_frame.shape[:2]
        overlay_h, overlay_w = resized_tactical.shape[:2]
        
        # Calculate position
        if position == "top-right":
            start_y, end_y = 10, 10 + overlay_h
            start_x, end_x = w - overlay_w - 10, w - 10
        elif position == "top-left":
            start_y, end_y = 10, 10 + overlay_h
            start_x, end_x = 10, 10 + overlay_w
        elif position == "bottom-right":
            start_y, end_y = h - overlay_h - 10, h - 10
            start_x, end_x = w - overlay_w - 10, w - 10
        elif position == "bottom-left":
            start_y, end_y = h - overlay_h - 10, h - 10
            start_x, end_x = 10, 10 + overlay_w
        else:
            start_y, end_y = 10, 10 + overlay_h
            start_x, end_x = w - overlay_w - 10, w - 10
        
        # Add tactical overlay to original frame
        combined_frame[start_y:end_y, start_x:end_x] = resized_tactical
        
        # Add border around overlay
        cv2.rectangle(combined_frame, (start_x-2, start_y-2), (end_x+2, end_y+2), (255, 255, 255), 2)
        
        # Add label
        cv2.putText(combined_frame, "Tactical View", (start_x, start_y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return combined_frame

    def _det_to_pitch_bottom_center(self, detections: sv.Detections,
                                     view_transformer) -> np.ndarray:
        """Transform bottom-center of bounding boxes to pitch coordinates.

        Using bottom-center is more accurate than center for players because
        their feet are actually on the ground (projection plane).
        """
        if detections is None or len(detections.xyxy) == 0 or view_transformer is None:
            return np.array([]).reshape(0, 2)
        bboxes = detections.xyxy
        bottom_centers = np.array([
            [(bbox[0] + bbox[2]) / 2.0, bbox[3]]
            for bbox in bboxes
        ], dtype=np.float32)
        result = self.homography_transformer.transform_points_to_pitch(bottom_centers, view_transformer)
        return result if result is not None else np.array([]).reshape(0, 2)

    def _draw_minimap_colored(self, team0_pts: np.ndarray, team1_pts: np.ndarray,
                               ball_pts: np.ndarray, referee_pts: np.ndarray) -> np.ndarray:
        """Draw 2-D pitch minimap with team colors matching the video overlay.

        Team 0 → red  (#FF2222)   BGR: (34, 34, 255)
        Team 1 → blue (#2266FF)   BGR: (255, 102, 34)
        Ball   → white             BGR: (255, 255, 255)
        Ref    → yellow            BGR: (0, 255, 255)
        """
        pitch_frame = draw_pitch(self.pitch_config)
        h, w = pitch_frame.shape[:2]

        PITCH_W, PITCH_H = 12000.0, 7000.0  # coordinate system units

        def to_px(pt):
            x = int(np.clip(pt[0] / PITCH_W * w, 0, w - 1))
            y = int(np.clip(pt[1] / PITCH_H * h, 0, h - 1))
            return (x, y)

        COLOR_TEAM0 = (34, 34, 255)      # red   (BGR)
        COLOR_TEAM1 = (255, 102, 34)     # blue  (BGR)
        COLOR_BALL  = (255, 255, 255)    # white (BGR)
        COLOR_REF   = (0, 200, 255)      # yellow(BGR)

        for pts, color, r in [
            (team0_pts,   COLOR_TEAM0, 9),
            (team1_pts,   COLOR_TEAM1, 9),
            (referee_pts, COLOR_REF,   8),
        ]:
            for pt in pts:
                if not np.isnan(pt).any() and 0 <= pt[0] <= PITCH_W and 0 <= pt[1] <= PITCH_H:
                    cv2.circle(pitch_frame, to_px(pt), r, color, -1)
                    cv2.circle(pitch_frame, to_px(pt), r, (0, 0, 0), 1)  # thin outline

        for pt in ball_pts:
            if not np.isnan(pt).any() and 0 <= pt[0] <= PITCH_W and 0 <= pt[1] <= PITCH_H:
                cv2.circle(pitch_frame, to_px(pt), 6, COLOR_BALL, -1)
                cv2.circle(pitch_frame, to_px(pt), 6, (0, 0, 0), 1)

        return pitch_frame

    def generate_minimap_overlay(self, original_frame: np.ndarray,
                                  player_detections: sv.Detections,
                                  ball_detections: sv.Detections,
                                  referee_detections: sv.Detections,
                                  keypoints: np.ndarray,
                                  overlay_size: Tuple[int, int] = (420, 270),
                                  position: str = "bottom-right",
                                  target_player_id: int = None,
                                  player_tracks: dict = None,
                                  target_trail: list = None,
                                  trail_max_len: int = 60) -> np.ndarray:
        """Generate a 2-D tactical minimap and overlay it on the video frame.

        Returns the original frame unchanged if homography cannot be computed
        (e.g. when not enough pitch keypoints are visible).

        Args:
            original_frame: Annotated video frame to overlay on
            player_detections: Current frame's player detections (with class_id 0/1 for teams)
            ball_detections: Current frame's ball detections
            referee_detections: Current frame's referee detections
            keypoints: Pitch keypoints from keypoint model (shape N×29×3)
            overlay_size: (width, height) of the minimap overlay in pixels
            position: Corner to place the minimap ("bottom-right", "bottom-left", etc.)
            target_player_id: tracker_id of the player to spotlight with a trail
            player_tracks: {tracker_id: [x1,y1,x2,y2]} for the current frame (for trail update)
            target_trail: mutable list of past pitch-coords for the target player
            trail_max_len: how many past positions to keep in the trail

        Returns:
            Frame with minimap overlay, or original frame if homography failed
        """
        # ── Compute raw homography from this frame's keypoints ─────────────────
        view_transformer = self.homography_transformer.transform_to_pitch_keypoints(keypoints)
        if view_transformer is None:
            # No valid homography this frame — reuse last smoothed matrix if we have one
            if self._H_smooth is None:
                return original_frame
            H = self._H_smooth
        else:
            H_new = view_transformer.m.astype(np.float64)   # raw 3×3 matrix
            if self._H_smooth is None:
                # First valid frame — initialise directly
                self._H_smooth = H_new.copy()
            else:
                # EMA on the matrix entries: smoother = lower _H_ALPHA
                self._H_smooth = self._H_ALPHA * H_new + (1.0 - self._H_ALPHA) * self._H_smooth
            H = self._H_smooth

        # ── Transform bottom-center of bboxes using the smoothed H ─────────────
        def _transform_with_H(detections):
            if detections is None or len(detections.xyxy) == 0:
                return np.array([]).reshape(0, 2)
            bboxes = detections.xyxy
            pts = np.array([[(b[0]+b[2])/2.0, b[3]] for b in bboxes], dtype=np.float32)
            res = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H.astype(np.float32))
            return res.reshape(-1, 2).astype(np.float32)

        # Separate teams by class_id
        if player_detections is not None and len(player_detections.xyxy) > 0:
            team0_det = player_detections[player_detections.class_id == 0]
            team1_det = player_detections[player_detections.class_id == 1]
        else:
            team0_det = sv.Detections.empty()
            team1_det = sv.Detections.empty()

        team0_pts = _transform_with_H(team0_det)
        team1_pts = _transform_with_H(team1_det)
        ball_pts  = _transform_with_H(ball_detections)
        ref_pts   = _transform_with_H(referee_detections)

        minimap = self._draw_minimap_colored(team0_pts, team1_pts, ball_pts, ref_pts)

        # ── Target player trail ────────────────────────────────────────────────
        # If a target player is specified we draw their movement trail on the minimap:
        # a fading polyline of past positions + a gold star at the current position.
        if target_player_id is not None and target_trail is not None and player_tracks is not None:
            # Find the bbox for the target player in this frame
            tgt_bbox = None
            for k, v in player_tracks.items():
                if int(k) == target_player_id and v[0] is not None:
                    try:
                        if not np.isnan(float(v[0])):
                            tgt_bbox = v
                    except Exception:
                        pass
                    break

            if tgt_bbox is not None:
                # Transform the player's bottom-center to pitch coordinates
                bx = np.array([[(float(tgt_bbox[0]) + float(tgt_bbox[2])) / 2.0, float(tgt_bbox[3])]],
                               dtype=np.float32)
                res = cv2.perspectiveTransform(bx.reshape(-1, 1, 2), H.astype(np.float32))
                px_pitch = res.reshape(-1, 2)[0]
                target_trail.append((float(px_pitch[0]), float(px_pitch[1])))
                if len(target_trail) > trail_max_len:
                    target_trail.pop(0)

            # Draw the trail on the minimap (before scaling to overlay_size)
            mh, mw = minimap.shape[:2]
            PITCH_W, PITCH_H = 12000.0, 7000.0

            def _to_px_mm(pt):
                return (int(np.clip(pt[0] / PITCH_W * mw, 0, mw - 1)),
                        int(np.clip(pt[1] / PITCH_H * mh, 0, mh - 1)))

            n = len(target_trail)
            for i in range(1, n):
                alpha = i / n  # older = more transparent
                color_trail = (0, int(180 * alpha), int(255 * alpha))  # gold-ish fade
                thickness = max(1, int(3 * alpha))
                p1 = _to_px_mm(target_trail[i - 1])
                p2 = _to_px_mm(target_trail[i])
                cv2.line(minimap, p1, p2, color_trail, thickness, cv2.LINE_AA)

            # Gold dot for current position
            if n > 0:
                curr_px = _to_px_mm(target_trail[-1])
                cv2.circle(minimap, curr_px, 7, (0, 215, 255), -1)
                cv2.circle(minimap, curr_px, 7, (0, 0, 0), 1)

        return self.create_overlay_frame(original_frame, minimap, overlay_size, position)

    def create_side_by_side_frame(self, original_frame: np.ndarray, tactical_frame: np.ndarray, 
                                metadata: Dict, frame_height: int = 480) -> np.ndarray:
        """Create a side-by-side frame with original video and tactical view.
        
        Args:
            original_frame: Original video frame
            tactical_frame: Tactical analysis frame
            metadata: Metadata dictionary with detection counts and status
            frame_height: Height for resizing frames
            
        Returns:
            Combined side-by-side frame with metadata overlay
        """
        # Calculate dimensions maintaining aspect ratio
        frame_width = int(original_frame.shape[1] * (frame_height / original_frame.shape[0]))
        tactical_width = int(tactical_frame.shape[1] * (frame_height / tactical_frame.shape[0]))
        
        # Resize frames
        resized_frame = cv2.resize(original_frame, (frame_width, frame_height))
        resized_tactical = cv2.resize(tactical_frame, (tactical_width, frame_height))
        
        # Create combined display
        combined_frame = np.hstack([resized_frame, resized_tactical])
        
        # Add text overlay with metadata
        text_y = 30
        cv2.putText(combined_frame, f"Players: {metadata['num_players']}", 
                  (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        cv2.putText(combined_frame, f"Ball: {metadata['num_balls']}", 
                  (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        cv2.putText(combined_frame, f"Referees: {metadata['num_referees']}", 
                  (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        cv2.putText(combined_frame, f"Transform: {'Valid' if metadata['transformation_valid'] else 'Invalid'}", 
                  (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                  (0, 255, 0) if metadata['transformation_valid'] else (0, 0, 255), 2)
        
        # Add labels for each side
        cv2.putText(combined_frame, "Original Video", (10, frame_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined_frame, "Tactical View", (frame_width + 10, frame_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add divider line
        cv2.line(combined_frame, (frame_width, 0), (frame_width, frame_height), (255, 255, 255), 2)
        
        return combined_frame

    def analyze_video(self, video_path: str, output_path: str, frame_count: int = -1, 
                     output_mode: str = "overlay", overlay_position: str = "top-right"):
        """Analyze a complete video and create tactical analysis output.
        
        Args:
            video_path: Path to input video
            output_path: Path to save tactical analysis video
            frame_count: Number of frames to process (-1 for all frames)
            output_mode: Output mode - "overlay", "side-by-side", or "tactical-only"
            overlay_position: Position of tactical overlay (for overlay mode)
        """
        self.initialize_models()
        
        print(f"Reading video frames...")
        video_frames = self.processing_pipeline.read_video_frames(video_path, frame_count)
        
        print(f"Processing frames for tactical analysis in {output_mode} mode...")
        output_frames = []
        all_metadata = []
        
        for i, frame in enumerate(video_frames):
            tactical_frame, metadata = self.process_frame_for_tactical_analysis(frame)
            all_metadata.append(metadata)
            
            # Create output frame based on mode
            if output_mode == "overlay":
                # Create overlay frame with tactical view on original video
                output_frame = self.create_overlay_frame(frame, tactical_frame, position=overlay_position)
                
            elif output_mode == "side-by-side":
                # Create side-by-side frame with metadata
                output_frame = self.create_side_by_side_frame(frame, tactical_frame, metadata)
                
            else:  # tactical-only
                # Use only tactical frame
                output_frame = tactical_frame
            
            output_frames.append(output_frame)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(video_frames)} frames")
        
        print("Writing tactical analysis video...")
        self.processing_pipeline.write_video_output(output_frames, output_path)
        
        print(f"Tactical analysis complete! Output saved to: {output_path}")
        return all_metadata
    
    def analyze_realtime(self, video_path: str, display_mode: str = "overlay"):
        """Run real-time tactical analysis on a video stream.
        
        Args:
            video_path: Path to input video or camera index (0 for webcam)
            display_mode: Display mode - "overlay", "side-by-side", or "tactical-only"
        """
        self.initialize_models()

        print("Opening video stream...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")

        print("Starting real-time tactical analysis. Press 'q' to quit.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Process frame for tactical analysis
            tactical_frame, metadata = self.process_frame_for_tactical_analysis(frame)
            
            # Display frames based on mode
            if display_mode == "overlay":
                # Display original with tactical overlay
                combined_frame = self.create_overlay_frame(frame, tactical_frame)
                cv2.imshow("Soccer Tactical Analysis - Overlay", combined_frame)
                
            elif display_mode == "side-by-side":
                # Create side-by-side display using the dedicated function
                combined_frame = self.create_side_by_side_frame(frame, tactical_frame, metadata)
                cv2.imshow("Soccer Tactical Analysis - Side by Side", combined_frame)
                
            else:  # tactical-only
                # Display only tactical frame
                cv2.imshow("Soccer Tactical Analysis - Tactical View", tactical_frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Real-time tactical analysis stopped.")


if __name__ == "__main__":
    from keypoint_detection.keypoint_constants import keypoint_model_path
    from player_detection.detection_constants import model_path
    from constants import test_video
    
    # Example usage
    pipeline = TacticalPipeline(keypoint_model_path, model_path)
    
    # Choose analysis mode (uncomment desired option)
    
    # Video Analysis Options:
    
    # Option 1: Video analysis with overlay (saves to file)
    # output_path = test_video.replace('.mp4', '_tactical_overlay.mp4')
    # pipeline.analyze_video(test_video, output_path, frame_count=300, output_mode="overlay")
    
    # Option 2: Video analysis side-by-side with metadata (saves to file)
    # output_path = test_video.replace('.mp4', '_tactical_sidebyside.mp4')
    # pipeline.analyze_video(test_video, output_path, frame_count=300, output_mode="side-by-side")
    
    # Option 3: Video analysis tactical-only (saves to file)
    # output_path = test_video.replace('.mp4', '_tactical_only.mp4')
    # pipeline.analyze_video(test_video, output_path, frame_count=300, output_mode="tactical-only")
    
    # Real-time Analysis Options:
    
    # Option 4: Real-time analysis with overlay
    # pipeline.analyze_realtime(test_video, display_mode="overlay")
    
    # Option 5: Real-time analysis side-by-side
    # pipeline.analyze_realtime(test_video, display_mode="side-by-side")
    
    # Option 6: Real-time analysis tactical-only
    pipeline.analyze_realtime(test_video, display_mode="tactical-only")