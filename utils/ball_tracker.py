import cv2
import numpy as np


class BallKalmanTracker:
    """
    Kalman Filter for ball tracking without any model training.

    Uses a constant-velocity motion model: state = [cx, cy, vx, vy]
    Measurement: ball center [cx, cy]

    Why Kalman instead of interpolation:
    - Works frame-by-frame in real time (no post-processing pass needed)
    - Velocity-aware: ball continues in the direction it was moving
    - Handles occlusions naturally — coasts for up to `max_coasted_frames`
    - Same core algorithm used by SORT, ByteTrack, DeepSORT internally
    - No training data required — purely physics-based
    """

    def __init__(
        self,
        process_noise: float = 0.05,
        measurement_noise: float = 2.0,
        max_coasted_frames: int = 45,
    ):
        """
        Args:
            process_noise: Uncertainty in the motion model per frame.
                           Lower = smoother but slower to react to direction changes.
            measurement_noise: Uncertainty in YOLO detections.
                               Lower = trust detections more, less smoothing.
            max_coasted_frames: After this many consecutive miss frames, return None
                                (ball truly lost — avoid infinite coasting across the pitch).
        """
        # State vector: [cx, cy, vx, vy]  (center x/y + velocity x/y)
        self.kf = cv2.KalmanFilter(4, 2)

        # Transition matrix — constant velocity model
        # x_new = x + vx,  y_new = y + vy,  vx_new = vx,  vy_new = vy
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )

        # Measurement matrix — we only observe [cx, cy], not velocity
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )

        # Process noise covariance Q — how much can velocity change per frame
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # Measurement noise covariance R — detection uncertainty
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # Initial error covariance — start with high uncertainty until first detection
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0

        self.initialized = False
        self.last_bbox_size = [28.0, 28.0]  # Estimated ball size in pixels
        self.frames_since_detection = 0
        self.max_coasted_frames = max_coasted_frames

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1

    def _center_to_bbox(self, cx, cy):
        hw = self.last_bbox_size[0] / 2.0
        hh = self.last_bbox_size[1] / 2.0
        return [cx - hw, cy - hh, cx + hw, cy + hh]

    def _is_valid(self, bbox):
        if bbox is None:
            return False
        arr = np.array(bbox, dtype=float)
        return not np.any(np.isnan(arr)) and not np.any(arr == 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, ball_bbox):
        """
        Feed one frame's ball detection into the Kalman filter.

        Args:
            ball_bbox: [x1, y1, x2, y2] from YOLO, or None if not detected.

        Returns:
            Estimated [x1, y1, x2, y2], or None if tracker has lost the ball.
        """
        has_detection = self._is_valid(ball_bbox)

        if has_detection:
            cx, cy, w, h = self._bbox_center(ball_bbox)
            self.last_bbox_size = [max(w, 10.0), max(h, 10.0)]
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])

            if not self.initialized:
                # Warm-start state with first real detection
                self.kf.statePost = np.array(
                    [[cx], [cy], [0.0], [0.0]], dtype=np.float32
                )
                self.initialized = True

            self.kf.predict()
            corrected = self.kf.correct(measurement)
            self.frames_since_detection = 0

            pcx, pcy = float(corrected[0, 0]), float(corrected[1, 0])
            return self._center_to_bbox(pcx, pcy)

        else:
            # No detection this frame — predict from velocity
            if not self.initialized:
                return None

            self.frames_since_detection += 1
            if self.frames_since_detection > self.max_coasted_frames:
                return None  # Ball is truly lost

            predicted = self.kf.predict()
            pcx, pcy = float(predicted[0, 0]), float(predicted[1, 0])
            return self._center_to_bbox(pcx, pcy)

    def reset(self):
        """Reset tracker state (e.g. after a cut in the video)."""
        self.initialized = False
        self.frames_since_detection = 0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
