import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import cv2
from sklearn.cluster import KMeans

# umap and EmbeddingExtractor are heavy optional deps — imported lazily
# inside ClusteringManager.__init__ so ColorKMeansClusterer works without them.


class ColorKMeansClusterer:
    """
    Fast team assignment using K-Means on jersey colors (top half of player crop).
    Much faster than SigLIP embeddings - no deep learning model needed.
    
    Approach:
    1. Crop top half of each player bounding box (jersey region, avoids legs/grass)
    2. Convert to HSV color space for better color discrimination
    3. Run K-Means(k=2) to find the 2 dominant colors in each crop
    4. Identify which cluster is the background (grass) by checking corner pixels
    5. The non-background cluster center = jersey color
    6. Collect jersey colors from multiple training frames
    7. Run K-Means(k=2) on all collected colors to define team 0 and team 1
    """

    def __init__(self):
        self.team_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.is_trained = False
        self.team_colors = None  # shape (2, 3) - mean color per team in HSV

    def _get_jersey_color(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Extract dominant jersey color from a single player crop using HSV.

        Strategy:
        1. Use the central vertical strip of the top-60% of the crop — this
           avoids edges which are mostly background/grass.
        2. Mask out grass-coloured pixels in HSV before clustering. Grass is
           reliably H≈35-85, S>40, so filtering it leaves only jersey/skin.
        3. If too few non-grass pixels survive, fall back to the full top-half
           with the original corner-based background detection.
        """
        h, w = crop_bgr.shape[:2]
        if h < 8 or w < 8:
            return np.array([60, 100, 180], dtype=np.float32)  # neutral fallback

        # --- Region: top 60 %, centre 60 % of width (away from background edges)
        top    = crop_bgr[:int(h * 0.60), :]
        cx     = w // 2
        half_w = max(2, int(w * 0.30))
        centre = top[:, max(0, cx - half_w): cx + half_w]

        hsv_c = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

        # --- Grass mask: H in [30, 90], S > 35  (covers most natural pitches)
        H, S = hsv_c[:, 0], hsv_c[:, 1]
        grass_mask = (H >= 30) & (H <= 90) & (S > 35)
        jersey_px  = hsv_c[~grass_mask]

        if len(jersey_px) < 10:
            # Fallback: full top-half, remove grass the same way
            hsv_full = cv2.cvtColor(top, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
            H2, S2   = hsv_full[:, 0], hsv_full[:, 1]
            jersey_px = hsv_full[~((H2 >= 30) & (H2 <= 90) & (S2 > 35))]

        if len(jersey_px) < 6:
            return np.array([60, 100, 180], dtype=np.float32)  # can't determine

        # --- K-Means(k=2) on non-grass pixels: separates jersey from skin/shadow
        km = KMeans(n_clusters=min(2, len(jersey_px)), random_state=0, n_init=5, max_iter=100)
        km.fit(jersey_px)

        if len(km.cluster_centers_) == 1:
            return km.cluster_centers_[0]

        # Pick the cluster with higher saturation — jerseys are more saturated than skin
        s0, s1 = km.cluster_centers_[0][1], km.cluster_centers_[1][1]
        return km.cluster_centers_[0] if s0 >= s1 else km.cluster_centers_[1]

    def get_jersey_colors(self, frame: np.ndarray, player_detections) -> np.ndarray:
        """Extract jersey colors for all players in a frame."""
        colors = []
        for bbox in player_detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                colors.append(np.array([0, 0, 128], dtype=np.float32))
            else:
                colors.append(self._get_jersey_color(crop))
        return np.array(colors) if colors else np.empty((0, 3), dtype=np.float32)

    def train_from_frames(self, frames_and_detections: list):
        """Train team K-Means from multiple frames for robust team identification.
        
        Args:
            frames_and_detections: list of (frame, player_detections) tuples
        """
        all_colors = []
        for frame, player_detections in frames_and_detections:
            if len(player_detections.xyxy) > 0:
                colors = self.get_jersey_colors(frame, player_detections)
                all_colors.append(colors)
        
        if not all_colors:
            return
        
        all_colors = np.vstack(all_colors)
        if len(all_colors) >= 2:
            self.team_kmeans.fit(all_colors)
            self.team_colors = self.team_kmeans.cluster_centers_
            self.is_trained = True
            print(f"  Team color clustering trained on {len(all_colors)} player crops")
            # Print team colors in BGR for reference
            for i, hsv_color in enumerate(self.team_colors):
                bgr = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
                print(f"    Team {i}: BGR=({bgr[0]},{bgr[1]},{bgr[2]}) -> {'Red' if i == 0 else 'Blue'} annotation")

    def train(self, frame: np.ndarray, player_detections) -> np.ndarray:
        """Fit team K-Means on the jersey colors of players in a given frame."""
        colors = self.get_jersey_colors(frame, player_detections)
        if len(colors) >= 2:
            labels = self.team_kmeans.fit_predict(colors)
            self.team_colors = self.team_kmeans.cluster_centers_
            self.is_trained = True
            return labels
        return np.zeros(len(colors), dtype=int)

    def predict(self, frame: np.ndarray, player_detections) -> np.ndarray:
        """Assign team labels to all players using trained color model."""
        if not self.is_trained:
            return self.train(frame, player_detections)
        colors = self.get_jersey_colors(frame, player_detections)
        if len(colors) == 0:
            return np.array([], dtype=int)
        return self.team_kmeans.predict(colors)


class ClusteringManager:
    """
    Manager class for player clustering and team assignment.
    Handles UMAP dimensionality reduction and K-means clustering.
    """
    
    def __init__(self, n_components=3, n_clusters=2):
        """
        Initialize clustering models.
        
        Args:
            n_components: Number of components for UMAP reduction
            n_clusters: Number of clusters for K-means (typically 2 for teams)
        """
        import umap.umap_ as umap
        from .embeddings import EmbeddingExtractor
        self.reducer = umap.UMAP(n_components=n_components)
        self.cluster_model = KMeans(n_clusters=n_clusters)
        self.embedding_extractor = EmbeddingExtractor()
        
    def project_embeddings(self, data, train=False):
        """
        Project embeddings to lower-dimensional space using UMAP.
        
        Args:
            data: High-dimensional embeddings
            train: Whether to fit the reducer or just transform
            
        Returns:
            Tuple of (reduced_embeddings, reducer)
        """
        if train:
            reduced_embeddings = self.reducer.fit_transform(data)
        else:
            reduced_embeddings = self.reducer.transform(data)
        
        return reduced_embeddings, self.reducer
    
    def cluster_embeddings(self, data, train=False):
        """
        Cluster embeddings using K-means.
        
        Args:
            data: Reduced embeddings for clustering
            train: Whether to fit the model or just predict
            
        Returns:
            Tuple of (cluster_labels, cluster_model)
        """
        if train:
            cluster_labels = self.cluster_model.fit_predict(data)
        else:
            cluster_labels = self.cluster_model.predict(data)
        
        return cluster_labels, self.cluster_model
    
    def process_batch(self, crop_batches, train=False):
        """
        Process a batch of crops through the full clustering pipeline.
        
        Args:
            crop_batches: Batched player crops
            train: Whether to train models or just predict
            
        Returns:
            Tuple of (cluster_labels, reducer, cluster_model)
        """
        # Extract embeddings
        embeddings = self.embedding_extractor.get_embeddings(crop_batches)
        
        # Reduce dimensionality
        reduced_embeddings, _ = self.project_embeddings(embeddings, train=train)
        
        # Cluster
        cluster_labels, _ = self.cluster_embeddings(reduced_embeddings, train=train)
        
        return cluster_labels, self.reducer, self.cluster_model


    def train_clustering_models(self, crops):
        """
        Train the UMAP and K-means models on player crops.
        
        Args:
            crops: List of player crop images
            
        Returns:
            Tuple of (cluster_labels, reducer, cluster_model)
        """
        if crops is None or len(crops) == 0:
            raise ValueError("Crops list cannot be None or empty")
        
        # Process crops and train models
        crop_batches = self.embedding_extractor.create_batches(crops, 24)
        cluster_labels, reducer, cluster_model = self.process_batch(crop_batches, train=True)
        
        return cluster_labels, reducer, cluster_model

    def get_cluster_labels(self, frame, player_detections, crops=None):
        """
        Get cluster labels for players in a single frame.
        
        Args:
            frame: Input video frame
            player_detections: Player detection results
            crops: Pre-extracted crops (optional)
            
        Returns:
            Cluster labels for the players
        """
        if crops is None:
            # Extract player crops
            crops = self.embedding_extractor.get_player_crops(frame, player_detections)
        
        # Get cluster assignments
        cluster_labels, _, _ = self.process_batch([crops], train=False)
        
        return cluster_labels