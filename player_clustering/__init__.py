import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from player_clustering.clustering import ClusteringManager, ColorKMeansClusterer

try:
    from player_clustering.embeddings import EmbeddingExtractor
except ImportError:
    print("Warning: EmbeddingExtractor could not be imported (transformers may not be available)")
    EmbeddingExtractor = None

__all__ = ["ClusteringManager", "ColorKMeansClusterer", "EmbeddingExtractor"]