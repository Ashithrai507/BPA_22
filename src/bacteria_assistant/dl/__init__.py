from __future__ import annotations

import cv2
import torch

# OpenCV and torch both link their own OpenMP thread pools; running them
# concurrently in one process causes intermittent memory corruption / segfaults
# on macOS. Pin both to single-threaded CPU operation — heavy compute runs on
# MPS, so this has negligible impact on training throughput.
cv2.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from .model import EmbeddingModel, build_embedding_model

__all__ = ["EmbeddingModel", "build_embedding_model"]
