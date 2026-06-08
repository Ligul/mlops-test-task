"""
Export the PyTorch recommendation model to ONNX format

Usage:
    python scripts/export_model.py [--output models/model.onnx] [--seed 67]

The exported model:
    input:  user_history  int64[history_len]  - item ids seen by user
    output: recommendations  int64[num_recommendations]  - recommended item ids
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

NUM_ITEMS = 10_000
EMBEDDING_DIM = 32
DEFAULT_NUM_RECOMMENDATIONS = 10
OPSET_VERSION = 17


class Model(nn.Module):
    _item_embeddings: torch.Tensor

    def __init__(self, num_recommendations: int = DEFAULT_NUM_RECOMMENDATIONS, device: str = "cpu") -> None:
        super().__init__()
        self.register_buffer("_item_embeddings", torch.rand((NUM_ITEMS, EMBEDDING_DIM), device=device))
        self._num_recommendations = num_recommendations

    def forward(self, user_history: torch.Tensor) -> torch.Tensor:
        user_embedding = self._item_embeddings[user_history].mean(dim=0)
        scores = user_embedding @ self._item_embeddings.T
        topk = torch.topk(scores, k=self._num_recommendations)
        return topk.indices


def export(output_path: Path, seed: int) -> None:
    torch.manual_seed(seed)

    model = Model()
    model.eval()

    dummy_input = torch.randint(0, NUM_ITEMS, (5,), dtype=torch.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("exporting to %s (opset %d) ...", output_path, OPSET_VERSION)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            input_names=["user_history"],
            output_names=["recommendations"],
            dynamic_axes={"user_history": {0: "history_len"}},
            opset_version=OPSET_VERSION,
            dynamo=False,
        )
    log.info("export done: %.2f MB", output_path.stat().st_size / 1024 / 1024)

    onnx.checker.check_model(onnx.load(str(output_path)))
    log.info("onnx graph check passed")


def verify(output_path: Path, seed: int) -> None:
    """Run the same input through PyTorch and ONNX Runtime, assert outputs match"""

    torch.manual_seed(seed)
    model = Model()
    model.eval()

    test_input = torch.randint(0, NUM_ITEMS, (7,), dtype=torch.int64)

    with torch.no_grad():
        pt_output: np.ndarray = model(test_input).numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_output = np.asarray(session.run(
        ["recommendations"],
        {"user_history": test_input.numpy()},
    )[0])

    if not np.array_equal(pt_output, ort_output):
        log.error("mismatch: pytorch=%s  onnxruntime=%s", pt_output, ort_output)
        sys.exit(1)

    log.info("verification passed: %s", pt_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("models/model.onnx"))
    parser.add_argument("--seed", type=int, default=67)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(args.output, args.seed)
    verify(args.output, args.seed)
