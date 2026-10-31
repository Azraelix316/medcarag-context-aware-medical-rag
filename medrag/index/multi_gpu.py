"""Multi-GPU bi-encoder embedding.

`sentence_transformers` drives one device per model, so a multi-GPU box is left
sitting idle by the single-GPU path. This module replicates the encoder across
every visible CUDA device and shards each batch across them, which nearly
doubles throughput on a two-GPU host.

Threads rather than processes are used because PyTorch releases the GIL during
CUDA kernel launches, so the shards genuinely run concurrently while avoiding
the serialisation cost of shipping weights across processes. Ordering is
reassembled by index, so the output is deterministic and identical to the
single-GPU path up to float non-determinism.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ..config import EmbedConfig
from .embeddings import DenseEmbedder, gpu_compute_capabilities, select_fp16

log = logging.getLogger(__name__)


class MultiGpuEmbedder:
    """Replicates a bi-encoder across all visible CUDA devices."""

    def __init__(self, cfg: EmbedConfig | None = None) -> None:
        self.cfg = cfg or EmbedConfig()
        self.n_devices = 0
        self.dim = self.cfg.dim
        self.fp16 = False
        self._models: list[Any] = []
        self._tokenizers: list[Any] = []
        self._pool = ThreadPoolExecutor(max_workers=max(1, self.n_devices))

    def available_devices(self) -> int:
        try:
            import torch

            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:  # noqa: BLE001
            return 0

    def load(self) -> "MultiGpuEmbedder":
        if self._models:
            return self
        n = self.available_devices()
        if n < 2:
            raise RuntimeError(f"MultiGpuEmbedder needs >=2 CUDA devices, found {n}")

        from sentence_transformers import SentenceTransformer

        self.fp16 = select_fp16(self.cfg.fp16, "cuda")
        self.n_devices = n
        caps = gpu_compute_capabilities()
        log.info("loading %s on %d GPUs %s (fp16=%s)", self.cfg.model_name, n, caps, self.fp16)

        for i in range(n):
            model = SentenceTransformer(self.cfg.model_name, device=f"cuda:{i}")
            model.max_seq_length = self.cfg.max_seq_length
            if self.fp16:
                model = model.half()
            self.dim = int(model.get_sentence_embedding_dimension())
            self._models.append(model)
        self._pool = ThreadPoolExecutor(max_workers=n)
        log.info("multi-GPU embedder ready (dim=%d)", self.dim)
        return self

    @property
    def dim_per_device(self) -> int:
        return self.dim

    def token_count(self, text: str) -> int:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.cfg.model_name)
        return len(tok.encode(text, add_special_tokens=False))

    def embed(self, texts: Sequence[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        """Embed texts, sharding the batch across every GPU."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not self._models:
            self.load()

        payload = list(texts)
        if is_query and self.cfg.query_prefix:
            payload = [f"{self.cfg.query_prefix}{t}" for t in payload]

        bs = batch_size or self.cfg.batch_size
        # Split into whole batches first, then hand batches to devices
        # round-robin. Working at batch granularity keeps every device's
        # encode() call full-size, which is where the GPU is most efficient.
        batch_slices = [list(range(i, len(payload), bs)) for i in range(min(bs, len(payload)))]
        per_device_batches: list[list[list[int]]] = [[] for _ in range(self.n_devices)]
        for device_idx in range(self.n_devices):
            for chunk_idx in range(device_idx, len(batch_slices), self.n_devices):
                per_device_batches[device_idx].append(batch_slices[chunk_idx])

        def run(device_idx: int) -> tuple[int, list[list[int]], np.ndarray]:
            model = self._models[device_idx]
            blocks: list[np.ndarray] = []
            for idx_list in per_device_batches[device_idx]:
                chunk_texts = [payload[i] for i in idx_list]
                blocks.append(
                    model.encode(chunk_texts, batch_size=bs, convert_to_numpy=True,
                                 normalize_embeddings=self.cfg.normalize, show_progress_bar=False)
                )
            if not blocks:
                return device_idx, [], np.zeros((0, self.dim), dtype=np.float32)
            return device_idx, per_device_batches[device_idx], np.vstack(blocks)

        futures = [self._pool.submit(run, i) for i in range(self.n_devices)]
        results = [f.result() for f in futures]

        out = np.zeros((len(payload), self.dim), dtype=np.float32)
        for _device_idx, batches, matrix in results:
            idx = [i for batch in batches for i in batch]
            if idx:
                out[np.asarray(idx)] = matrix
        return out

    def embed_iter(self, texts: Any, *, is_query: bool = False, batch_size: int | None = None) -> Any:
        """Yield embeddings grouped by batch, mirroring DenseEmbedder.embed_iter."""
        buffer: list[str] = []
        for text in texts:
            buffer.append(text)
            if len(buffer) >= (batch_size or self.cfg.batch_size):
                yield self.embed(buffer, is_query=is_query)
                buffer = []
        if buffer:
            yield self.embed(buffer, is_query=is_query)

    def warmup(self) -> None:
        """Run a tiny batch on every device to trigger lazy CUDA init."""
        self.embed(["warmup"] * (self.cfg.batch_size // self.n_devices + 1))


def build_embedder(cfg: EmbedConfig) -> DenseEmbedder | MultiGpuEmbedder:
    """Pick the widest embedder the hardware supports."""
    # The NVML workaround lives in DenseEmbedder.load(), which the multi-GPU path
    # never reaches -- so on a host with a broken NVML the wide path died and the
    # narrow one worked, which is the worst possible way for it to fail: the same
    # machine embedding fine as one process and not as two, with no error that points
    # at the difference. Measured on worker-A, whose kernel is 580.173.02 against a
    # 580.178.04 userspace: a shard started here aborted with
    #
    #   RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL
    #   ASSERT FAILED at CUDACachingAllocator.cpp:96
    #
    # on the first GELU of the encoder, after writing zero blocks. Applying the same
    # workaround here fixes it for both paths, and is a no-op on a healthy host.
    from .embeddings import _force_math_sdpa_if_nvml_broken

    _force_math_sdpa_if_nvml_broken()
    try:
        n = MultiGpuEmbedder(cfg).available_devices()
    except Exception:  # noqa: BLE001
        n = 0
    if n >= 2:
        log.info("%d CUDA devices visible -> using MultiGpuEmbedder", n)
        return MultiGpuEmbedder(cfg)
    return DenseEmbedder(cfg)
