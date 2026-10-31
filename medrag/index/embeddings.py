"""Dense bi-encoder embeddings with batching, resume, and shard-level caching.

Embedding a multi-million chunk corpus on two consumer GPUs is the pipeline's
longest stage, so this module is built around resumability: embeddings are
written to fixed-size shards with a sidecar recording which chunk ids they
cover, and a restart skips shards that are already complete.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..config import EmbedConfig

log = logging.getLogger(__name__)

#: How many times to retry a block that keeps aborting on NVML once the batch size is
#: already at its floor, and how long to wait (multiplied by the attempt number).
_NVML_MAX_STALL_RETRIES = 6
_NVML_STALL_BACKOFF_S = 5.0


def resolve_device(requested: str = "auto") -> str:
    """Pick the best available torch device, honouring an explicit override."""
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception as exc:  # noqa: BLE001
        log.warning("device probe failed, defaulting to cpu: %s", exc)
    return "cpu"


def supports_native_fp16() -> bool:
    """True when the GPU has native fp16 throughput, i.e. compute capability >= 7.0.

    This matters: on Pascal/Volta-era consumer cards (sm_61 like the Quadro
    P2000) fp16 is emulated in fp32 units, and running the model in fp16 is
    measured ~3x SLOWER than plain fp32. Defaulting fp16 off unless the hardware
    genuinely benefits avoids a large, silent throughput regression.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major = torch.cuda.get_device_capability(0)[0]
        return major >= 7
    except Exception:  # noqa: BLE001
        return False


def gpu_compute_capabilities() -> list[tuple[int, int]]:
    try:
        import torch

        return [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        return []


def select_fp16(requested: bool, device: str) -> bool:
    """Resolve the effective fp16 setting for ``device``.

    fp16 is only honoured on CUDA devices with native fp16 support. On CPU/MPS
    and on pre-Volta CUDA it is forced off.
    """
    if not requested:
        return False
    if not device.startswith("cuda"):
        return False
    if not supports_native_fp16():
        caps = gpu_compute_capabilities()
        log.info("disabling fp16: GPU compute capability %s has no native fp16 "
                 "(fp16 is emulated and slower than fp32 here)", caps or "unknown")
        return False
    return True


def _nvml_is_broken() -> bool:
    """True when ``nvmlInit_v2`` cannot succeed on this host.

    Probed with ``nvidia-smi -L`` rather than by importing torch, so it can run before
    torch exists. The mismatch is reported on STDOUT with exit 18, not on stderr --
    checking stderr alone reports a healthy host, which is how this went undetected
    while the fix appeared not to work.
    """
    import subprocess
    import sys as _sys

    if not _sys.platform.startswith("linux"):
        return False
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=20)
    except Exception:  # noqa: BLE001
        return False
    out = (r.stdout or b"") + (r.stderr or b"")
    return r.returncode != 0 and b"NVML" in out


# Set at import, before torch is ever imported on this path.
#
# PYTORCH_NVML_BASED_CUDA_CHECK=0 is read by torch when it initialises CUDA, so
# setting it inside a function -- after `import torch` has already happened -- is too
# late and the failure still happens. Measured on worker-A, whose kernel is 580.173.02
# against a 580.178.04 userspace library:
#
#   set in the shell, before python   -> MultiGpuEmbedder runs, 141 rows/s
#   set inside the process, before use -> RuntimeError: NVML_SUCCESS ==
#                                        DriverAPI::get()->nvmlInit_v2_() INTERNAL
#                                        ASSERT FAILED at CUDACachingAllocator.cpp
#
# with the traceback landing on whichever CUDA call happened to allocate first -- GELU
# in one path, scaled_dot_product_attention in another, so the message points at
# innocent code each time.
#
# Plain CUDA compute is unaffected: a 4096x4096 matmul completes and returns a correct
# checksum on the host that cannot init NVML. The cost of the escape is that torch
# stops using NVML for CUDA health checks, which is why it is applied only when the
# probe says NVML is actually broken and not unconditionally.
if os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK") != "0" and _nvml_is_broken():
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
    logging.getLogger(__name__).warning(
        "NVML cannot initialise on this host; setting "
        "PYTORCH_NVML_BASED_CUDA_CHECK=0 before torch loads. Plain CUDA compute is "
        "unaffected, but torch will stop using NVML for CUDA health checks.")


def _force_math_sdpa_if_nvml_broken() -> None:
    """Disable torch's fused attention kernels when NVML cannot initialise.

    The kernel module here is 580.173.02 while the userspace library is 580.178.04,
    so ``nvmlInit_v2`` fails. torch's flash and memory-efficient scaled-dot-product
    attention paths query NVML for device properties, and abort with

        RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL
        ASSERT FAILED at CUDACachingAllocator.cpp:96

    partway through a forward pass -- so a 554,069-document embedding run dies after
    two hours of work with a message that names neither the cause nor the fix. Plain
    CUDA compute is unaffected: a 2048-cube matmul still runs at 1.8 TFLOP/s.

    The math backend is slower and uses more memory, so it is only selected when
    NVML is actually broken; on a healthy host nothing changes.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return
    if _nvml_is_broken():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        # The caching-allocator half of the same problem is handled at module import,
        # because torch reads PYTORCH_NVML_BASED_CUDA_CHECK when it initialises CUDA
        # and setting it here would be too late. See the note above the import-time
        # block: with only this SDPA fix, a shard on worker-A died during the first GELU
        # of the encoder with zero blocks written.
        #
        log.warning("NVML is unavailable on this host; forcing the math SDPA backend "
                    "and PYTORCH_NVML_BASED_CUDA_CHECK=0. Attention will be slower and "
                    "use more memory, but fused kernels and the caching allocator both "
                    "abort on a driver/library version mismatch. Plain CUDA compute is "
                    "unaffected.")


class DenseEmbedder:
    """Wraps a sentence-transformers bi-encoder with batching and fp16."""

    def __init__(self, cfg: EmbedConfig | None = None) -> None:
        self.cfg = cfg or EmbedConfig()
        self.device = resolve_device(self.cfg.device)
        self._model: Any = None
        self._tokenizer: Any = None
        self.dim = self.cfg.dim
        # fp16 is emulated on pre-Volta CUDA and measured ~3x slower than fp32
        # there, so it is resolved against the actual compute capability.
        self.fp16 = select_fp16(self.cfg.fp16, self.device)

    # ------------------------------------------------------------------ model
    def load(self) -> "DenseEmbedder":
        if self._model is not None:
            return self
        from sentence_transformers import SentenceTransformer

        _force_math_sdpa_if_nvml_broken()
        t0 = time.time()
        log.info("loading embedder %s on %s", self.cfg.model_name, self.device)
        model = SentenceTransformer(self.cfg.model_name, device=self.device)
        model.max_seq_length = self.cfg.max_seq_length
        if self.fp16:
            try:
                model = model.half()
            except Exception as exc:  # noqa: BLE001
                log.warning("fp16 not applied: %s", exc)
                self.fp16 = False
        log.info("embedder precision: %s", "fp16" if self.fp16 else "fp32")
        # Trust the model's actual hidden size over the config default.
        self.dim = int(model.get_sentence_embedding_dimension())
        if self.dim != self.cfg.dim:
            log.info("embed dim %d (config said %d); using model value", self.dim, self.cfg.dim)
        self._model = model
        self._tokenizer = model.tokenizer
        log.info("embedder ready in %.1fs (dim=%d)", time.time() - t0, self.dim)
        return self

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    # --------------------------------------------------------------- embedding
    def token_count(self, text: str) -> int:
        """Authoritative token count via the model's tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def embed(self, texts: Sequence[str], *, is_query: bool = False,
              batch_size: int | None = None, block: int | None = None) -> np.ndarray:
        """Embed a batch of texts into an (n, dim) float32 array.

        Queries get the BGE instruction prefix; passages do not, which is what
        the model was trained for.

        NVML here is broken at the driver level -- kernel 580.173.02 against userspace
        580.178.04 -- and torch's flash and memory-efficient attention abort on it part
        way through a forward pass. Both a smaller batch and a retry of the whole block
        are needed, and the reason for the second is worth recording: the same
        1,506-character page texts abort at batch 64 in one process and embed in 54s in
        the next. It is not a size threshold, so shrinking to the floor does not fix
        it and the job dies instead of waiting five seconds.

        The batch size backs off on failure rather than being fixed. On this host the
        CUDA caching allocator queries NVML once a batch allocation crosses a size
        threshold, and NVML cannot initialise (kernel 580.173.02 against userspace
        580.178.04). Measured: batch 32 and 128 embed 4,000 rows fine, batch 256
        aborts with

            RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL
            ASSERT FAILED at CUDACachingAllocator.cpp

        Forcing the math SDPA backend in :func:`_force_math_sdpa_if_nvml_broken` is
        not sufficient on its own -- the abort is in the allocator, not the attention
        kernel. Halving and retrying finds a working size without hardcoding one, and
        the ceiling it settles on is recorded in the log rather than discovered again
        two hours into a job. On this corpus that ceiling turned out to be no ceiling
        at all: 64 -> 32 -> 16 -> 8 all aborted, so ``block`` below is what actually
        fixed it.

        Blocks are deliberately *not* separated by ``torch.cuda.empty_cache()``. That
        was in the first version of this and it is what kept killing the job: the
        flush returns segments to the driver, so the next block's first allocation
        has to cudaMalloc fresh, and a fresh segment allocation goes through the
        allocator's NVML-querying path -- the one thing that is broken on this host.
        Dropping the reference lets the allocator reuse its segments instead. Twelve
        blocks of 20,000 rows then run clean in 287s; with the flush the same job
        aborted inside the third.

        ``block`` is the other real fix. Sentence-transformers tokenises the entire input
        list before running any batch, so a single ``encode`` over 554,069 texts at
        ``max_seq_length=512`` materialises 554,069 token sequences before a single
        embedding exists -- which is what exhausted the allocator and tripped the
        NVML-dependent path. Inputs above ``block`` rows are embedded in blocks and
        concatenated. Measured identical: the same 60,000 rows embed in 132s either
        way, and the outputs agree.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payload = list(texts)
        if is_query and self.cfg.query_prefix:
            payload = [f"{self.cfg.query_prefix}{t}" for t in payload]

        bs = batch_size or self.cfg.batch_size
        block = block or self.cfg.block_size
        if block and len(payload) > block:
            parts: list[np.ndarray] = []
            for start in range(0, len(payload), block):
                parts.append(self.embed(payload[start:start + block], is_query=False,
                                        batch_size=bs, block=0))
            return np.concatenate(parts, axis=0) if parts else np.zeros(
                (0, self.dim), dtype=np.float32)

        floor = 8
        stalled = 0
        while True:
            try:
                vectors = self.model.encode(
                    payload,
                    batch_size=bs,
                    convert_to_numpy=True,
                    normalize_embeddings=self.cfg.normalize,
                    show_progress_bar=False,
                )
                return np.asarray(vectors, dtype=np.float32)
            except RuntimeError as exc:
                if "NVML" not in str(exc):
                    raise
                if bs > floor:
                    bs = max(floor, bs // 2)
                    log.warning("embedding batch aborted on NVML; retrying at "
                                "batch_size=%d", bs)
                    continue
                # Already at the floor and still aborting. Shrinking further is not
                # the lever: the same 1,506-character texts that abort in one process
                # embed fine in another at the same batch size, so this is not a size
                # threshold. What has gone wrong is the allocator's state, so drop the
                # cached segments, pause, and try the block again.
                stalled += 1
                if stalled > _NVML_MAX_STALL_RETRIES:
                    raise
                time.sleep(_NVML_STALL_BACKOFF_S * stalled)
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                log.warning("embedding aborted on NVML at batch_size=%d with the "
                            "batch already at the floor; cleared the cache and "
                            "retrying (%d/%d)", bs, stalled, _NVML_MAX_STALL_RETRIES)

    def embed_iter(
        self,
        texts: Iterable[str],
        *,
        is_query: bool = False,
        batch_size: int | None = None,
        flush_every: int = 2048,
    ) -> Iterator[np.ndarray]:
        """Stream embeddings in small groups so callers can write incrementally."""
        buffer: list[str] = []
        n = 0
        for text in texts:
            buffer.append(text)
            if len(buffer) >= (batch_size or self.cfg.batch_size):
                yield self.embed(buffer, is_query=is_query)
                n += len(buffer)
                buffer = []
                if n % flush_every < len(buffer or [1]):
                    log.info("embedded %d texts", n)
        if buffer:
            yield self.embed(buffer, is_query=is_query)


def save_embeddings(path: str | Path, matrix: np.ndarray, ids: Sequence[str], rows: Sequence[int] | None = None) -> Path:
    """Persist embeddings plus their chunk ids and row ids as a single .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "embeddings": matrix.astype(np.float32),
        "ids": np.asarray(ids, dtype=object),
    }
    if rows is not None:
        payload["rows"] = np.asarray(rows, dtype=np.int64)
    np.savez(path, **payload)
    log.info("saved embeddings %s shape=%s (%.1f MiB)", path.name, matrix.shape, path.stat().st_size / 2**20)
    return path


def load_embeddings(path: str | Path) -> tuple[np.ndarray, list[str], list[int]]:
    data = np.load(path, allow_pickle=True)
    ids = [str(x) for x in data["ids"]]
    if "rows" in data:
        rows = [int(x) for x in data["rows"]]
    else:
        # Legacy file: positions were row ids, so the default is correct.
        rows = list(range(len(ids)))
    return data["embeddings"], ids, rows
