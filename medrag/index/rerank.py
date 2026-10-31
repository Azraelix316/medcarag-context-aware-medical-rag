"""Cross-encoder reranking of first-stage candidates.

A bi-encoder embeds query and passage independently, so it cannot model their
interaction. A cross-encoder concatenates them, which is far more accurate but
quadratic in cost, so it only ever reranks a short candidate list. This is the
single largest quality gain in the hybrid pipeline and is the main ablation axis
in the paper.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .hybrid import Candidate

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Lazy-loading cross-encoder scorer over (query, passage) pairs."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        device: str = "auto",
        fp16: bool = True,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        from .embeddings import resolve_device, select_fp16

        self.model_name = model_name
        self.device = resolve_device(device)
        # Must go through select_fp16: a plain `.half()` on a pre-Volta GPU is
        # emulated in fp32 units and measured ~300x slower here (616s vs 2s for
        # 100 pairs on sm_61).
        self.fp16 = select_fp16(fp16, self.device)
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> "CrossEncoderReranker":
        if self._model is not None:
            return self
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        t0 = time.time()
        log.info("loading reranker %s on %s (fp16=%s)", self.model_name, self.device, self.fp16)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()
        if self.fp16:
            model = model.half()
        self._model = model.to(self.device)
        log.info("reranker ready in %.1fs (%s)", time.time() - t0,
                 "fp16" if self.fp16 else "fp32")
        return self

    def score(self, query: str, passages: Sequence[str], *, top_k: int | None = None) -> list[float]:
        """Score each passage against the query; higher is better."""
        if not passages:
            return []
        self.load()
        import torch

        selected = passages[: top_k] if top_k else list(passages)
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(selected), self.batch_size):
                batch = selected[start : start + self.batch_size]
                enc = self._tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**enc).logits
                if logits.shape[-1] == 1:
                    scores.extend(logits.squeeze(-1).float().cpu().tolist())
                else:
                    # Binary classification head: take the positive-class logit.
                    scores.extend(logits[:, -1].float().cpu().tolist())
        return scores

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        texts: dict[str, str] | Sequence[str],
        *,
        top_k: int = 10,
    ) -> list[Candidate]:
        """Rerank ``candidates`` and return the top ``top_k``, order preserved.

        ``texts`` may be either a mapping keyed by chunk id or a sequence
        positionally aligned with ``candidates``.
        """
        if not candidates:
            return []
        ids = [c.chunk_id for c in candidates]
        if isinstance(texts, dict):
            passages = [texts.get(cid, "") for cid in ids]
        else:
            passages = list(texts)[: len(ids)]
            if len(passages) != len(ids):
                passages = (passages + [""] * len(ids))[: len(ids)]
        scores = self.score(query, passages)
        if len(scores) != len(candidates):
            log.warning("reranker returned %d scores for %d candidates", len(scores), len(candidates))
            return list(candidates[:top_k])

        scored = []
        for cand, score in zip(candidates, scores):
            cand.rerank_score = float(score)
            # The reranker becomes the ordering signal, but the fused score is
            # retained so downstream code can still inspect first-stage rank.
            cand.score = float(score)
            scored.append(cand)
        scored.sort(key=lambda c: (-c.score, c.chunk_id))
        return scored[:top_k]

    def stats(self) -> dict[str, Any]:
        return {"model": self.model_name, "device": self.device, "fp16": self.fp16, "loaded": self._model is not None}

    def unload(self) -> None:
        """Free GPU memory; the reranker is large and only needed per request."""
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
