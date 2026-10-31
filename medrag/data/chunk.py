"""Structure-aware chunking.

Naive fixed-window chunking destroys the section boundaries that make clinical
reasoning possible ("the p-value was significant" only means something next to
its Methods). This module splits on document structure first, then packs
sections to a token budget, so every chunk is semantically homogeneous and
carries its section label into the embedding text.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from typing import Any

from ..config import ChunkConfig
from .schema import Chunk, Document, Section

log = logging.getLogger(__name__)

# Sentence splitter that avoids common clinical abbreviations, which otherwise
# cause premature splits ("e.g.", "i.v.", "vs.", "no. of patients").
_ABBREV = {
    "e.g", "i.e", "i.v", "i.m", "v.s", "v.s.", "vs", "etc", "fig", "no", "approx",
    "ca", "cf", "dr", "mr", "ms", "mg", "kg", "hr", "min", "sec", "sd", "se",
    "ci", "p", "t", "n", "al", "ie", "eg", "eq", "tab", "ref", "sp", "suppl",
    "vol", "ed", "eds", "st", "ave", "blvd", "inc", "ltd", "dept", "univ",
    "mg/kg", "a.c", "b.p", "r.r", "w.i.p", "a.m", "p.m", "b.i.d", "q.d", "q.h",
    "s.c", "p.o", "incl", "excl", "resp", "pts", "yrs", "mos", "wk", "mo",
}
# The lookahead accepts lowercase because a sentence may legitimately begin
# with an abbreviation ("... twice daily. e.g. no change in HbA1c.").
# Abbreviations that must NOT end a sentence are protected separately, below,
# by replacing their trailing whitespace with a sentinel.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Za-z0-9(\[])")


def _protect_abbreviations(text: str) -> str:
    """Insert a sentinel after periods that end abbreviations, then restore."""
    def repl(m: re.Match[str]) -> str:
        token = m.group(1)
        return f"{token}\x00 " if token.rstrip(".").lower() in _ABBREV else m.group(0)

    return re.sub(r"\b([A-Za-z][A-Za-z.]{0,4}\.)\s", repl, text)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, tolerating clinical abbreviations."""
    if not text:
        return []
    protected = _protect_abbreviations(text)
    # Protected abbreviations have no whitespace after them, so they cannot
    # match _SENT_BOUNDARY; everything else splits normally.
    parts = _SENT_BOUNDARY.split(protected)
    return [p.replace("\x00", "").strip() for p in parts if p.strip()]


def approximate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token for clinical prose).

    Used for budget arithmetic; the authoritative count is applied by the
    embedder's tokenizer when available.
    """
    return max(1, (len(text) + 3) // 4)


def _pack_sentences(
    sentences: list[str],
    target: int,
    max_tokens: int,
    overlap: int,
) -> Iterator[tuple[str, int, int]]:
    """Greedily pack sentences into windows, carrying ``overlap`` tokens forward.

    Returns (text, char_start, char_end) triples positioned against the input
    text so citations can point back at exact spans.
    """
    if not sentences:
        return
    window: list[str] = []
    window_tokens = 0
    consumed_for_overlap: list[str] = []
    overlap_tokens = 0
    offset = 0

    for sentence in sentences:
        # Track the char offset of this sentence inside the parent text.
        pos = offset
        offset += len(sentence) + 1
        stokens = approximate_tokens(sentence)

        if window_tokens + stokens > max_tokens and window:
            text = " ".join(window)
            yield text, pos - len(text), pos
            # Roll the trailing sentences forward to create overlap.
            window = list(consumed_for_overlap)
            window_tokens = overlap_tokens
            consumed_for_overlap = []
            overlap_tokens = 0

        window.append(sentence)
        window_tokens += stokens
        consumed_for_overlap.append(sentence)
        overlap_tokens += stokens
        while overlap_tokens > overlap and len(consumed_for_overlap) > 1:
            removed = consumed_for_overlap.pop(0)
            overlap_tokens -= approximate_tokens(removed)

        if window_tokens >= target:
            text = " ".join(window)
            yield text, pos - len(text), pos
            window = []
            window_tokens = 0
            consumed_for_overlap = []
            overlap_tokens = 0

    if window:
        text = " ".join(window)
        start = max(0, offset - len(text) - 1)
        yield text, start, offset


def _merge_short(chunks: list[tuple[str, int, int, str]], min_tokens: int) -> list[tuple[str, int, int, str]]:
    """Fold sub-``min_tokens`` chunks into their neighbour to avoid orphan slivers."""
    if not chunks:
        return []
    merged: list[tuple[str, int, int, str]] = []
    for text, start, end, section in chunks:
        if merged and approximate_tokens(text) < min_tokens:
            ptext, pstart, _pend, psection = merged[-1]
            if section == psection and approximate_tokens(ptext) + approximate_tokens(text) <= 1024:
                merged[-1] = (f"{ptext} {text}", pstart, end, psection)
                continue
        merged.append((text, start, end, section))
    return merged


def _build_embed_text(chunk_text: str, doc: Document, section: Section, cfg: ChunkConfig) -> str:
    """Prepend document context so a chunk is retrievable out of its parent context.

    A passage saying "median OS was 21 months" is unretrievable without the
    cancer and the regimen; prepending title/journal prefixes fixes that.
    """
    prefixes: list[str] = []
    for field_name in cfg.context_fields:
        value = getattr(doc, field_name, "") if field_name != "section" else section.title
        if field_name == "year" and not value:
            continue
        value = str(value).strip()
        if not value:
            continue
        if len(value) > 300:
            value = value[:300].rsplit(" ", 1)[0]
        prefixes.append(value)
    if not prefixes:
        return chunk_text
    # Keep the chunk body dominant; context is a prefix, not a summary.
    return f"{' | '.join(prefixes)}\n\n{chunk_text}"


def chunk_document(doc: Document, cfg: ChunkConfig) -> list[Chunk]:
    """Split one Document into retrievable Chunks under ``cfg``."""
    if cfg.require_abstract and not doc.abstract.strip() and not doc.sections:
        return []

    chunks: list[Chunk] = []
    index = 0

    def emit(text: str, section: Section, is_abstract: bool) -> None:
        nonlocal index
        text = text.strip()
        if approximate_tokens(text) < 8:
            return
        embed_text = _build_embed_text(text, doc, section, cfg)
        chunk_id = f"{doc.doc_id}#{index:04d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=text,
                chunk_index=index,
                section_title=section.title,
                section_kind=section.kind,
                token_count=approximate_tokens(text),
                title=doc.title,
                journal=doc.journal,
                year=doc.year,
                pmid=doc.pmid,
                pmcid=doc.pmcid,
                doi=doc.doi,
                publication_types=doc.publication_types,
                categories=doc.categories,
                embed_text=embed_text,
                is_abstract=is_abstract,
            )
        )
        index += 1

    if cfg.strategy == "abstract":
        if doc.abstract.strip():
            emit(doc.abstract, Section(title="Abstract", kind="background"), True)
        for section in doc.sections:
            # Same de-duplication as the default strategy: a structured abstract's
            # parts are already covered by the single abstract chunk above.
            if section.text.strip() and section.text.strip() != doc.abstract.strip():
                emit(section.text, section, False)
        return chunks

    # Default: section strategy. Abstract first (it is the most information-dense
    # unit and the highest-value retrieval target), then full-text sections.
    #
    # For PubMed records, doc.sections *are* the structured abstract's labelled
    # parts. Those must be emitted exactly once: tracking which Section objects
    # were consumed above is what prevents the full-text pass from re-emitting
    # the same text a second time as non-abstract chunks.
    consumed: set[int] = set()
    if doc.abstract.strip():
        abstract_sections = [
            s for s in doc.sections
            if s.kind in ("background", "methods", "results", "conclusions", "") and s.text.strip()
        ]
        if not abstract_sections:
            abstract_sections = [Section(title="Abstract", kind="background", text=doc.abstract)]
        for section in abstract_sections:
            consumed.add(id(section))
            for text, _s, _e in _pack_sentences(
                split_sentences(section.text), cfg.target_tokens, cfg.max_tokens, cfg.overlap_tokens
            ):
                emit(text, section, True)

    # Full-text sections (PMC), skipping anything already emitted above.
    abstract_norm = re.sub(r"\W+", " ", doc.abstract.lower()).strip()[:400]
    for section in doc.sections:
        if id(section) in consumed:
            continue
        text = section.text.strip()
        if not text or text == doc.abstract.strip():
            continue
        norm = re.sub(r"\W+", " ", text.lower()).strip()[:400]
        if norm and norm == abstract_norm:
            continue
        packed = list(
            _merge_short(
                [(t, s, e, section.title) for t, s, e in _pack_sentences(
                    split_sentences(text), cfg.target_tokens, cfg.max_tokens, cfg.overlap_tokens
                )],
                cfg.min_tokens,
            )
        )
        for chunk_text, _start, _end, _sec in packed:
            emit(chunk_text, section, False)

    return chunks


def chunk_corpus(
    docs: Iterable[Document],
    cfg: ChunkConfig,
    *,
    show_progress: bool = False,
) -> Iterator[Chunk]:
    """Stream chunks for an iterable of Documents.

    Streams rather than materialising because a multi-million-document corpus
    does not fit in memory as Chunk objects.
    """
    n_docs = 0
    n_chunks = 0
    for doc in docs:
        n_docs += 1
        for chunk in chunk_document(doc, cfg):
            n_chunks += 1
            yield chunk
        if show_progress and n_docs % 10_000 == 0:
            log.info("chunked %d docs -> %d chunks", n_docs, n_chunks)
    log.info("chunking complete: %d docs -> %d chunks", n_docs, n_chunks)
