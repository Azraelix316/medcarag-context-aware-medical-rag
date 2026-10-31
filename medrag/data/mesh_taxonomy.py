"""MeSH taxonomy resolution for clinical specialty facets.

PubMed's XML gives each article its MeSH *descriptors* and their UIs, but not
their tree positions. Those must be looked up separately, and guessing them from
the UI string is wrong: every descriptor UI starts with ``D`` regardless of
whether it is a disease (``D003924`` Diabetes Mellitus, Type 2, tree **C19**) or
a drug (``D008968`` Metformin, tree **D02**). Deriving categories that way
silently mislabels the entire corpus.

This module resolves ``UI -> tree root -> coarse specialty`` against the NLM
MeSH API (``id.nlm.nih.gov``), with three layers, in order of preference:

1. a on-disk cache, so a lookup is paid once and never again;
2. the NLM REST API, rate-limited and cached as it goes;
3. a curated fallback that infers the category from the descriptor label for
   common high-frequency headings, so the pipeline still works offline.

The result is used for two things: stratified evaluation slices, and rejecting
too-generic MeSH terms when constructing benchmark queries.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

API_TEMPLATE = "https://id.nlm.nih.gov/mesh/{ui}.json"

# MeSH's own top-level tree categories -> names used as evaluation facets.
# Labels follow the MeSH thesaurus top level, not an ad-hoc grouping:
#   E = Analytical, Diagnostic and Therapeutic Techniques
#   G = Phenomena and Processes  (i.e. biological/physiological)
#   I = Information Technology;  M = Persons (M01 = Population)
TREE_CATEGORIES: dict[str, str] = {
    "A": "anatomy",
    "B": "organisms",
    "C": "diseases",
    "D": "drugs-and-therapeutics",
    "E": "diagnostic-and-therapeutic-techniques",
    "F": "psychiatry-and-psychology",
    "G": "phenomena-and-processes",
    "H": "disciplinary-activities",
    "I": "information-technology",
    "J": "technology-industry-agriculture",
    "K": "humanities",
    "L": "language-and-communication",
    "M": "persons-population",
    "N": "health-care-services",
    "O": "miscellaneous-topics",
    "P": "biological-sciences",
    "Q": "science",
    "R": "arts",
    "S": "social-sciences",
    "T": "technology",
    "U": "education",
    "V": "publication-types",
    "W": "organizations",
    "X": "medical-history",
    "Y": "humanities",
    "Z": "geographic-locations",
}

# Roots too generic to form a discriminative benchmark query.
#   M (Adult, Aged, Humans, Adolescent) and I/H/T/J (Information Technology,
#   Disciplines, Technology) cover most of PubMed and yield near-useless queries.
GENERIC_ROOTS = frozenset({
    "I", "H", "J", "T", "W", "Y", "V", "L", "K", "M", "N", "O", "Q", "E", "Z", "X",
})

# Roots that reliably indicate real clinical content for a benchmark query.
CLINICAL_ROOTS = frozenset({"A", "C", "D", "F", "G", "P"})

# Fallback keyword groups used when the API is unreachable or a descriptor has
# no UI. Ordered MOST SPECIFIC FIRST, because first match wins: a broad
# "disorder" stem would otherwise swallow every psychiatric diagnosis
# ("Depressive Disorder" is MeSH tree F01, not C). Each entry is a list of
# stems plus the MeSH top-level root it implies, so the fallback returns the
# same kind of value as the API path and the two are interchangeable.
_LABEL_GROUPS: tuple[tuple[tuple[str, ...], str], ...] = (
    # Specific disease families first.
    (("neoplasm", "carcinoma", "tumour", "tumor", "cancer", "leukemia", "leukaemia",
      "lymphoma", "sarcoma", "melanoma", "adenoma", "metasta"), "C"),
    # Psychiatry before the generic disease stems.
    (("depress", "anxiety", "schizophren", "bipolar", "psychiatr", "psycholog",
      "mental illness", "substance use", "cannabis", "tobacco",
      "sleep", "wake", "dream", "nightmare"), "F"),
    # Drugs and therapeutic agents.
    (("drug", "pharmacol", "chemotherap", "immunotherap", "antibi", "anesthe",
      "anaesthe", "analgesi", "vaccin", "supplement", "vitamin", "inhibitor",
      "antagonist", "agonist", "steroid", "narcotic", "opioid", "antineoplastic",
      "antiviral", "anticonvulsant", "antihypertensive", "antimicrobial",
      "anti-inflammatory", "cardiovascular agent", "therapeutic use"), "D"),
    (("mice", "rat", "animal", "in vitro", "cell line", "model organism"), "B"),
    (("genetic", "gene", "genom", "mutation", "polymorphism", "protein",
      "receptor", "kinase", "apoptosis", "immune", "cytokine", "enzyme"), "P"),
    # Generic disease stems last among the disease-ish groups.
    (("diabetes", "hypertension", "asthma", "arthritis", "infection", "sepsis",
      "stroke", "obesity", "anaemia", "anemia", "epilepsy", "heart failure",
      "renal", "kidney", "hepatitis", "tuberculosis", "pneumonia", "syndrome",
      "disease", "disorder", "failure", "ischemi", "infarct", "inflammat", "deficien",
      "injury", "pain", "fever", "insufficien", "dysfunction", "abnormalit"), "C"),
    (("surgery", "operative", "postoperat", "preoperat", "transplant",
      "orthopa", "reconstruct", "resection", "diagnos", "screen", "therapy"), "E"),
    (("nurs", "midwi", "physician", "patient care", "hospital", "clinic",
      "emergency", "primary care", "general practice", "health service",
      "quality", "adherence", "compliance", "access", "equity", "disparit",
      "cost", "economic"), "N"),
    (("diet", "nutrition", "exercise", "fitness", "feeding", "body weight",
      "physical activity", "mortality", "survival", "outcome", "epidemiolog",
      "incidence", "prevalence", "prognos", "cohort", "risk factor"), "G"),
    (("education", "teaching", "curriculum", "competency", "training"), "H"),
)


def _build_label_matchers() -> tuple[tuple[re.Pattern[str], str], ...]:
    """Compile the fallback groups into inflection-tolerant regexes.

    MeSH descriptors rarely use a bare stem: it is "Neoplasms", "Depressive
    Disorder", "Therapeutic Uses", "Adherence". Each stem therefore allows up to
    four trailing letters. The trailing ``\\b`` keeps this from over-matching --
    "drugstore" does not satisfy ``drug[a-z]{0,4}\\b`` -- while still covering
    plural, verbal, and adjectival forms.
    """
    compiled: list[tuple[re.Pattern[str], str]] = []
    for stems, root in _LABEL_GROUPS:
        alts = [re.escape(stem) + r"[a-z]{0,4}" if " " not in stem else re.escape(stem) + r"s?"
                for stem in stems]
        compiled.append((re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.I), root))
    return tuple(compiled)


_LABEL_MATCHERS = _build_label_matchers()

# Drug-name stems for single-word labels no keyword group can catch. INN stems
# are distinctive enough to classify without an API call, which matters because
# the most frequent MeSH drug descriptors are exactly these bare names
# ("Metformin", "Atorvastatin", "Lisinopril").
_DRUG_SUFFIXES: tuple[str, ...] = (
    "formin", "mycin", "cillin", "prin", "pril", "sartan", "statin", "azole",
    "caine", "olol", "profen", "prazole", "tidine", "zine", "dine", "mustine",
    "platin", "taxel", "rubicin", "citabine", "silib", "vir", "ximab", "mab",
    "numab", "cept", "oxacin", "cycline", "conazole", "tropium", "onium",
)
_DRUG_SUFFIX_RE = re.compile(r"\b[a-z]{0,14}(?:" + "|".join(_DRUG_SUFFIXES) + r")\b", re.I)



class MeshTaxonomy:
    """Cache-backed ``UI -> tree root`` resolver with a label-based fallback."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        *,
        use_api: bool = True,
        rate_limit: float = 30.0,
        timeout: int = 20,
        workers: int = 12,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        self.use_api = use_api
        # id.nlm.nih.gov is NLM's linked-data service, not E-utilities, and
        # carries no documented 3 req/s cap. Measured throughput saturates near
        # 15 req/s regardless of client concurrency, so 12 workers / 30 req/s
        # sits just past the knee without hammering the service.
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.workers = workers
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self.n_cache_hits = 0
        self.n_api_calls = 0
        self.n_api_failures = 0
        self._session: Any = None
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    # ------------------------------------------------------------------- cache
    def _load_cache(self) -> None:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._cache = {str(k): str(v) for k, v in data.items()}
                log.info("loaded MeSH taxonomy cache: %d descriptors", len(self._cache))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read MeSH taxonomy cache %s: %s", self.cache_path, exc)

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache, sort_keys=True), encoding="utf-8")
            tmp.replace(self.cache_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not write MeSH taxonomy cache: %s", exc)

    def flush(self) -> None:
        self._save_cache()

    # -------------------------------------------------------------------- API
    def _throttle(self) -> None:
        """Fixed-interval limiter shared across threads.

        The sleep happens outside the lock so worker threads overlap their
        network waits; the aggregate request rate stays bounded by
        ``rate_limit``.
        """
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            slot = now + wait
            self._next_allowed = slot + (1.0 / self.rate_limit if self.rate_limit else 0.0)
        if wait > 0:
            time.sleep(wait)

    def _fetch_tree_root(self, ui: str) -> str | None:
        import requests

        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"Accept": "application/json", "User-Agent": "medrag/0.1"})
        self._throttle()
        self.n_api_calls += 1
        try:
            resp = self._session.get(API_TEMPLATE.format(ui=ui), timeout=self.timeout)
            if resp.status_code != 200:
                self.n_api_failures += 1
                return None
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.n_api_failures += 1
            log.debug("MeSH lookup failed for %s: %s", ui, exc)
            return None
        root = self._extract_root(payload)
        if root:
            with self._lock:
                self._cache[ui] = root
        return root

    @staticmethod
    def _extract_root(payload: dict[str, Any]) -> str | None:
        """Pull the first MeSH tree number's letter out of the JSON-LD payload."""
        for key, value in payload.items():
            if key.endswith("treeNumber"):
                values = value if isinstance(value, list) else [value]
                for entry in values:
                    text = str(entry)
                    match = re.search(r"mesh/([A-Z])\d", text) or re.match(r"([A-Z])\d", text)
                    if match:
                        return match.group(1)
        return None

    # --------------------------------------------------------------- resolving
    def tree_root(self, ui: str, label: str = "") -> str:
        """Return the MeSH tree root letter for ``ui`` (or infer it from ``label``)."""
        if ui:
            root = self._cache.get(ui)
            if root:
                self.n_cache_hits += 1
                return root
            if self.use_api:
                root = self._fetch_tree_root(ui)
                if root:
                    return root
        return self.infer_root(label)

    @staticmethod
    def infer_root(label: str) -> str:
        """Best-effort MeSH tree root from the descriptor label, for offline use.

        Returns "" when nothing matches: an unknown label must yield no category
        rather than a wrong one, since a wrong specialty silently corrupts the
        stratified evaluation slices.
        """
        for matcher, root in _LABEL_MATCHERS:
            if matcher.search(label or ""):
                return root
        # Bare drug names ("Metformin") match only by INN stem.
        if _DRUG_SUFFIX_RE.search(label or ""):
            return "D"
        return ""

    def category(self, ui: str, label: str = "") -> str:
        """Coarse specialty facet for a descriptor."""
        root = self.tree_root(ui, label)
        return TREE_CATEGORIES.get(root, "")

    def resolve_many(
        self,
        descriptors: list[tuple[str, str]],
        *,
        save_every: int = 500,
        workers: int | None = None,
    ) -> dict[str, str]:
        """Resolve many ``(ui, label)`` pairs, persisting the cache periodically.

        Network latency dominates here: a single serial pass resolves roughly
        4 descriptors/second because each lookup waits ~250ms. Lookups are
        therefore issued from a bounded thread pool, with ``rate_limit`` still
        capping the *aggregate* request rate so the service is not hammered.

        Descriptors already in the cache are skipped without any request, which
        makes a resumed run essentially instant.
        """
        todo = [(ui, label) for ui, label in descriptors if ui and ui not in self._cache]
        cached = len(descriptors) - len(todo)
        log.info("mesh taxonomy: %d descriptors (%d cached, %d to fetch, %d workers)",
                 len(descriptors), cached, len(todo), workers)

        out: dict[str, str] = {}
        unresolved = 0
        if not todo:
            self.flush()
            return dict(self._cache)

        from concurrent.futures import ThreadPoolExecutor

        workers = workers or self.workers
        done = 0
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(self._resolve_one, ui, label) for ui, label in todo]
            for ui, future in zip([p[0] for p in todo], futures):
                root = future.result()
                out[ui] = root
                if not root:
                    unresolved += 1
                done += 1
                if save_every and done % save_every == 0:
                    self.flush()
                    rate = done / max(1e-9, time.monotonic() - t0)
                    log.info("mesh taxonomy: %d/%d fetched (%d unresolved, %.1f/s, ETA %.1f min)",
                             done, len(todo), unresolved, rate,
                             max(0.0, len(todo) - done) / max(1e-9, rate) / 60.0)

        self.flush()
        log.info("mesh taxonomy complete: %d descriptors, %d cache hits, %d api calls, %d failures",
                 len(descriptors), self.n_cache_hits, self.n_api_calls, self.n_api_failures)
        return out

    def _resolve_one(self, ui: str, label: str) -> str:
        """Resolve a single descriptor, falling back to label heuristics."""
        root = self._fetch_tree_root(ui) if self.use_api else None
        return root or self.infer_root(label)

    def stats(self) -> dict[str, Any]:
        return {
            "cached": len(self._cache),
            "cache_hits": self.n_cache_hits,
            "api_calls": self.n_api_calls,
            "api_failures": self.n_api_failures,
        }


def taxonomy_cache_path(data_root: str | Path) -> Path:
    """Conventional location for the taxonomy cache, inside the data root."""
    return Path(data_root) / "cache" / "mesh_taxonomy.json"


def load_taxonomy(cfg: Any) -> MeshTaxonomy:
    """Build a taxonomy resolver from a :class:`PipelineConfig`."""
    return MeshTaxonomy(
        taxonomy_cache_path(cfg.paths.data_root),
        use_api=True,
        rate_limit=8.0,
    )
