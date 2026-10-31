"""Answer-readiness: is the retrieved set actually sufficient to answer safely?

Why this exists
---------------
``context_provision.py`` counts *section roles*: a retrieved set is qualified if it
holds a gold document's evidential chunk plus a Conclusions or Discussion chunk
from the same document. That is a structural test, and it inherits two problems:

1. **Definitional circularity.** The mechanisms supply framing passages and
   ``qar`` counts framing passages, so showing that a mechanism which supplies
   qualifying text raises the rate of qualifying text proves nothing about readers.
2. **Section role is a proxy, not the thing itself.** A Conclusions chunk that
   adds no caveat still counts. So does a Discussion section that hedges nothing.

This module measures the thing those proxies stand in for: whether the retrieved
text *contains the information a clinician needs in order to state a qualified
answer*. It looks at the words, not the section label, so a passage qualifies for
containing a rate whether it came from Results or Conclusions, and fails to qualify
if it contains no rate at all.

What it checks
--------------
Three elements that the motivating paper measures as missing from deployed systems
(arXiv 2502.14898: complication rates mentioned in 4-5% of responses, benefits in
6-10%):

``rate``        a quantified estimate: percentage, ratio with CI, count over n.
``population``  who the estimate applies to: sample size, demographics, comparator.
``caveat``      what limits the estimate: explicit limitation language, hedging,
                insufficient-power statements.

A retrieved set is **ready** when it holds a gold finding plus at least a rate and
a population. It is **bare** when it holds a gold finding and neither. This
mirrors ``qar``/``bar`` deliberately so the two can be compared directly — and the
comparison between them is itself informative: where they disagree, section role
and actual content disagree, which is the failure mode a structural metric hides.

Honest limits
-------------
Pattern matching is not comprehension. A passage can contain a percentage that is
not a rate of the event asked about, and a "limitation" sentence can be boilerplate.
So this measures *presence of the ingredients*, which is strictly weaker than
judging whether a reader would use them correctly, and strictly stronger than
counting section labels. It is the cheapest measurement that is not circular with
the mechanism under test; the LLM- or human-judged read test is still the real
test, and this does not substitute for it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# --- rate: a quantified estimate of the thing asked about
_RATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "12.3%", "45 %"
    re.compile(r"\d+(?:\.\d+)?\s?%"),
    # "95% CI 1.2-3.4", "95% confidence interval"
    re.compile(r"95\s?%\s?(?:ci|confidence interval)", re.I),
    # effect measures with an estimate
    re.compile(r"\b(?:hazard ratio|hr|odds ratio|orr|risk ratio|rr|relative risk|"
               r"relative hazard)\b[^.;]{0,40}?\d", re.I),
    # "risk was 2.3 (1.1 to 4.7)"
    re.compile(r"\d+(?:\.\d+)?\s*\(\s*\d+(?:\.\d+)?\s*(?:to|-|–)\s*\d+(?:\.\d+)?\s*\)"),
    # absolute counts: "12 of 240 patients"
    re.compile(r"\b\d+\s+of\s+[\d,]+\b"),
    # number needed to treat / harm / benefit. The forms are 3 or 4 letters
    # (NNT, NNTB, NNTH, NNH, NNI), so an `nn[tih]` alternation followed by \b
    # silently misses every 4-letter form.
    re.compile(r"\b(?:nnt|nntb|nnth|nnh|nni)\b[^.;]{0,30}?\d", re.I),
    # p-values, which at least mark a tested comparison
    re.compile(r"\bp\s*[=<>]\s*0?\.\d+", re.I),
)

# --- population: who the estimate applies to
_POPULATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Bare count followed by the population noun: "1600 randomised patients",
    # "800 participants". Without this, a sample size stated in prose -- the most
    # common way papers report it -- is not recognised, and an answer that states a
    # rate and the sample size scores as having no population.
    re.compile(r"\b[\d,]+\s+(?:randomi[sz]ed\s+|enrolled\s+|recruited\s+)?"
               r"(?:patients|participants|subjects|individuals|women|men|"
               r"males|females|adults|children|adolescents|infants)\b", re.I),
    re.compile(r"\b(?:n\s?=\s?[\d,]+|total of\s+[\d,]+\s+(?:patients|participants|"
               r"subjects|women|men|children))\b", re.I),
    re.compile(r"\b(?:patients|participants|subjects|individuals|cases)\s+"
               r"(?:with|who|aged|older|younger)\b", re.I),
    re.compile(r"\b(?:mean|median)\s+age\b", re.I),
    re.compile(r"\b(?:in|among|across)\s+(?:patients|participants|subjects|"
               r"individuals)\s+with\b", re.I),
    re.compile(r"\b(?:men|women|males|females|adults|children|adolescents|"
               r"elderly)\b[^.;]{0,40}\b(?:aged|with|included|enrolled)\b", re.I),
    re.compile(r"\b(?:inclusion|exclusion)\s+criteria\b", re.I),
    re.compile(r"\b(?:randomi[sz]ed|assigned)\s+(?:to|1:1|2:1)", re.I),
    # comparator
    re.compile(r"\b(?:versus|vs\.?|compared with|relative to|placebo|control group)\b", re.I),
)

# --- caveat: what limits the estimate
_CAVEAT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:limitation|limitations|limited by|we did not|could not|"
               r"unable to)\b", re.I),
    re.compile(r"\b(?:not powered|underpowered|insufficient (?:power|sample|evidence))\b", re.I),
    re.compile(r"\b(?:should be interpreted|interpreted with caution|caution is "
               r"warranted|warrant caution)\b", re.I),
    re.compile(r"\b(?:small sample|sample size was|few (?:patients|participants))\b", re.I),
    re.compile(r"\b(?:further (?:studies|research|investigation|trials?) (?:are|is)"
               r"?\s*(?:needed|required|warranted))\b", re.I),
    re.compile(r"\b(?:did not (?:reach|differ|exclude|rule out))\b", re.I),
    re.compile(r"\b(?:no (?:statistically )?significant difference|"
               r"not statistically significant)\b", re.I),
    re.compile(r"\b(?:benefit|harms?|risk|adverse)\b[^.;]{0,60}\b"
               r"(?:must|should) be (?:weighed|interpreted|considered)", re.I),
    re.compile(r"\b(?:generalizab\w+|extrapolat\w+)\b", re.I),
)


def _matches(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(p.search(text) for p in patterns)


def find_rate(text: str) -> bool:
    """Whether ``text`` states a quantified estimate."""
    return _matches(text, _RATE_PATTERNS)


def find_population(text: str) -> bool:
    """Whether ``text`` says who an estimate applies to."""
    return _matches(text, _POPULATION_PATTERNS)


def find_caveat(text: str) -> bool:
    """Whether ``text`` states what limits an estimate."""
    return _matches(text, _CAVEAT_PATTERNS)


# --- benefit: what the intervention is good for
#
# The motivating paper (arXiv 2502.14898) reports THREE failure modes, not two:
# complication rates in 4-5% of responses, **procedure benefits in 6-10%**, and
# disputed-diagnosis caveats dropping from 100% to 56-69% on presupposing queries.
# Rates and caveats were both detected; benefits had no detector anywhere in this
# project, so one third of the taxonomy the evaluation is supposed to use was simply
# absent, and nothing reported that.
#
# A benefit is not just the word "benefit". It is a statement that something got
# better, which needs an object -- "reduced" alone is not a finding, and "lower blood
# pressure" is a benefit only if something was trying to lower it. So each pattern
# pairs a direction or an outcome term with what it acts on.
#
# Deliberately NOT counted: "risks and benefits must be weighed" and its relatives. That
# is the language of *not* taking a position, and it already matches a caveat pattern;
# counting it as a benefit would let a passage hedge completely and still register as
# having stated what the procedure is good for.
_BENEFIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Explicit efficacy, with the benefit word carrying the claim.
    re.compile(r"\b(?:show(?:ed|s|n|ed\s+that)|demonstrat(?:ed|es)|confirm(?:ed|s)|"
               r"establish(?:ed|es)|prov(?:ed|es))\b"
               r"[^.;]{0,60}\b(?:benefit|efficac\w+|effective\w*)\b", re.I),
    re.compile(r"\b(?:benefit|beneficial|efficacious)\b[^.;]{0,60}"
               r"\b(?:versus|vs\.?|compared (?:with|to)|placebo|control)\b", re.I),
    # Directional improvement: a verb of reduction or improvement, and its object.
    re.compile(r"\b(?:reduce[sd]?|reduc(?:ing|tion)|decreas\w+|lower(?:s|ed|ing)?|"
               r"prevent(?:s|ed|ion)?|improv\w+|enhanc\w+|shorten\w+|"
               r"declin\w+|diminish\w+)\b[^.;]{0,70}"
               r"\b(?:risk|mortality|death\w*|complication\w*|event\w*|"
               r"symptom\w*|pain|relapse\w*|readmission\w*|rehospitali[sz]ation\w*|"
               r"hospitali[sz]ation\w*|bleeding\w*|infection\w*|recurrence\w*|"
               r"blood pressure|systolic|ldl|hdl|cholesterol|hba1c|a1c|"
               r"pain score)\b",
               re.I),
    # Comparative outcomes.
    re.compile(r"\b(?:superior|favou?rable|outperform\w*|better than|"
               r"no worse than)\b", re.I),
    # Absolute counts of benefit.
    re.compile(r"\b(?:number needed to (?:treat|prevent)|NNT)\b", re.I),
    # Fewer events than the comparator. The outcome word comes BEFORE "than" in the
    # natural phrasing -- "fewer major bleeding events than with dual antiplatelet
    # therapy" -- so the pattern anchors on the outcome and requires "than" after it.
    re.compile(r"\b(?:fewer|less)\b[^.;]{0,40}\b(?:event\w*|complication\w*|"
               r"death\w*|hospitali[sz]ation\w*|relapse\w*|recurrence\w*|"
               r"readmission\w*)\b[^.;]{0,30}\bthan\b", re.I),
)

#: Negation that turns a benefit phrase into its opposite. "We did not observe a
#: reduction in readmission" contains every token the directional pattern needs and
#: states the reverse of the finding, which is exactly the sentence a bare-claim
#: detector must not count.
_BENEFIT_NEGATION = re.compile(
    r"\b(?:not|no|never|without|failed to|neither|nor|lack(?:ing|ed)? of|"
    r"absent|unclear|uncertain|did not|was not|were not)\b",
    re.I,
)


def find_benefit(text: str) -> bool:
    """Whether ``text`` states what the intervention is good for.

    The benefit side of the motivating paper's taxonomy, alongside the complication
    rate (4-5% of responses there) and the disputed-diagnosis caveat. Paired with the
    others it is what makes a decontextualised answer detectable in the other
    direction: a system that reports "reduced mortality" without saying against what is
    the failure this whole project is about, and with no benefit signal there is no way
    to notice one.

    Negated findings are excluded. A negated phrase is not a smaller benefit signal, it
    is the absence of one, and counting it would make the metric improve on exactly the
    passages that report no effect.
    """
    for pattern in _BENEFIT_PATTERNS:
        for m in pattern.finditer(text):
            # Negation attaches to the clause, not the whole paragraph: a caveat
            # earlier in the text must not veto an unqualified result later in it.
            start = max(text.rfind(";", 0, m.start()), text.rfind(".", 0, m.start())) + 1
            if _BENEFIT_NEGATION.search(text[start:m.start()]):
                continue
            return True
    return False


@dataclass
class RetrievedPassage:
    """Minimal view of a retrieved chunk for readiness scoring."""

    row_id: int
    doc_id: str
    text: str
    section_kind: str = ""


@dataclass
class Readiness:
    """Per-query verdict on whether the retrieved set can support a safe answer."""

    qid: str = ""
    found_gold: bool = False
    has_finding: bool = False
    has_rate: bool = False
    has_population: bool = False
    has_caveat: bool = False
    #: The benefit side of the motivating paper's taxonomy: 6-10% of responses there.
    has_benefit: bool = False
    #: Finding and a rate. A bare number with nothing to anchor it.
    quantified: bool = False
    #: Finding, a rate, and a population -- enough to state "x% of whom".
    ready: bool = False
    evidence: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "found_gold": self.found_gold,
            "has_finding": self.has_finding,
            "has_rate": self.has_rate,
            "has_population": self.has_population,
            "has_caveat": self.has_caveat,
            "has_benefit": self.has_benefit,
            "quantified": self.quantified,
            "ready": self.ready,
        }


def assess_readiness(
    passages: Iterable[RetrievedPassage],
    gold_doc_ids: Iterable[str],
    *,
    qid: str = "",
) -> Readiness:
    """Classify a retrieved set by what its text actually contains.

    Only gold-document passages count, mirroring the context-provision metric's
    document-locality rule: a passage from an unrelated study cannot make this
    study's result interpretable.
    """
    gold = set(gold_doc_ids)
    out = Readiness(qid=qid)
    for p in passages:
        if p.doc_id not in gold:
            continue
        out.found_gold = True
        out.has_finding = True
        if find_rate(p.text):
            out.has_rate = True
        if find_population(p.text):
            out.has_population = True
        if find_caveat(p.text):
            out.has_caveat = True
        if find_benefit(p.text):
            out.has_benefit = True
        if out.has_rate and out.has_population and out.has_caveat and out.has_benefit:
            break
    out.quantified = out.has_finding and out.has_rate
    out.ready = out.has_finding and out.has_rate and out.has_population
    return out


@dataclass
class ReadinessReport:
    n_queries: int = 0
    n_found: int = 0
    n_ready: int = 0
    n_quantified: int = 0
    n_with_caveat: int = 0
    n_with_population: int = 0
    n_with_benefit: int = 0
    per_query: list[Readiness] = field(default_factory=list)

    @property
    def readiness(self) -> float:
        """Share of all queries whose retrieved set can support a qualified answer."""
        return self.n_ready / self.n_queries if self.n_queries else 0.0

    @property
    def quantification_rate(self) -> float:
        """Share of all queries whose set contains a rate at all."""
        return self.n_quantified / self.n_queries if self.n_queries else 0.0

    @property
    def caveat_rate(self) -> float:
        return self.n_with_caveat / self.n_queries if self.n_queries else 0.0

    @property
    def population_rate(self) -> float:
        return self.n_with_population / self.n_queries if self.n_queries else 0.0

    @property
    def benefit_rate(self) -> float:
        """Share of queries whose set states what the intervention is good for.

        The third ingredient of the motivating paper's taxonomy, which measured
        benefits in 6-10% of responses alongside complication rates in 4-5%. It had no
        detector here at all, so the taxonomy was two-thirds implemented and nothing
        said so.
        """
        return self.n_with_benefit / self.n_queries if self.n_queries else 0.0

    @property
    def bare_rate(self) -> float:
        """Found a gold finding but could not state it with a rate."""
        if not self.n_queries:
            return 0.0
        return (self.n_found - self.n_quantified) / self.n_queries

    def summary(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "n_found_gold": self.n_found,
            "n_ready": self.n_ready,
            "n_quantified": self.n_quantified,
            "n_with_population": self.n_with_population,
            "n_with_caveat": self.n_with_caveat,
            "n_with_benefit": self.n_with_benefit,
            "readiness": round(self.readiness, 4),
            "quantification_rate": round(self.quantification_rate, 4),
            "population_rate": round(self.population_rate, 4),
            "caveat_rate": round(self.caveat_rate, 4),
            "benefit_rate": round(self.benefit_rate, 4),
            "bare_rate": round(self.bare_rate, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary(), "per_query": [q.to_dict() for q in self.per_query]}


def aggregate_readiness(per_query: Iterable[Readiness]) -> ReadinessReport:
    rep = ReadinessReport()
    for q in per_query:
        rep.n_queries += 1
        if q.found_gold:
            rep.n_found += 1
        if q.ready:
            rep.n_ready += 1
        if q.quantified:
            rep.n_quantified += 1
        if q.has_population:
            rep.n_with_population += 1
        if q.has_caveat:
            rep.n_with_caveat += 1
        if q.has_benefit:
            rep.n_with_benefit += 1
        rep.per_query.append(q)
    return rep


def readiness_from_rows(
    rows: Sequence[int],
    store: Any,
    gold_doc_ids: Iterable[str],
    *,
    qid: str = "",
    max_chars: int = 4000,
) -> Readiness:
    """Convenience wrapper over row ids plus a chunk store."""
    passages: list[RetrievedPassage] = []
    for row in rows:
        chunk = store.get(int(row))
        if chunk is None:
            continue
        passages.append(
            RetrievedPassage(
                row_id=int(row),
                doc_id=chunk.doc_id,
                text=(chunk.text or "")[:max_chars],
                section_kind=chunk.section_kind or "",
            )
        )
    return assess_readiness(passages, gold_doc_ids, qid=qid)


def compare_readiness(
    baseline: ReadinessReport, candidate: ReadinessReport
) -> dict[str, Any]:
    """Baseline vs candidate with the direction of each change made explicit."""
    out: dict[str, Any] = {}
    for name in ("readiness", "quantification_rate", "population_rate",
                 "caveat_rate", "bare_rate"):
        b, c = getattr(baseline, name), getattr(candidate, name)
        out[name] = {
            "baseline": round(b, 4),
            "candidate": round(c, 4),
            "delta": round(c - b, 4),
            # Higher is better for everything except bare_rate.
            "better": (c < b) if name == "bare_rate" else (c > b),
        }
    return out