"""Tests for JATS section-kind classification with inheritance.

Full-text articles nest, and the inner headings are specific rather than coarse:
"Search strategy", "Eligibility criteria" and "Study selection" sit under
"Methods". Without inheritance those sections carry no kind, and an unclassified
section gets no dependency edges -- so the dependency graph is blind to exactly the
Methods detail a Results chunk needs, which is the structure it exists to model.

Measured on live PMC Open Access RCT XML, the effect is large:

    classified by own title    354 / 1,484  (23.9%)
    classified by inheritance  893 / 1,484  (+60.2 points)
    BEFORE                     23.9%
    AFTER                      84.0%

Two earlier versions were wrong in ways that produced plausible numbers rather
than errors, and both are pinned here: a flat ``node.iter("sec")`` walk never
populated the inherited context, and keeping that flat walk while adding recursion
double-processed every nested section.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from medrag.data.pubmed_parser import parse_pmc_xml, section_kind


def _wrap(body: str) -> str:
    return f"""<?xml version="1.0"?>
<pmc-articleset><article>
 <front><article-meta>
  <article-id pub-id-type="pmid">90000001</article-id>
  <title-group><article-title>A nested trial</article-title></title-group>
 </article-meta></front>
 <body>{body}</body>
</article></pmc-articleset>
"""


def test_unrecognised_subheading_inherits_its_parent_kind() -> None:
    body = """
    <sec><title>Methods</title>
      <p>Randomised allocation used a permuted block sequence.</p>
      <sec><title>Search strategy</title>
        <p>We searched MEDLINE to January 2020.</p></sec>
      <sec><title>Eligibility criteria</title>
        <p>Adults with confirmed disease were eligible.</p></sec>
    </sec>"""
    doc = parse_pmc_xml(_wrap(body))[0]
    by_title = {s.title: s for s in doc.sections}
    assert by_title["Methods"].kind == "methods"
    # The core regression: these were both "" before inheritance.
    assert by_title["Search strategy"].kind == "methods", by_title["Search strategy"]
    assert by_title["Eligibility criteria"].kind == "methods", by_title["Eligibility criteria"]
    print("  ok  Methods sub-headings inherit 'methods'")


def test_inheritance_is_only_from_a_classified_parent() -> None:
    """An unrecognised heading must not mask the enclosing classified one."""
    body = """
    <sec><title>Results</title>
      <p>Event rates differed between groups.</p>
      <sec><title>Some unrecognised wrapper</title>
        <p>Body text.</p>
        <sec><title>Secondary endpoint</title><p>More numbers.</p></sec>
      </sec>
    </sec>"""
    doc = parse_pmc_xml(_wrap(body))[0]
    by_title = {s.title: s for s in doc.sections}
    assert by_title["Some unrecognised wrapper"].kind == "results", by_title
    assert by_title["Secondary endpoint"].kind == "results", by_title
    print("  ok  unrecognised wrappers do not break inheritance")


def test_classified_subheading_keeps_its_own_kind() -> None:
    """Inheritance must never override a heading that classifies itself."""
    body = """
    <sec><title>Methods</title>
      <p>Design and procedure.</p>
      <sec><title>Results</title><p>Numbers here.</p></sec>
    </sec>"""
    doc = parse_pmc_xml(_wrap(body))[0]
    by_title = {s.title: s for s in doc.sections}
    assert by_title["Results"].kind == "results", by_title["Results"]
    print("  ok  a self-classifying sub-heading is not overridden")


def test_limitations_read_as_discussion_at_any_depth() -> None:
    """'Limitations' is exactly the caveat text the readiness metric looks for."""
    body = """
    <sec><title>Discussion</title>
      <p>Our findings should be interpreted with caution.</p>
      <sec><title>Limitations</title><p>The trial was not powered for rare events.</p></sec>
    </sec>"""
    doc = parse_pmc_xml(_wrap(body))[0]
    by_title = {s.title: s for s in doc.sections}
    assert by_title["Limitations"].kind == "discussion", by_title["Limitations"]
    print("  ok  nested Limitations sections are classified as discussion")


def test_no_section_is_visited_twice() -> None:
    """Regression: flat iter() plus recursion double-processed nested sections.

    Detected as duplicated section text within a single document.
    """
    body = """
    <sec><title>Methods</title>
      <p>Outer body.</p>
      <sec><title>Search strategy</title><p>Inner body.</p>
        <sec><title>Databases</title><p>Deepest body.</p></sec>
      </sec>
    </sec>"""
    doc = parse_pmc_xml(_wrap(body))[0]
    texts = [s.text for s in doc.sections if s.text]
    assert len(texts) == len(set(texts)), (
        f"a section was visited twice: {[t[:24] for t in texts]}"
    )
    titles = [s.title for s in doc.sections]
    assert titles == ["Methods", "Search strategy", "Databases"], titles
    print(f"  ok  each section visited once, in document order: {titles}")


def test_parent_text_is_not_duplicated_into_children() -> None:
    body = """
    <sec><title>Methods</title>
      <p>Parent prose.</p>
      <sec><title>Search strategy</title><p>Child prose.</p></sec>
    </sec>"""
    doc = parse_pmc_xml(_wrap(body))[0]
    joined = " ".join(s.text for s in doc.sections)
    assert joined.count("Parent prose") == 1, "parent prose leaked into the child"
    print("  ok  parent prose stays with the parent")


def test_numbered_headings_classify() -> None:
    """PMC sections are usually numbered ('3. Results'), which must still work."""
    for title, want in (("1. Introduction", "background"), ("3. Results", "results"),
                        ("4. Discussion", "discussion"), ("2. Methods", "methods")):
        assert section_kind(title) == want, (title, section_kind(title))
    print("  ok  numbered headings classify correctly")


def main() -> int:
    tests = [
        test_unrecognised_subheading_inherits_its_parent_kind,
        test_inheritance_is_only_from_a_classified_parent,
        test_classified_subheading_keeps_its_own_kind,
        test_limitations_read_as_discussion_at_any_depth,
        test_no_section_is_visited_twice,
        test_parent_text_is_not_duplicated_into_children,
        test_numbered_headings_classify,
    ]
    print("JATS section-kind tests")
    for test in tests:
        test()
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())