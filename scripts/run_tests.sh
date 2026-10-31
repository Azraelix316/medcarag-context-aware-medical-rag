#!/usr/bin/env bash
# Run every test suite and report a combined result.
#
# Usage: bash scripts/run_tests.sh [--fast]
#   --fast  skip the BM25 memory-scaling test (it builds a 100k-document index)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"

SUITES=(
  "tests/test_bm25.py"
  "tests/test_pipeline.py"
  "tests/test_taxonomy_benchmark.py"
  "tests/test_crawl_partitioning.py"
  "tests/test_dense_index_policy.py"
  "tests/test_citations.py"
  "tests/test_mesh_graph.py"
  "tests/test_embed_sharding.py"
  "tests/test_verify_probes.py"
  "tests/test_dependency_graph.py"
  "tests/test_context_provision.py"
  "tests/test_citation_labels.py"
  "tests/test_sibling_bleed.py"
  "tests/test_cocitation.py"
  "tests/test_page_index.py"
  "tests/test_readiness.py"
  "tests/test_benefit_taxonomy.py"
  "tests/test_multihop_context.py"
  "tests/test_section_kinds.py"
  "tests/test_read_test.py"
  "tests/test_attribution.py"
  "tests/test_acquire_filter.py"
  "tests/test_acquire_manifest.py"
  "tests/test_attribution_agent.py"
  "tests/test_backfill.py"
  "tests/test_shard_accounting.py"
  "tests/test_page_scoring.py"
  "tests/test_dense_sharding.py"
  "tests/test_bm25_csc_fill.py"
  "tests/test_corpus_shards.py"
  "tests/test_crawl_sweep.py"
  "tests/test_segmented.py"
  "tests/test_page_blocks.py"
  "tests/test_graph_inputs.py"
  "tests/test_20m_pipeline.py"
)

overall=0
declare -a summary=()

# How many tests a suite actually ran, read from its own closing line.
#
# This exists because "ALL SUITES PASSED" says nothing about coverage. For a while
# four test files were absent from the suite list entirely, and the count of '^def
# test_' in the tree -- which is how the number used to be reported here -- counts
# files that are never executed and functions that are not tests. Asking each suite
# what it ran is the only version of this number that means anything.
count_tests() {
  "$PY" "$1" 2>/dev/null | grep -oiE "all [0-9]+( BM25)? tests passed" \
    | grep -oE "[0-9]+" | head -1 || echo 0
}

total=0
for suite in "${SUITES[@]}"; do
  printf '\n========== %s ==========\n' "$suite"
  if [ "${1:-}" = "--fast" ] && [ "$suite" = "tests/test_bm25.py" ]; then
    if "$PY" -c "
import sys; sys.argv=['t']
sys.path.insert(0,'tests')
import test_bm25 as t
for fn in [t.test_matches_reference, t.test_save_load_roundtrip,
           t.test_query_term_repetition, t.test_empty_and_missing_terms,
           t.test_incremental_and_resume]:
    fn()
print('  ok  5 BM25 correctness tests (memory test skipped via --fast)')
" 2>&1 | tail -12; then
      summary+=("PASS  $suite (fast)"); total=$((total + $(count_tests "$suite")))
    else
      summary+=("FAIL  $suite (fast)"); overall=1
    fi
  elif "$PY" "$suite" 2>&1 | tail -40; then
    summary+=("PASS  $suite"); total=$((total + $(count_tests "$suite")))
  else
    summary+=("FAIL  $suite"); overall=1
  fi
done

printf '\n========== summary ==========\n'
for line in "${summary[@]}"; do
  echo "  $line"
done
echo "  ${#summary[@]} suites, ${total} tests"

if [ $overall -eq 0 ]; then
  echo "  ALL SUITES PASSED"
else
  echo "  SOME SUITES FAILED"
fi
exit $overall
