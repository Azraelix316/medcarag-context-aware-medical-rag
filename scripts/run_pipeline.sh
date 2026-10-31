#!/usr/bin/env bash
# Supervised long-running pipeline: acquire -> chunk -> index -> benchmark.
#
# Runs detached with setsid so it survives the parent shell, logs every stage to
# its own file, and is safe to re-invoke: each stage detects its completed
# artifacts and skips itself, so a killed run resumes where it stopped.
#
# Usage:  bash scripts/run_pipeline.sh [config.yaml ...]
#         bash scripts/run_pipeline.sh --status
#         bash scripts/run_pipeline.sh --stop
#
# Config paths may be given bare ("configs/mid.yaml") or with a leading -c;
# both forms are normalised to -c <path> below.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p logs data
PY="$ROOT/.venv/bin/python"
PIDFILE="logs/pipeline.pid"
LOGDIR="logs"

CONFIGS=()
STATUS=0
STOP=0
for arg in "$@"; do
  case "$arg" in
    --status) STATUS=1 ;;
    --stop)   STOP=1 ;;
    -c|--config)
      # Next argument is the path; handled by the loop's pending flag below.
      CONFIGS+=("__PENDING__")
      ;;
    __PENDING__)
      CONFIGS+=("$arg")
      ;;
    -*)
      echo "unknown option: $arg" >&2
      exit 2
      ;;
    *)
      CONFIGS+=("$arg")
      ;;
  esac
done

# Normalise to repeated -c <path> pairs.
ARGS=()
i=0
while [ $i -lt ${#CONFIGS[@]} ]; do
  if [ "${CONFIGS[$i]}" = "__PENDING__" ]; then
    i=$((i + 1))
    continue
  fi
  ARGS+=("-c" "${CONFIGS[$i]}")
  i=$((i + 1))
done
CONFIGS=("${ARGS[@]}")

if [ "${STATUS:-0}" = "1" ]; then
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "pipeline running (pid $(cat "$PIDFILE"))"
    ps -o pid,etime,rss,cmd -p "$(cat "$PIDFILE")"
  else
    echo "pipeline not running"
  fi
  echo "--- recent log ---"
  tail -n 15 "$LOGDIR/pipeline.log" 2>/dev/null || echo "(no log)"
  exit 0
fi

if [ "${STOP:-0}" = "1" ]; then
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null && echo "sent TERM to $(cat "$PIDFILE")"
    rm -f "$PIDFILE"
  else
    echo "no pidfile"
  fi
  exit 0
fi

if [ ! -x "$PY" ]; then
  echo "error: $PY not found. Run: bash scripts/setup_env.sh" >&2
  exit 1
fi

# Default config stack when none given.
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(-c configs/default.yaml -c configs/production.yaml)
fi

if [ ! -f "${CONFIGS[1]}" ]; then
  echo "error: config not found: ${CONFIGS[1]}" >&2
  exit 2
fi

run_stage() {
  local name="$1"; shift
  local log="$LOGDIR/$name.log"
  echo "[$(date -u +%H:%M:%S)] starting stage: $name" | tee -a "$LOGDIR/pipeline.log"
  local t0 t1
  t0=$(date +%s)
  "$PY" -m medrag.cli "${CONFIGS[@]}" "$@" >>"$log" 2>&1
  local rc=$?
  t1=$(date +%s)
  if [ $rc -eq 0 ]; then
    echo "[$(date -u +%H:%M:%S)] stage $name OK in $((t1 - t0))s -> $log" | tee -a "$LOGDIR/pipeline.log"
  else
    echo "[$(date -u +%H:%M:%S)] stage $name FAILED rc=$rc after $((t1 - t0))s; tail of $log:" | tee -a "$LOGDIR/pipeline.log"
    tail -n 30 "$log" | tee -a "$LOGDIR/pipeline.log"
    return $rc
  fi
}

# should_run <stage-name> <artifact>
#
# Returns 0 (true) when the stage still needs to run, 1 when its artifact is
# already present. Used so restarting a pipeline after an interruption does not
# repeat hours of completed work.
should_run() {
  local name="$1" marker="$2"
  if [ -n "$marker" ] && [ -e "$marker" ]; then
    echo "[$(date -u +%H:%M:%S)] stage $name skipped (artifact present: $marker)" \
      | tee -a "$LOGDIR/pipeline.log"
    return 1
  fi
  return 0
}

# Resolve the configured artifact paths from the CLI itself, so stage markers
# are correct under config layering (a config file may override data_root).
PATHS_OUT="$("$PY" -m medrag.cli "${CONFIGS[@]}" paths 2>/dev/null)"
path_of() { printf '%s\n' "$PATHS_OUT" | sed -n "s/^$1=//p" | head -1; }
DATA_ROOT="$(path_of data_root)"
CORPUS="$(path_of corpus_dir)"
INDEX="$(path_of index_dir)"
RUNS="$(path_of runs_dir)"
if [ -z "$DATA_ROOT" ] || [ -z "$CORPUS" ] || [ -z "$INDEX" ]; then
  echo "error: could not resolve paths from the CLI" >&2
  printf '%s\n' "$PATHS_OUT" >&2
  exit 2
fi
echo "paths: data_root=$DATA_ROOT corpus=$CORPUS index=$INDEX" | tee -a "$LOGDIR/pipeline.log"

main() {
  {
    echo "=== medrag pipeline start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo "configs: ${CONFIGS[*]}"
    echo "python:  $("$PY" --version 2>&1)"
    "$PY" -m medrag.cli doctor 2>&1 | tail -n 20
  } >>"$LOGDIR/pipeline.log" 2>&1

  # 1. Crawl the corpus. Resumable: re-running skips already-written records.
  #    Categories here come from label heuristics only; see stage 2.
  corpus_manifest="$(ls "$CORPUS"/*/manifest.json 2>/dev/null | head -1)"
  if should_run acquire "$corpus_manifest"; then
    run_stage acquire acquire || return 1
  fi

  # 2. Resolve authoritative MeSH tree positions, then rewrite categories.
  #    Split from the crawl because HTTP lookups would stall the crawl loop.
  #    The taxonomy cache is itself the skip marker, so a resumed run only
  #    fetches descriptors it has not seen.
  run_stage taxonomy taxonomy || return 1
  run_stage recategorize recategorize || return 1

  # 3. Chunk -> chunk store -> BM25 -> FAISS. Embedding blocks are cached on
  #    disk, so a restart resumes at the last completed block.
  if should_run index "$INDEX/dense.faiss"; then
    run_stage index build-index --store-cache 200000 || return 1
  fi

  # 4. Build the evaluation benchmark from the corpus's MeSH structure.
  #    Independent of the index, so it is safe to have completed earlier.
  if should_run benchmark "$RUNS/benchmark/qrels_combined.json"; then
    run_stage benchmark benchmark --n-clinical 3000 --n-multihop 800 || return 1
  fi

  # 5. Retrieval ablation across all four modes.
  #    Reranking costs ~10s/query, so the sample is bounded to keep the whole
  #    ablation inside a few hours rather than running overnight.
  run_stage evaluate evaluate --sample "${EVAL_SAMPLE:-500}" --out "$RUNS/eval_report.json" || return 1

  echo "[$(date -u +%H:%M:%S)] === pipeline complete ===" | tee -a "$LOGDIR/pipeline.log"
}

echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

# Supervisor: re-invoke the pipeline on failure. Every stage is resumable and
# skips its own completed artifacts, so a retry continues from the failure
# point rather than starting over. This is what makes an unattended overnight
# run survive a transient network or OOM failure.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
attempt=1
while :; do
  main
  rc=$?
  if [ $rc -eq 0 ]; then
    break
  fi
  if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    echo "[$(date -u +%H:%M:%S)] giving up after $attempt attempts (rc=$rc)" \
      | tee -a "$LOGDIR/pipeline.log"
    exit $rc
  fi
  delay=$(( attempt * 60 ))
  if [ $delay -gt 900 ]; then delay=900; fi
  echo "[$(date -u +%H:%M:%S)] attempt $attempt failed (rc=$rc); retrying in ${delay}s" \
    | tee -a "$LOGDIR/pipeline.log"
  sleep "$delay"
  attempt=$(( attempt + 1 ))
done
exit 0
