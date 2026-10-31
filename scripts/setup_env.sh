#!/usr/bin/env bash
# Bootstrap the local environment. Safe to re-run; idempotent.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
LOG="$ROOT/logs/setup.log"
mkdir -p "$ROOT/logs"

log() { printf '[setup %s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

if [ ! -d "$VENV" ]; then
  log "creating venv at $VENV"
  python3 -m venv "$VENV" || { log "FAILED: venv creation"; exit 1; }
fi

PY="$VENV/bin/python"
log "python: $($PY --version 2>&1)"

if ! "$PY" -m pip --version >/dev/null 2>&1; then
  log "bootstrapping pip"
  "$PY" -m ensurepip --upgrade >>"$LOG" 2>&1
fi
"$PY" -m pip install --upgrade pip setuptools wheel >>"$LOG" 2>&1
log "pip: $($PY -m pip --version 2>&1)"

# ---------------------------------------------------------------------------
# Torch install. A cheap HEAD probe against the CUDA index decides whether to
# pull the GPU build; a failed probe or a failed install falls back to CPU.
# We do NOT use `pip install --dry-run`, which downloads the full ~2.5GB wheel
# just to resolve it.
# ---------------------------------------------------------------------------
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
want_cuda=0
if command -v nvidia-smi >/dev/null 2>&1; then
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$TORCH_INDEX/torch/" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && want_cuda=1
fi

if [ "$want_cuda" = "1" ]; then
  log "CUDA index reachable -> installing torch (cu121)"
  if ! "$PY" -m pip install 'torch>=2.2' --index-url "$TORCH_INDEX" >>"$LOG" 2>&1; then
    log "cu121 torch install failed -> falling back to CPU wheels"
    "$PY" -m pip install 'torch>=2.2' >>"$LOG" 2>&1 || { log "FAILED: torch install"; exit 1; }
  fi
else
  log "no CUDA index (probe=$code) -> installing torch (cpu)"
  "$PY" -m pip install 'torch>=2.2' >>"$LOG" 2>&1 || { log "FAILED: torch install"; exit 1; }
fi

log "installing remaining requirements"
"$PY" -m pip install -r "$ROOT/requirements.txt" >>"$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "FAILED: requirements install (rc=$rc); tail of $LOG:"
  tail -40 "$LOG"
  exit $rc
fi

log "verifying imports"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import importlib, sys
mods = ["numpy", "scipy", "torch", "transformers", "sentence_transformers",
        "faiss", "lxml", "fastapi", "pyarrow", "loguru", "tqdm", "requests"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  ok   {m:22s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        bad.append((m, repr(e)))
        print(f"  FAIL {m:22s} {e!r}")
try:
    import torch
    print(f"  torch cuda available: {torch.cuda.is_available()} devices={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"    gpu{i}: {p.name} sm_{p.major}{p.minor} {p.total_memory/2**30:.1f}GiB")
except Exception as e:
    print(f"  torch device probe failed: {e!r}")
sys.exit(1 if bad else 0)
PYEOF
rc=$?
log "verify rc=$rc -> environment ready"
exit $rc
