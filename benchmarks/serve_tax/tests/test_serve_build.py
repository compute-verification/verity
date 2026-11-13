"""run.sh's vLLM build: the pin unless SERVE_VLLM names another, which is served from an environment of its own and
compiles into caches of its own, so that the engines at the pin find nothing it wrote."""
from __future__ import annotations

import re
from pathlib import Path

SH = (Path(__file__).resolve().parents[1] / "run.sh").read_text()
PIN = "d9105ea80"
LEASED = 'if [ "${1:-}" = leased ]; then'


def build_block() -> tuple[str, str]:
    """run.sh's choice of environment: the pin's branch and the other builds'."""
    block = SH[SH.index(f'if [ "$SERVE_VLLM" = {PIN} ]; then'):SH.index("VENV=${VENV:-$venv}")]
    pin, other = block.split("\nelse\n")
    return pin, other


def test_the_build_is_the_pin_unless_the_job_names_another():
    assert f"SERVE_VLLM=${{SERVE_VLLM:-{PIN}}}" in SH
    pin, other = build_block()
    assert "venv312" in pin and "vllm_env.sh" not in pin
    # another build's environment is the adapter's script's, whose last line is its path
    assert 'adapters/vllm/verity_vllm/ops/vllm_env.sh" | tail -n 1)' in other and "venv312" not in other


def test_a_build_beside_the_pin_compiles_into_caches_under_its_own_environment():
    pin, other = build_block()
    for var in ("VLLM_CACHE_ROOT", "FLASHINFER_WORKSPACE_BASE", "TRTLLM_DG_CACHE_DIR", "TRITON_CACHE_DIR"):
        assert re.search(rf"\b{var}=\$venv/cache\b", other), var
        assert var not in pin, f"{var}: the pin's engines keep the node's caches"
    # exported before the leased branch: the job's second entry and every arm's server inherit them
    assert SH.index("export VLLM_CACHE_ROOT=$venv/cache/vllm") < SH.index(LEASED)
    # and the record of FlashInfer's caches reads the one the engines write
    assert '"tensorrt_llm_cache:${TRTLLM_DG_CACHE_DIR:-$HOME/.tensorrt_llm/cache}"' in SH


def test_the_environment_must_hold_the_build_the_job_names():
    check = SH[SH.index("from vllm._version import __commit_id__"):SH.index(LEASED)]
    assert '[ "$v" = "$SERVE_VLLM" ] || {' in check and "exit 2" in check
    assert 'cp "$VENV/freeze.txt" "$OUT/vllm_env.freeze.txt"' in check
