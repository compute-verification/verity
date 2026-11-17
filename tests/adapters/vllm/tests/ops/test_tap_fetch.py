"""`ops/pod_norm_tap.sh` and `ops/pod_router_tap.sh` fetch their pinned vLLM sources with curl's own retry, `--retry 5`,
which waits out a 429 or a 5xx from GitHub in any curl from 7.66.0 on (the bootstrap refuses an older one); a file that
never comes fails the tap in one line naming it, and a file already fetched is not fetched again. The retry itself is
curl's: the stub `curl` here records how it is called and answers at once."""
import subprocess
from pathlib import Path

import pytest

OPS = Path(__file__).resolve().parents[5] / "adapters" / "vllm" / "verity_vllm" / "ops"
TAPS = {"pod_norm_tap.sh": ("NT", "NORM-TAP-FAIL", "type_convert.cuh"),
        "pod_router_tap.sh": ("RT", "ROUTER-TAP-FAIL", "csrc/cuda_compat.h")}
CURL = """#!/bin/bash
echo "$*" >> "$CALLS"
while [ $# -gt 0 ]; do case $1 in -o) out=$2; shift;; https://*) url=$1;; esac; shift; done
if [ -n "${CURL_REFUSE:-}" ] && [[ $url == *"$CURL_REFUSE" ]]; then
  echo "curl: (22) The requested URL returned error: 429" >&2; exit 22
fi
echo "// $url" > "$out"
"""


def _tap(tmp_path: Path, script: str, refuse: str = "") -> tuple[subprocess.CompletedProcess, list[str]]:
    var, _, _ = TAPS[script]
    bin_ = tmp_path / "venv" / "bin"
    bin_.mkdir(parents=True, exist_ok=True)
    for name, body in (("curl", CURL), ("nvcc", "#!/bin/sh\necho 'Build cuda_12.9'\n"), ("ninja", "#!/bin/sh\n"),
                       ("verity-vllm", "#!/bin/sh\nexit 1\n")):
        (bin_ / name).write_text(body)
        (bin_ / name).chmod(0o755)
    calls = tmp_path / "calls"
    env = {"PATH": "/usr/bin:/bin", "VENV": str(tmp_path / "venv"), "CUDA_HOME": str(tmp_path / "cuda"),
           var: str(tmp_path / "tap"), "CALLS": str(calls), "CURL_REFUSE": refuse}
    r = subprocess.run(["bash", str(OPS / script)], env=env, capture_output=True, text=True)
    return r, calls.read_text().splitlines() if calls.exists() else []


@pytest.mark.parametrize("script", sorted(TAPS))
def test_each_pinned_source_is_fetched_with_curls_retry_once_and_not_again(tmp_path, script):
    r, calls = _tap(tmp_path, script)
    assert len(calls) == 3, r.stdout
    for call in calls:
        flags = call.split()
        assert "-fsSL" in flags and flags[flags.index("--retry") + 1] == "5" and "--retry-all-errors" not in flags, call
    assert "fetch" not in r.stdout.splitlines()[-1], r.stdout
    (tmp_path / "calls").unlink()
    _, again = _tap(tmp_path, script)
    assert again == []


@pytest.mark.parametrize("script", sorted(TAPS))
def test_a_source_curl_gives_up_on_fails_the_tap_naming_it(tmp_path, script):
    _, prefix, name = TAPS[script]
    r, _ = _tap(tmp_path, script, refuse=name)
    assert r.returncode == 1
    assert r.stdout.splitlines()[-1] == f"{prefix} fetch {name}"
