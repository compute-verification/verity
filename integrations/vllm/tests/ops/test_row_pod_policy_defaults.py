"""[r19-int-12, F-r19-int-13] row_pod.sh owns the Match policy of record.

veritor policy r17-rules-1 [checker] (ruling M-1332): MATCH_IMPL=fast inside MATCH_PIPELINE=shared is the Match of record; the record
checker is the asynchronous audit route.  In R17/R18 the launcher run_row.sh carried the default (`have MATCH_IMPL || ENVS+=("MATCH_IMPL=fast")`)
and did not move into Verity with row_pod.sh (research run replaced it): every R19 launch without the env ran the record checker (#73: GM-01
single-core to its 14400 s timeout; its FP8 twin under cheap's explicit env: 12 min).  The default lives in the runner now, before the first
stage; an explicit MATCH_IMPL=record / MATCH_PIPELINE= still selects the caller's choice.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]
ROW_POD = TREE / "verity_vllm" / "ops" / "row_pod.sh"


def _defaults_block() -> str:
    src = ROW_POD.read_text()
    m = re.search(r"^export MATCH_IMPL=.*$", src, re.MULTILINE)
    assert m, "MATCH_IMPL default not found in row_pod.sh"
    return m.group(0)


def test_the_match_of_record_is_the_default_before_any_stage_and_the_caller_can_override():
    src = ROW_POD.read_text()
    line = _defaults_block()
    assert line == "export MATCH_IMPL=${MATCH_IMPL:-fast} MATCH_PIPELINE=${MATCH_PIPELINE:-shared}"
    assert src.index(line) < src.index("tl stage.build start") and src.index(line) < src.index('if [ "${MATCH_PIPELINE:-}" = "shared" ]')
    assert "r17-rules-1" in src and "M-1332" in src            # provenance of the policy beside the default
    env = {k: v for k, v in os.environ.items() if k not in ("MATCH_IMPL", "MATCH_PIPELINE")}
    out = subprocess.run(["bash", "-c", line + '; echo "$MATCH_IMPL/$MATCH_PIPELINE"'], capture_output=True, text=True, env=env, check=False).stdout.strip()
    assert out == "fast/shared"
    out = subprocess.run(["bash", "-c", line + '; echo "$MATCH_IMPL/$MATCH_PIPELINE"'], capture_output=True, text=True,
                         env={**env, "MATCH_IMPL": "record", "MATCH_PIPELINE": "standalone"}, check=False).stdout.strip()
    assert out == "record/standalone"                          # the audit checker stays selectable by name
    assert subprocess.run(["bash", "-n", str(ROW_POD)], capture_output=True, check=False).returncode == 0
