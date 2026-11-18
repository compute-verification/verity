"""`benchmarks/kernels/smoke_h100.sh`'s text, with no card and nothing run: where the script takes the target its builds
are for."""
from __future__ import annotations

import re
from pathlib import Path

SMOKE = Path(__file__).resolve().parents[3] / "benchmarks" / "kernels" / "smoke_h100.sh"


def test_the_smoke_asks_pearl_cs_build_for_its_target_and_writes_none_itself() -> None:
    """The script's one statement of its target is `pearl_c_sm90/build.sh --arch`'s answer, asked before any step and
    refused by name unless it is one target (the pouw suite holds that the verb answers); the machine's record and the
    held card's are compared with that answer, and no line of the script that runs writes an architecture itself."""
    text = SMOKE.read_text()
    code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    asks = [i for i, line in enumerate(code) if line.startswith('TARGET=$(bash "$PEARL/build.sh" --arch)')]
    assert len(asks) == 1 and sum(line.startswith("TARGET=") for line in code) == 1
    assert "PEARL=$POUW/kernels/pearl_c_sm90" in code
    assert "exit 2" in " ".join(code[asks[0]:asks[0] + 2]) and "names no one target" in code[asks[0] + 1]
    assert asks[0] < min(i for i, line in enumerate(code) if re.match(r"\s*step \S+ \S+$", line))
    held = [line for line in code if re.search(r'\[ "\$arch" = "\$TARGET" \]', line)]
    assert len(held) == 2 and all("return 1" in line for line in held)
    assert not [line for line in code if re.search(r"(?<![\w-])sm_?\d{2,3}a?(?![\w-])|\b\d\.\da\b", line)]
