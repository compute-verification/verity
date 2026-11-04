"""TINY's checkpoint tie inputs for `benchmarks/prove_path/run.sh`'s ``PP_WEIGHTS_TIE`` arm: checkpoint X
(`test_window_circuit._checkpoint`, seed 3) and the manifest pinning it in ``DIR/x``, checkpoint Y (seed 4, another
checkpoint's words) in ``DIR/y``, and the registered φ (`test_steps._registered`'s, default seed 7) in ``DIR/phi.json``,
with the window grain `test_window_circuit.GRAIN` printed for ``verity-vllm weights-tie --grain``.

    (cd integrations/vllm && python -m tests.program.predict.weights_tie_fixture DIR)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import GRAIN, _checkpoint


def main(argv: list[str]) -> int:
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    (x, xm), (y, ym) = _checkpoint(out / "x"), _checkpoint(out / "y", seed=4)
    (out / "phi.json").write_text(json.dumps(S._registered().phi.to_json(), indent=1, sort_keys=True) + "\n")
    print(json.dumps({"x": x, "x_manifest": xm, "y": y, "y_manifest": ym, "phi": str(out / "phi.json"),
                      "grain": f"{GRAIN.gemm_words},{GRAIN.rows},{GRAIN.select}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
