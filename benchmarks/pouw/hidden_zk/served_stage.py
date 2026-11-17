"""`served_zk/stage_tile.py` (PoUW's prover, `kernels/served_zk/`) with its Flock staging: `verity_flock`'s
`class_statement.stage_unit`, through this directory's `hash_scratch` pieces and, under ``HZ_STAGE=fast``, `fast_stage`.

  python benchmarks/pouw/hidden_zk/served_stage.py COMMIT_DIR PARAMS UNIT OUT    as `stage_tile.py`'s
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import hash_scratch as H
from verity_config.settings import of
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import ir_lower as IL

from verity.protocols.resource_accounting.compute.pouw.prover.kernels.served_zk import stage_tile


class Stager(stage_tile.Stager):
    def stage(self, d: Any, name: str, in_rows: list, out_rows: list, rows: dict, out: Path, seed: bytes, salts: dict,
              staging: str) -> dict:
        if staging == "fast":
            import fast_stage
            fast_stage.install(out)
        bad = H.check()
        assert not any(bad.values()), f"hash_scratch's pieces disagree with their primitives: {bad}"
        IL.PIECES.update(H.PIECES)
        got = CS.stage_unit(d, name, in_rows, out_rows, rows, out, seed, salts=salts)
        text = fast_stage.composed() if staging == "fast" else (out / "circuit.txt").read_text()
        got = {**got, "circuit_sha512": FC.digest(text), "class_sha512": FC.class_digest(text),
               "circuit_bytes": fast_stage.size(text) if staging == "fast" else len(text)}
        if staging == "fast":
            del text
            fast_stage.release()
        return got


if __name__ == "__main__":
    stage_tile.main(stager=Stager(), staging=of(__file__).HZ_STAGE)
