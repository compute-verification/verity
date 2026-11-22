"""run.sh publishes nothing it keeps into $OUT, which custody uploads: it writes no kept file there, refuses a keep that
custody would publish, and lets a log leave only as `public_log.py` lets it."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


RUN_SH = Path(__file__).resolve().parents[3] / "benchmarks" / "prove_path" / "run.sh"
PUBF = RUN_SH.parents[1] / "one_stage" / "public_files.py"
PL = RUN_SH.parent / "public_log.py"
#: the files a run keeps in PP_KEEP and never writes under $O: a window's partition and the files beside it, the advice's
#: length, the tools' error output and logs no reader takes from $OUT, the compilers' argv (local paths), the machine's shape
KEPT = ("partition.json", "index.json", "window.json", "m*-public.bin", "audit.err", "compilers.json", "advice.log",
        "sync.log", "build-lean.log", "build-sampled-proofs.log", "m0-build", "sets-$SET.log", "fetch-main.log", "keep.log",
        "tie-fixture.err", "standin.err")


def test_no_kept_file_is_written_under_out():
    text = RUN_SH.read_text()
    lines = [line for line in text.splitlines() if "$O/" in line and not line.lstrip().startswith("#")]
    bare = [k.replace("*", "") for k in KEPT]
    hit = sorted({k for line in lines for k in bare if f"$O/{k}" in line or ("$O/" in line and f"/{k}" in line
                                                                              and "$K/" not in line)})
    assert not hit, f"run.sh writes {hit} under $OUT; they go to $K (PP_KEEP)"
    assert 'K=${PP_KEEP:-$W/keep}' in text and "chmod 700" in text


def test_pp_keep_under_out_or_the_run_dir_is_refused_before_anything_is_written(tmp_path):
    lines = RUN_SH.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("K=${PP_KEEP"))
    code = "\n".join(lines[i:i + 6]) + "\n"
    for keep in (tmp_path / "run" / "out" / "k", tmp_path / "run" / "k"):
        got = subprocess.run(["bash", "-c", f"O={tmp_path / 'run' / 'out'}; RESEARCH_RUN_DIR={tmp_path / 'run'}; "
                              f"PP_KEEP={keep}; W={tmp_path}\n{code}"], capture_output=True, text=True)
        assert got.returncode == 2 and "which custody publishes" in got.stdout and not keep.exists()
    keep = tmp_path / "elsewhere"
    got = subprocess.run(["bash", "-c", f"O={tmp_path / 'run' / 'out'}; RESEARCH_RUN_DIR={tmp_path / 'run'}; "
                          f"PP_KEEP={keep}; W={tmp_path}\n{code}"], capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert keep.stat().st_mode & 0o777 == 0o700


def test_public_log_lets_only_what_its_reader_reads_go():
    """Each tagged line leaves with only the fields `proof_layers.read_prove_path` reads; a field it doesn't read (the
    advice's length, the leak counted against it, a charge, a served scope's population, the count of sampled proof
    blocks, which a template proved whole grows with the advice, a member's name) is held, and an error line leaves as its
    exception's class."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("public_log", PL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.public_line('RECORDS {"ok": true, "accepted": true, "negatives_refused": true, "draw_counts": {"t": 2}, '
                         '"population": 9, "draw": [9], "circuit": "9", "leak": {"advice_bits": 9, "bound": 9}}') \
        == 'RECORDS {"accepted": true, "negatives_refused": true, "ok": true}'
    rank = {"accepted": True, "ok": True, "population": 9, "draw": [9], "draw_counts": {"t": 2}, "seconds": 9,
            "charge": {"c": 9}, "advice_before_draw": True, "forged": {"x": 9},
            "per_unit": {"t": {"units": 2, "proof_bytes": 3, "prove_s_per_unit": 1.5, "lean_verify_s_per_unit": 0.5, "x": 9}}}
    line = {"advice_bits": 9, "delta": 9, "ok": True, "proof": "zk", "coins": "gate", "run_root": "9",
            "transfer_gates": 9, "ranks": {"0": rank}}
    assert json.loads(m.public_line("PROVE-PATH " + json.dumps(line)).split(" ", 1)[1]) == {
        "ok": True, "proof": "zk", "coins": "gate", "ranks": {"0": {
            "accepted": True, "per_unit": {"t": {"proof_bytes": 3}}}}}
    a5 = {"source": "served", "draw": [9], "audit": {"accepted": True, "wrong_units": 9, "why": {"x": "9"},
                                                     "verdicts": {"lean-verify": True, "x": False}}}
    assert m.public_line("A5 " + json.dumps(a5)) == 'A5 {"audit": {"accepted": true, "verdicts": {"lean-verify": true, "x": false}}}'
    assert m.public_line('A5 {"memory": {"pool": 9}}') == "A5 {}" and m.public_line("PROOF-EDITS {}") is None
    assert m.public_line("ValueError: request 9 of user-9") == "ValueError: held in the kept log"
    assert "9" not in m.public_line("PROVE-PATH not json 9") and m.public_line("[prove-path commit] rank 0: {}") is None


