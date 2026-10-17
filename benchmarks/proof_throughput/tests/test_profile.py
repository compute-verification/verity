import gzip
import json

from verity_proof_throughput import cli
from verity_proof_throughput import profile as PF

SCRIPT = """    flock-verify 
\t    7f01 l_Flock_Sha512_compress
\t    7f02 l_Flock_MerkleScheme_climb
\t    7f03 l_Flock_ligerito___lambda__2
\t    7f04 main

flock-verify
\t    7f05 l_Flock_F128_mul
\t    7f06 l_Array_foldlMUnsafe_fold___at___Flock_bindAndZerocheck_spec__3
\t    7f04 main

flock-verify
\t    7f07 lean_dec_ref_cold
\t    7f08 l_Lean_Json_Parser_anyCore
\t    7f09 l_Flock_Circuit_parse
\t    7f04 main

flock-verify
\t    7f0a l_Flock_F128_mul
\t    7f0b l_Flock_queryLevel
\t    7f03 l_Flock_ligerito___lambda__2

flock-verify
\t    7f0c [unknown]
"""

STDERR = "inputs: 16000 ms\nsetup: 48000 ms\n/s/l0001-9: 104000 ms\n"


def test_each_sample_is_its_innermost_phase_and_kernel():
    got = [PF.attribute(f) for f in PF.samples(SCRIPT.splitlines(keepends=True))]
    assert got == [("merkle_paths", "sha512"), ("zerocheck", "gf128"), ("circuit_parse", "alloc_rc"),
                   ("ligerito", "gf128"), ("other", "other")]


def test_a_samples_stage_is_its_outermost_stage_frame():
    lp = "lp_flock__verifier_"
    scope = [f"{lp}Flock_Net_parse", f"{lp}Flock_HmOut_parse", f"{lp}Flock_ProvedScope_check", f"{lp}verifyCmd"]
    setup = [f"{lp}Flock_HmOut_parse", f"{lp}Flock_Stmt_setupHidden", f"{lp}buildSession", f"{lp}verifyCmd"]
    fold = [f"{lp}Flock_F128_clmulGo", f"{lp}Flock_Stmt_baseOf", f"{lp}Flock_lincheck", f"{lp}Flock_verifyRep",
            f"{lp}Flock_verify", f"{lp}verifyCmd"]
    assert [PF.stage(f) for f in (scope, setup, fold, ["main"])] == ["scope_check", "setup", "verify", "other"]
    assert PF.attribute(fold) == ("statement", "gf128")
    s = PF.summarize([scope, setup, fold, fold])
    assert s["stage"] == {"verify": 0.5, "scope_check": 0.25, "setup": 0.25}
    assert s["stage_phase"]["verify/statement"] == 0.5 and s["stage_phase"]["setup/circuit_parse"] == 0.25


def test_summary_shares_and_the_verifiers_own_seconds():
    s = PF.summarize(PF.samples(SCRIPT.splitlines()))
    assert s["samples"] == 5 and s["phase"]["ligerito"] == 0.2 and s["kernel"]["gf128"] == 0.4
    assert s["phase_kernel"]["merkle_paths/sha512"] == 0.2
    assert PF.times(STDERR) == {"inputs_s": 16.0, "setup_s": 48.0, "session_s": 56.0}


def test_profile_phases_writes_a_result(tmp_path):
    (tmp_path / "out").mkdir()
    with gzip.open(tmp_path / "script.gz", "wt") as f:
        f.write(SCRIPT)
    (tmp_path / "time").write_text("170.2 168.0 1.5 4630000\n")
    (tmp_path / "err").write_text(STDERR)
    (tmp_path / "stdout").write_text('VERDICT {"session": "s", "accepted": true, "why": ""}\n')
    rc = cli.main(["profile-phases", "--script", str(tmp_path / "script.gz"), "--time", str(tmp_path / "time"), "--stderr",
                   str(tmp_path / "err"), "--stdout", str(tmp_path / "stdout"), "--out", str(tmp_path / "result.json")])
    doc = json.loads((tmp_path / "result.json").read_text())
    ms = {m["name"]: m["value"] for m in doc["measurements"]}
    assert rc == 0 and doc["validation"]["status"] == "passed"
    assert ms["profile.cpu_s"] == 169.5 and ms["profile.threads_busy"] == round(169.5 / 170.2, 3)
    assert ms["profile.session_s"] == 56.0 and ms["profile.phase.ligerito"] == 0.2
    assert ms["profile.unclocked_s"] == round(170.2 - 16.0 - 48.0 - 56.0, 3)
    assert json.loads((tmp_path / "out" / "profile.json").read_text())["samples"] == 5
