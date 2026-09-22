"""The overhead ledger and its plots: entries validate, unit numbers are x96 extrapolations, plots render."""

from pathlib import Path

import pytest

from verity_numerical.bench import contract, ledger, plots


def test_unit_entry_is_x96_and_component_may_lack_overhead(tmp_path: Path) -> None:
    e = ledger.entry(lane="t", track="A", label="unit", scope="unit", proof_class="NON_ZK_PROOF_DIAGNOSTIC", hardware="cpu",
                     security_target="2^-128", seconds_per_unit=1e-3, B=4096, omitted=["chain linkage"])
    assert e["seconds_per_vu"] == pytest.approx(0.096) and e["overhead"] == pytest.approx(contract.overhead(0.096, 1536))
    c = ledger.entry(lane="t", track="shared", label="hash rate", scope="component", proof_class="COMPONENT", hardware="4090",
                     security_target="n/a", note="Blake3 51 GB/s")
    assert c["overhead"] is None
    with pytest.raises(ValueError):
        ledger.entry(lane="t", track="A", label="no number", scope="vu", proof_class="NON_ZK_PROOF", hardware="x", security_target="2^-128")
    with pytest.raises(ValueError):
        ledger.entry(lane="t", track="C", label="zk w/o assumptions", scope="vu", proof_class="COMPLETE_ZK_BACKEND", hardware="x",
                     security_target="2^-128", overhead=3e6)
    p = ledger.add(e, tmp_path)
    ledger.add(c, tmp_path)
    assert p == tmp_path / "t.jsonl" and len(ledger.load(p)) == 2


def test_cli_add_check_and_plots(tmp_path: Path) -> None:
    d, out = tmp_path / "ledger", tmp_path / "plots"
    assert ledger.main(["add", "--dir", str(d), "--lane", "A-x", "--track", "A", "--label", "first", "--scope", "vu",
                        "--proof-class", "NON_ZK_PROOF_DIAGNOSTIC", "--hardware", "RTX 4090", "--security", "2^-128",
                        "--seconds-per-vu", "0.03", "--B", "4096", "--omitted", "hashing"]) == 0
    assert ledger.main(["add", "--dir", str(d), "--lane", "A-x", "--track", "A", "--label", "packed", "--scope", "vu",
                        "--proof-class", "NON_ZK_PROOF_DIAGNOSTIC", "--hardware", "RTX 4090", "--security", "2^-128",
                        "--overhead", "1e7", "--breakthrough"]) == 0
    assert ledger.main(["check", "--dir", str(d)]) == 0
    assert plots.main(["--dir", str(d), "--out", str(out)]) == 0
    assert (out / "overhead_A-x.png").stat().st_size > 1000 and (out / "overhead_all.png").exists()
