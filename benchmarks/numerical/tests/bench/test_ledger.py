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


def test_plot_gate4_groups_by_design_and_hatches_buckets(tmp_path: Path) -> None:
    d, out = tmp_path / "ledger", tmp_path / "plots"
    common = ["--dir", str(d), "--lane", plots.GATE4_LANE, "--track", "shared", "--scope", "vu", "--security", "2^-128", "--B", "4096"]
    assert ledger.main(["add", *common, "--label", "A e2e non-ZK", "--proof-class", "NON_ZK_PROOF_DIAGNOSTIC",
                        "--hardware", "NVIDIA H100 80GB HBM3", "--seconds-per-vu", "1e-3"]) == 0
    assert ledger.main(["add", *common, "--label", "A buckets v1", "--proof-class", "ARITHMETIC_DIAGNOSTIC",
                        "--hardware", "NVIDIA H100 80GB HBM3", "--overhead", "7e6", "--omitted", "encoding"]) == 0
    assert ledger.main(["add", *common, "--label", "B e2e HVZK", "--proof-class", "COMPLETE_HVZK_BACKEND", "--assumptions", "hash",
                        "--hardware", "NVIDIA H100 80GB HBM3", "--seconds-per-vu", "1.5e-3"]) == 0
    assert ledger.main(["add", *common, "--label", "SP1 sound core, 1 VU", "--proof-class", "NON_ZK_PROOF",
                        "--hardware", "RTX 4090 (cited)", "--seconds-per-vu", "3.58"]) == 0
    entries = ledger.load(d / f"{plots.GATE4_LANE}.jsonl")
    assert [plots._design(e) for e in entries] == ["A", "A", "B", "SP1"]
    out.mkdir()
    p = plots.plot_gate4(entries, out)
    assert p.stat().st_size > 1000
    # main() draws it only when the gate-4 lane is present
    assert plots.main(["--dir", str(d), "--out", str(out)]) == 0 and (out / "gate4_h100.png").exists()
