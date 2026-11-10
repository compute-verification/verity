"""regen_hand_edits.py IN OUT: the hand edits `regen.py` leaves in the inference-only policy's and the slowdown
calculator's maps, for the layout's rows `layout_10_inference_only_map.toml` holds (each of `benchmarks/rho/`,
`benchmarks/nci/` and the policy's check keeps its directory's name under `verity/protocols/inference_only/`).

IN is `regen.py`'s output directory; OUT gets the maps as they run on the moved tree. The calculator's directory keeps
the layout's destination, so its move map and the rows that follow that move are dropped. Each edit must hit exactly
the count given, so a map that changed under it fails here rather than silently.
"""
import shutil
import sys
import tomllib
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    shutil.rmtree(out)
shutil.copytree(src, out)
(out / "regen-report.json").unlink(missing_ok=True)

EDITS = {
    "inference_only_map.toml": [
        # the check's directory after the layout: verity/protocols/inference_only/verifier/nci/
        ('"network/nci" = "network/inference_only"', '"verifier/nci" = "verifier/inference_only"', 1),
        # benchmarks/nci's path, kept where it is and under its name
        (r"'\bbenchmarks/nci\b'", r"'\binference_only/nci\b'", 1),
    ],
    "inference_only_ids_map.toml": [
        ('files = ["verity/core/protocols/network/inference_only/lean/lean-audit.json"]',
         'files = ["verity/protocols/inference_only/verifier/inference_only/lean/lean-audit.json"]', 1),
    ],
    "slowdown_calculator_names_map.toml": [
        # the command, the outputs, the two joins and the index's row followed the directory's rename, which is dropped
        ('[[words_in]]\nfiles = ["benchmarks/slowdown/**", "verity/composite/verifier/remote/README.md"]\n'
         'words = { "verity bench rho" = "verity bench slowdown", "rho.md" = "slowdown.md", "rho.json" = "slowdown.json" }\n\n', "", 1),
        ('[[words_in]]\nfiles = ["benchmarks/circuit_properties/run.py", "verity/protocols/inference_only/nci/statement.py"]\n'
         "words = { 'HERE.parent / \"rho\"' = 'HERE.parent / \"slowdown\"' }\n\n", "", 1),
        ('[[words_in]]\nfiles = ["benchmarks/README.md"]\nwords = { "`rho/`" = "`slowdown/`" }\n\n', "", 1),
        # "rho's": the files that say it, where the layout puts them
        ('files = ["benchmarks/**"]',
         'files = ["benchmarks/**", "verity/spot/spot_figures/**", "verity/spot/sampling_plan/**", '
         '"verity/protocols/inference_only/nci/**"]', 1),
        # a path that ends in the directory and then 's ("verity/protocols/inference_only/rho's shapes") is a path
        ("'lanes/[\\w.-]+']", "'lanes/[\\w.-]+', \"(?<=/)rho's\"]", 1),
    ],
}
#: a directory a comment names, where the layout puts it
COMMENT_DIRS = [
    ("verity/core/protocols/network/nci", "verity/protocols/inference_only/verifier/nci"),
    ("verity/core/protocols/network/tests", "verity/protocols/certifier/verifier/network/tests"),
    ("verity/core/protocols/remote", "verity/composite/verifier/remote"),
    ("verity/core/protocols/composition", "verity/composite/verifier/composition"),
    ("benchmarks/nci", "verity/protocols/inference_only/nci"),
    ("benchmarks/rho", "verity/protocols/inference_only/rho"),
    ("benchmarks/spot_figures", "verity/spot/spot_figures"),
    ("examples/lifecycle", "workloads/lifecycle"),
    ("Security/", "security/"),
]

(out / "slowdown_calculator_map.toml").unlink()
for name, edits in EDITS.items():
    p = out / name
    text = p.read_text()
    for old, new, n in edits:
        assert text.count(old) == n, f"{name}: {old!r} occurs {text.count(old)} times, not {n}"
        text = text.replace(old, new)
    p.write_text(text)
changed = 0
for p in sorted(out.glob("*.toml")):
    lines = p.read_text().split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            new = line
            for old_dir, new_dir in COMMENT_DIRS:
                new = new.replace(old_dir, new_dir)
            if new != line:
                lines[i], changed = new, changed + 1
    p.write_text("\n".join(lines))
    tomllib.loads(p.read_text())
print(f"{len(list(out.glob('*.toml')))} maps written to {out}; {changed} comment lines' directories rewritten")
