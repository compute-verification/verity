"""The bench's tables as Markdown, read from the code and data they describe (`run.py tables`): the registered steps,
the arms in specs/, the window lists, the models and the locked baseline. The README holds none of them."""
from __future__ import annotations

import json
from pathlib import Path

import methodology
import models
import report
import spec

BASELINE = Path(__file__).resolve().parent / "baseline.json"


def _cell(x) -> str:
    s = "-" if x is None or x == "" else str(x)
    return " ".join(s.split()).replace("|", "\\|")


def _table(head: list[str], body: list[list], right: tuple[int, ...] = ()) -> str:
    rule = "|" + "".join("---:|" if i in right else "---|" for i in range(len(head)))
    body = ["| " + " | ".join(_cell(x) for x in r) + " |" for r in body]
    return "\n".join(["| " + " | ".join(head) + " |", rule, *body])


def steps() -> str:
    body = []
    for k, v in spec.catalogue().items():
        on = "/".join(v["on"]) if k.startswith("attention.") else ""
        body.append([f"`{k}({', '.join(v['params'])})`", v["phase"], on, v["doc"].split("\n\n")[0], v["source"]])
    return _table(["step", "phase", "on", "what it does", "from"], body)


def arms() -> str:
    body = []
    for p in sorted(spec.SPECS.glob("*.toml")):
        a = spec.load(str(p))
        body.append([f"`{a.name}`", a.scope, "seeded" if a.seeded else "deterministic", "yes" if a.needs_calib else "",
                     a.plain and f"`{a.plain}`", a.description])
    return _table(["arm", "scope", "seeds", "calibration", "reported against", "description"], body)


def windows() -> str:
    body = [[ctx, f"0–{n - 1}", f"{methodology.scored_tokens(ctx):,}"] for ctx, n in methodology.CONTEXTS.items()]
    return _table(["context", "windows", "scored tokens (per arm and seed)"], body, right=(0, 1, 2))


def models_() -> str:
    body = [[f"`{k}`", f"`{m['repo']}` @ `{m['revision'][:8]}`", m["tokenizer"], m["gpus"]]
            for k, m in models.MODELS.items()]
    return _table(["model", "snapshot", "tokenizer", "GPUs"], body, right=(3,))


def baseline() -> str:
    return report.baseline_table(json.loads(BASELINE.read_text()))


SECTIONS = {"steps": ("The registered steps", steps), "arms": ("The arms in specs/", arms),
            "windows": ("The window lists", windows), "models": ("The models", models_),
            "baseline": ("The locked baseline", baseline)}


def render(sections: list[str] | None = None) -> str:
    if bad := [s for s in sections or () if s not in SECTIONS]:
        raise SystemExit(f"tables: no section {', '.join(bad)}; the sections are {', '.join(SECTIONS)}")
    out = ["# `verity bench ppl tables`"]
    for key in sections or SECTIONS:
        title, f = SECTIONS[key]
        out.append(f"## {title}\n\n{f()}")
    return "\n\n".join(out) + "\n"
