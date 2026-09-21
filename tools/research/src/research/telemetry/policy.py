"""policy.toml -> RULES.md (brief §2: RULES.md describes or renders the one tested configuration; it is not a second editable source).

    tele render-rules POLICY.toml [--template T] > RULES.md
    tele check-rules  POLICY.toml RULES.md [--template T]      # exit 1 with a unified diff when RULES.md is not the render

Template placeholders: {{section.key}}, {{list:section.key}} (", "-joined), {{codelist:...}} (`a`, `b`), {{slashlist:...}} (a / b),
{{oxford_or:...}} ("an OOM or a numerical mismatch"), {{count_word:...}} (four), {{section.key_md}} (derived prose, see DERIVED).
`policy_values(policy)` is what the runner records (job.json execution.policy_version / policy_sha256 + the resolved defaults).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = HERE / "policy" / "RULES.template.md"
PLACEHOLDER = re.compile(r"\{\{([a-z_]+:)?([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\}\}")
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def DERIVED(pol: dict[str, Any], section: str, key: str) -> str | None:
    """Prose derived from a value (so the toml stays a value, not a sentence)."""
    if (section, key) == ("checker", "record_checker_route_md"):
        return pol["checker"]["record_checker_route"].replace("/vol", "`/vol`")
    if (section, key) == ("checker", "standalone_batch_decomp_supported_md"):
        return "a supported route" if pol["checker"]["standalone_batch_decomp_supported"] else "not a supported route"
    if (section, key) == ("admission", "host_heavy_per_cgroup_word"):
        return WORDS.get(pol["admission"]["host_heavy_per_cgroup"], str(pol["admission"]["host_heavy_per_cgroup"])).capitalize()
    return None


def _article(x: str) -> str:
    return ("an " if x[:1].lower() in "aeiou" else "a ") + x


def render(policy_path: str | Path, template_path: str | Path = DEFAULT_TEMPLATE) -> str:
    pol = tomllib.loads(Path(policy_path).read_text())
    tpl = Path(template_path).read_text()

    def sub(m: re.Match) -> str:
        fn, sec, key = (m.group(1) or "").rstrip(":"), m.group(2), m.group(3)
        d = DERIVED(pol, sec, key)
        if d is not None:
            return d
        try:
            v = pol[sec][key]
        except KeyError:
            raise KeyError(f"policy has no [{sec}] {key} (placeholder {m.group(0)})") from None
        if fn == "list":
            return ", ".join(str(x) for x in v)
        if fn == "codelist":
            return ", ".join(f"`{x}`" for x in v)
        if fn == "slashlist":
            return " / ".join(str(x) for x in v)
        if fn == "oxford_or":
            items = [_article(str(x)) for x in v]
            return " or ".join(items) if len(items) <= 2 else ", ".join(items[:-1]) + ", or " + items[-1]
        if fn == "count_word":
            return WORDS.get(len(v), str(len(v)))
        if fn:
            raise KeyError(f"unknown placeholder function {fn!r}")
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)

    return PLACEHOLDER.sub(sub, tpl)


def policy_values(policy_path: str | Path) -> dict[str, Any]:
    p = Path(policy_path)
    pol = tomllib.loads(p.read_text())
    return {"policy_version": pol.get("policy", {}).get("version"), "policy_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "policy_path": str(p),
            "defaults": {"MATCH_IMPL": pol.get("checker", {}).get("MATCH_IMPL"), "MATCH_PIPELINE": pol.get("checker", {}).get("MATCH_PIPELINE"),
                         "GPU_GATE": pol.get("admission", {}).get("GPU_GATE"), "disk_reserve_gb": pol.get("admission", {}).get("disk_reserve_gb"),
                         "need_var": pol.get("admission", {}).get("need_var")}}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="tele render-rules|check-rules")
    ap.add_argument("cmd", choices=["render-rules", "check-rules"])
    ap.add_argument("policy")
    ap.add_argument("rules", nargs="?", default=None)
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    a = ap.parse_args(argv)
    out = render(a.policy, a.template)
    if a.cmd == "render-rules":
        sys.stdout.write(out)
        return 0
    if not a.rules:
        ap.error("check-rules needs RULES.md")
    cur = Path(a.rules).read_text()
    if cur == out:
        print(f"{a.rules} is the render of {a.policy}")
        return 0
    sys.stdout.writelines(difflib.unified_diff(out.splitlines(True), cur.splitlines(True), fromfile="render(policy)", tofile=a.rules))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
