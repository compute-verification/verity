"""No tracked file names a machine or an account by its address or service id: a public IPv4 address, a cloud provider's
resource id or endpoint, or a Slack, Notion or Linear id or link.  Code reads them at run time from the configuration or
secret that holds them, and a test uses a documentation address (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24) or an id that
says it is fake (`example`, `fake`, `dummy`, or a one- or two-letter name).

``ALLOWED`` bounds what each file still holds.  A file over its count fails; one under it passes, so a change that removes a
site needs no edit here and doesn't conflict with one in flight that removes another.  A count is raised only for the team's
own configuration that holds the id (the first group), and lowered by whoever recounts.  ``uv.lock`` is left out, its dotted
numbers being package versions, and so is this file, whose toy files name each form.  Each pattern starts with a literal, so
the whole tree scans in seconds.
"""
from __future__ import annotations

import ipaddress
import os
import re

from test_repository import _BINARY, ROOT, _tracked

#: Public resolvers a script routes toward to find its own interface; they are nobody's machine.
RESOLVERS = {"1.1.1.1", "1.0.0.1", "8.8.8.8", "8.8.4.4", "9.9.9.9"}

_DIGITS = "0123456789"
_WORD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
#: the middle of a dotted quad (`.2.3.4` of `1.2.3.4`), from which `_quads` widens to the whole
_QUAD_MIDDLE = re.compile(r"\.\d{1,3}\.\d{1,3}\.\d")
_QUAD = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
#: A dotted quad that is a version or a section number, not an address: `PTX ISA 9.7.14.5`, `§2.1.3.2`, `section 1.4.1.1`,
#: `==12.9.1.4`.
_VERSION_BEFORE = re.compile(
    r"(?:§|-|[=<>~!]=\s*|\b(?:ISA|PTX|CUDA|cuBLAS|cuDNN|NCCL|version|v|section)\s*[=:]?\s*[\"']?)$", re.I)

_NEBIUS = ("computeinstance", "computedisk", "computeimage", "computefilesystem", "computegpucluster", "project", "tenant",
           "serviceaccount", "vpcnetwork", "vpcsubnet", "vpcallocation", "vpcsecuritygroup", "publickey", "accesskey",
           "storagebucket", "mk8scluster", "mk8snodegroup", "registry", "iamgroup")
_STOP = r"[^\s)\]\"'`>]"

#: kind: [(a substring the file holds, the pattern then tried)]; a pattern that starts with a character class is tried only
#: near each match of its anchor (`_ANCHORS`)
KINDS = {
    "provider id": [("-", r"\b(?:" + "|".join(_NEBIUS) + r")-[eu]0\d[a-z0-9]{12,}"),
                    ("prj_", r"\bprj_[A-Za-z0-9]{20,}"), ("team_", r"\bteam_[A-Za-z0-9]{20,}")],
    "endpoint": [(".r2.cloudflarestorage.com", r"\b[0-9a-f]{32}\.r2\.cloudflarestorage\.com"),
                 (".r2.dev", r"\bpub-[0-9a-f]{32}\.r2\.dev"),
                 (".workers.dev", r"[a-z0-9-]+\.workers\.dev"), (".pages.dev", r"[a-z0-9-]+\.pages\.dev"),
                 (".vercel.app", r"[a-z0-9-]+\.vercel\.app"), (".proxy.runpod.net", r"[a-z0-9-]+\.proxy\.runpod\.net"),
                 ("@ssh.runpod.io", r"[a-z0-9-]+@ssh\.runpod\.io"),
                 (".iam.gserviceaccount.com", r"[a-z0-9-]+\.iam\.gserviceaccount\.com"), ("arn:aws:", r"arn:aws:[a-z0-9-]+:"),
                 (".neon.tech", r"\bep-[a-z0-9-]+\.[a-z0-9-]+\.aws\.neon\.tech"),
                 (".amazonaws.com", r"[a-z0-9.-]+\.s3[.-](?:[a-z0-9-]+\.)?amazonaws\.com")],
    "slack id": [("0", r"(?<![A-Za-z0-9])[CDGSTUWB]0[A-Z0-9]{7,10}(?![A-Za-z0-9])"),
                 (".slack.com", r"\b(?!(?:api|app|hooks|files|status|www|edgeapi|docs)\.)[a-z0-9-]{3,}\.slack\.com\b")],
    "notion link": [("notion.", rf"(?<![\w.])(?:www\.|app\.)?notion\.(?:so|site|com)/(?!v1/){_STOP}{{3,}}")],
    "linear link": [("linear.app/", rf"\blinear\.app/(?!oauth/)[a-z0-9-]{{3,}}/{_STOP}*")],
}
_ANCHORS = {"slack id": re.compile(r"0[A-Z0-9]{7,10}"), "provider id": re.compile(r"-[eu]0\d")}
_KINDS = [(kind, hint, re.compile(rx), _ANCHORS[kind] if len(hint) == 1 else None)
          for kind, rules in KINDS.items() for hint, rx in rules]
_FAKE = re.compile(r"example|fake|dummy|acme", re.I)
_SKIP = {"uv.lock", "tests/test_literal_addresses.py"}

#: Each file that still names an address or a service id, with at most how many it names.
ALLOWED: dict[str, int] = {
    # the team's own tooling and machines
    ".agents/skills/team/SKILL.md": 1,
    ".agents/skills/using-slack/SKILL.md": 1,
    ".agents/skills/using-slack/registry.json": 35,
    ".agents/skills/using-slack/slack-app.json": 2,
    "infra/cluster/src/cluster/status.py": 2,
    "infra/cluster/tests/test_policy.py": 1,
    "infra/control/README.md": 2,
    "infra/control/cloud_watch.toml": 7,
    "infra/control/deadman/api/deadman.py": 2,
    "infra/control/vm_stop.toml": 1,
    "infra/control/watchers_ssh_config": 4,
    "infra/cursor/launch_coordinator.py": 1,
    "infra/ops/tests/test_cloud_watch.py": 3,
    "infra/ops/tests/test_deadman.py": 1,
    "infra/ops/tests/test_watchers.py": 3,
    "infra/slack-events/README.md": 2,
    "infra/slack-events/service/slack_events/consumer.py": 1,
    "infra/slack-events/tests/conftest.py": 2,
    "infra/slack-events/tests/test_consumer.py": 1,
    "tools/research/console/verity_console.py": 1,
    "tools/research/src/research/goals.py": 1,
    "tools/research/src/research/jobs/client.py": 1,
    "tools/research/src/research/notes_sync.py": 1,
    "tools/research/src/research/pods/nebius/common.sh": 1,
    "tools/research/src/research/pods/nebius/fleet.py": 2,
    "tools/research/src/research/pods/nebius/monitoring/alert_sink.py": 1,
    "tools/research/src/research/pods/nebius/monitoring/grafana.yaml": 1,
    "tools/research/src/research/pods/nebius/monitoring/grafana_up.sh": 1,
    "tools/research/src/research/pods/nebius/n1_replacement.sh": 6,
    "tools/research/src/research/pods/nebius/peers": 3,
    "tools/research/src/research/pods/nebius/peers-challenger": 16,
    "tools/research/src/research/pods/nebius/verifier/verifier.json": 1,
    "tools/research/src/research/slack.py": 4,
    "tools/research/src/research/store/fetchset.py": 1,
    "tools/research/tests/test_bucket_index.py": 2,
    "tools/research/tests/test_budgets.py": 1,
    "tools/research/tests/test_env_credentials.py": 2,
    "tools/research/tests/test_nebius_wake.py": 2,
    "tools/research/tests/test_pod_lease.py": 2,
    "tools/research/tests/test_remote_ship_git.py": 2,
    "tools/research/tests/test_retention.py": 2,
    "tools/research/tests/test_slack.py": 36,
    # addresses and instance ids that benchmarks, examples and their tests record or run against
    "verity/composite/device/live_challenger/publish.sh": 2,
    "benchmarks/network_traces/site.json": 1,
    "benchmarks/network_traces/tests/fixtures/certifier-period-answers.json": 2,
    "benchmarks/numerical/tests/bench/data/flock-cells/6d1295ed.json": 3,
    "benchmarks/numerical/tests/bench/data/flock-cells/d1961ba4.json": 3,
    "benchmarks/numerical/tests/bench/data/flock-runs/r20260925-233818-a37e/job.json": 1,
    "benchmarks/numerical/tests/bench/data/flock-runs/r20260925-233831-a064/job.json": 7,
    "benchmarks/numerical/tests/bench/data/flock-runs/r20260925-233831-a064/result.json": 3,
    "benchmarks/numerical/tests/bench/test_placement.py": 4,
    "benchmarks/one_stage/served_verdicts.sh": 2,
    "benchmarks/one_stage/tests/test_served_verdicts.py": 1,
    "benchmarks/pouw/tests/fixtures/challenger-key.json": 1,
    "benchmarks/remote_session/auditor.sh": 1,
    "benchmarks/remote_session/gate.sh": 2,
    "benchmarks/remote_session/gate_run.sh": 2,
    "benchmarks/tc_probe_fp4/tests/fixtures/tc-sm100-f8-2026-10-19.json": 1,
    "benchmarks/one_stage/interfaces/tests/fixtures/block-record-b71a-w30.json": 7,
    "benchmarks/one_stage/interfaces/tests/fixtures/block-record-r1-window10.json": 4,
    "benchmarks/one_stage/interfaces/tests/fixtures/block-record-r2-period0.json": 8,
    "benchmarks/one_stage/interfaces/tests/test_report.py": 1,
    "demos/vllm/placement.toml": 1,
    "adapters/vllm/tests/protocols/pouw_h1_rehearsal.py": 1,
    "tools/verity/check/tests/test_placement.py": 3,
    "verity/definitions/ops/tests/fixtures/tc-sm100-f8-2026-10-19/records.json": 1,
    # stand-ins that predate the documentation ranges
    "tools/verity/check/tests/test_verdicts.py": 2,
    # version numbers the scan reads as addresses
    "benchmarks/pouw/dc/dc.cuh": 1,
    "benchmarks/pouw/harness/README.md": 1,
    # links in the locks and data, and the Lean cache's route
    "security/Properties/lean-audit.json": 2,
    "security/lean-audit.json": 1,
    "benchmarks/census/adversary.json": 1,
    "tools/verity/lean/cache.py": 1,
}

FIX = ("Read the address or id at run time from the configuration or secret that holds it, never from a literal here; in a "
       "test, use a documentation address (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24) or an id that says it is fake. "
       "ALLOWED in tests/test_literal_addresses.py is raised only for the team's own configuration that holds the id.")


def _quads(text: str):
    """(start, the quad) of each dotted quad of ``text`` that reads as an address: not part of a longer dotted or hyphenated
    token, and not a version or section number."""
    seen = set()
    for m in _QUAD_MIDDLE.finditer(text):
        s = m.start()
        while s > 0 and m.start() - s < 3 and text[s - 1] in _DIGITS:
            s -= 1
        q = _QUAD.match(text, s)
        if s == m.start() or s in seen or not q:
            continue
        seen.add(s)
        before, after = text[s - 1:s], text[q.end():q.end() + 2]
        if (before and (before in _WORD or before in ".§-")) or (after[:1] and (after[0] in _WORD or after[0] == "-")) \
                or (after[:1] == "." and after[1:2] in set(_DIGITS)) or _VERSION_BEFORE.search(text[max(0, s - 16):s]):
            continue
        yield s, q.group()


def _public(quad: str) -> bool:
    try:
        return ipaddress.IPv4Address(quad).is_global and quad not in RESOLVERS
    except ValueError:
        return False


def findings(text: str) -> list[tuple[str, int]]:
    """(kind, line) of each address or service id ``text`` names, in order."""
    found = [("ipv4", s) for s, quad in _quads(text) if _public(quad)]
    seen = set()
    for kind, hint, rx, anchor in _KINDS:
        if hint not in text:
            continue
        spans = [(max(0, a.start() - 20), a.start() + 20) for a in anchor.finditer(text)] if anchor else [(0, len(text))]
        for at, end in spans:
            for m in rx.finditer(text, at, end):
                if not _FAKE.search(m.group()) and (kind, m.start()) not in seen:
                    seen.add((kind, m.start()))
                    found.append((kind, m.start()))
    return [(kind, text.count("\n", 0, at) + 1) for kind, at in sorted(found, key=lambda x: x[1])]


def _read(f: str) -> str | None:
    """The text of tracked file ``f``, or None when it is deleted in the working tree or a submodule's directory."""
    path = ROOT / f
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None


def scan(files: list[str], read=_read) -> dict[str, list]:
    """{file: its findings} for each scanned file that has some."""
    out = {}
    for f in files:
        if f in _SKIP or os.path.splitext(f)[1] in _BINARY:
            continue
        if (text := read(f)) and (found := findings(text)):
            out[f] = found
    return out


def test_no_new_literal_addresses_or_service_ids():
    found = scan(_tracked())
    over = [f"  {f}:{line}: {kind}" for f, hits in sorted(found.items()) if len(hits) > ALLOWED.get(f, 0)
            for kind, line in hits]
    assert not over, "an address or a service id is written in the tree:\n" + "\n".join(over) + "\n" + FIX


def test_the_scan_reads_each_form():
    """The rules above, on toy files."""
    docs = {
        "a.sh": "ssh research@81.2.3.4 -p 22\nip route get 1.1.1.1\nlisten 127.0.0.1:80 and 10.0.0.2 and 192.0.2.7\n",
        "b.md": "PTX ISA 9.7.14.5, RFC 9162 §2.1.3.2, Tuning Guide section 81.4.1.1, cublas==12.9.1.4, pkg-13.0.3.0-py3, "
                "1.2.3.4.5, 999.1.1.1\n",
        "c.json": '{"vm": "computeinstance-e00abcdefghijklmnop", "url": "https://abc.vercel.app/x"}\n',
        "d.py": 'CH = "C0ABCDEFGHI"\nURL = "https://app.notion.com/p/abc"\nI = "https://linear.app/team-x/issue/X-1"\n',
        "e.txt": "C0FFEE; c0abcdefghi; XC0ABCDEFGHI; ep-x.neon.example; aaaa.r2.cloudflarestorage.com; api.notion.com/v1/x; "
                 "linear.app/x/issue/A-1; linear.app/oauth/token; api.slack.com; w.slack.com; fake-vm.vercel.app\n",
        "f.txt": "at 81.2.3.4.\n",
        "uv.lock": 'version = "81.2.3.4"\n',
    }
    assert scan(list(docs), read=docs.__getitem__) == {
        "a.sh": [("ipv4", 1)],
        "c.json": [("provider id", 1), ("endpoint", 1)],
        "d.py": [("slack id", 1), ("notion link", 2), ("linear link", 3)],
        "f.txt": [("ipv4", 1)],
    }
