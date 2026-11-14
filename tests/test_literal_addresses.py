"""No tracked file names a machine or an account by its address or service id: a public IPv4 address, a cloud provider's
resource id or endpoint, or a Slack, Notion or Linear id or link.  Code reads them at run time from the configuration or
secret that holds them, and a test uses a documentation address (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24) or an id that
says it is fake (`example`, `fake`, `dummy`, or a one- or two-letter name).

``ALLOWED`` bounds what each file still holds.  A file over its count fails; one under it passes, so a change that removes a
site needs no edit here and doesn't conflict with one in flight that removes another.  A count is raised only for the team's
own configuration that holds the id (the first group), and lowered by whoever recounts.  The rule is `research.addresses`'s,
which leaves out ``uv.lock``, its dotted numbers being package versions, and this file, whose toy files name each form.
"""
from __future__ import annotations

import test_repository
from research import addresses
from research.addresses import QUAD as _QUAD
from research.addresses import quads, reader, scan

from test_repository import ROOT, _tracked

#: a tracked file's text (None when it is deleted in the working tree), and any dotted quad: for tests that read one file
_read = reader(ROOT)

#: Each file that still names an address or a service id, with at most how many it names.
ALLOWED: dict[str, int] = {
    # the team's own tooling and machines
    ".agents/skills/team/SKILL.md": 1,
    ".agents/skills/using-slack/SKILL.md": 1,
    ".agents/skills/using-slack/registry.json": 36,
    ".agents/skills/using-slack/slack-app.json": 2,
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
    "benchmarks/network_traces/tests/fixtures/certifier-period-answers.json": 2,
    "benchmarks/numerical/tests/bench/data/flock-cells/6d1295ed.json": 3,
    "benchmarks/numerical/tests/bench/data/flock-cells/d1961ba4.json": 3,
    "benchmarks/numerical/tests/bench/data/flock-runs/r20260925-233818-a37e/job.json": 1,
    "benchmarks/numerical/tests/bench/data/flock-runs/r20260925-233831-a064/job.json": 7,
    "benchmarks/numerical/tests/bench/data/flock-runs/r20260925-233831-a064/result.json": 3,
    "benchmarks/numerical/tests/bench/test_placement.py": 4,
    "benchmarks/pouw/tests/fixtures/challenger-key.json": 1,
    "benchmarks/tc_probe_fp4/tests/fixtures/tc-sm100-f8-2026-10-19.json": 1,
    "benchmarks/one_stage/interfaces/tests/fixtures/block-record-b71a-w30.json": 7,
    "benchmarks/one_stage/interfaces/tests/fixtures/block-record-r1-window10.json": 4,
    "benchmarks/one_stage/interfaces/tests/fixtures/block-record-r2-period0.json": 8,
    "benchmarks/one_stage/interfaces/tests/test_report.py": 1,
    "verity/circuits/ops/tests/fixtures/tc-sm100-f8-2026-10-19/records.json": 1,
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


def test_no_new_literal_addresses_or_service_ids():
    found = scan(_tracked(), reader(ROOT))
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
        "g.sh": "pkg-81.2.3.4\nhost=${H:-81.2.3.4}\n",
        "uv.lock": 'version = "81.2.3.4"\n',
    }
    assert scan(list(docs), docs.__getitem__) == {
        "a.sh": [("ipv4", 1)],
        "c.json": [("provider id", 1), ("endpoint", 1)],
        "d.py": [("slack id", 1), ("notion link", 2), ("linear link", 3)],
        "f.txt": [("ipv4", 1)],
        "g.sh": [("ipv4", 2)],
    }


def test_the_rules_exceptions_are_these():
    """What `research.addresses` never reads or never counts, which `benchmarks/security_findings/` counts the tree by: a
    file, an extension or a resolver added there drops what it holds from that count without the file changing, so it
    changes this test too. The tree keeps one list of binary extensions, the rule's."""
    assert addresses.SKIP == {"uv.lock", "tests/test_literal_addresses.py"}
    assert addresses.BINARY == {".bin", ".npy", ".npz", ".lgto", ".neg", ".proof", ".gz", ".png", ".pdf", ".olean", ".ilean",
                                ".tar", ".zip"}
    assert addresses.RESOLVERS == {"1.1.1.1", "1.0.0.1", "8.8.8.8", "8.8.4.4", "9.9.9.9"}
    assert [n for n, v in vars(test_repository).items() if isinstance(v, frozenset | set) and ".olean" in v] == []


def test_read_and_quad_serve_a_test_that_reads_one_file():
    assert _QUAD.search("at 192.0.2.7") and _read("tests/test_literal_addresses.py") and _read("no/such/file") is None


def test_an_address_after_a_hyphen_is_read_unless_a_word_comes_before_the_hyphen():
    """A shell default (`${H:-ADDRESS}`), a flag's value and a range's end are addresses; `pkg-13.0.3.0` is a version."""
    text = "${H:-192.0.2.7} -192.0.2.8 =-192.0.2.9 x-192.0.2.10 pkg-192.0.2.11 v-192.0.2.12"
    assert [q for _, q in quads(text)] == ["192.0.2.7", "192.0.2.8", "192.0.2.9"]
