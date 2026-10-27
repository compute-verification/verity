"""A v2 window's withheld layout sized from its sessions' schedules (`v2_layout`), held to v2's own reading of a session's
uses (`proofs.session_uses`, Lean `OneStageZk.sessionUses`): family for family, each family's count is the coins its uses
read, so an in-process challenger of the layout releases every use once and its log shows exactly the uses
(`coins.exactly_reason`).

The fast tests build a session record to a schedule. The slow one reads the pinned `--zk --registered` session
`verity/core/protocols/sampled_proofs/tests/test_proofs.py` verifies (a RoPE d64 statement of 32 units, proved under a
subset draw of 16), whose schedule `flock-firewall` prints from its circuit and public files: what no record built here
can show is that C-Flock's own schedule is the one its records' uses read."""
from __future__ import annotations

import fcntl
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs import challenger as Ch, coins as CC
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity.core.protocols.sampled_proofs.one_stage.verifier import BUILD_LOCK

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5_coins as AC  # noqa: E402
import v2_layout as V2L  # noqa: E402

#: `test_proofs.py`'s pinned session, fetched from the evidence store as it fetches it
FIXTURE = "art:88544efa5af6f50ab24ad04a8e8d3118075e9f62898e41bbfb45ec3f3bc3d85b"
VERIFIER = AC.TREE / "verity" / "ml" / "flock" / "verifier"

#: a schedule of one table at m_pts = 9: its two reps' rounds, four of rep 0's one F128 each
LINKS = 2 * 9
PLAN = [("circuit/rep0", 6), ("circuit/rep0", 12), *[("circuit/rep0", 1)] * 4,
        ("circuit/rep1", 6), ("circuit/rep1", 7), ("circuit/rep1", 512)]


def record(links: int, plan: list[tuple[str, int]]) -> str:
    """A session record of the schedule: `links` link points, one table's root_B digest, and each stream's rounds in
    `plan`'s order, a round's coins its F128s."""
    streams: dict[str, list] = {}
    for k, (s, n) in enumerate(plan):
        streams.setdefault(s, []).append({"msg_sha512": f"{k:0128x}", "coins": [f"{k:032x}"] * n})
    return json.dumps({"link": {"points": [[f"{j:032x}" for j in range(links)]], "root_b_sha512": {"circuit": "aa" * 64}},
                       "streams": [{"stream": s, "rounds": r} for s, r in streams.items()]})


def families_of(uses: list) -> list[list]:
    """The coins each family's uses read, in the order the uses first name it."""
    out: dict[str, int] = {}
    for label, _after, n, _want in uses:
        out[label] = out.get(label, 0) + n
    return [[label, n] for label, n in out.items() if n]


def released(layout: list, uses: list) -> tuple[Ch.LiveCommitment, dict[str, list[bytes]]]:
    """Each use's coins opened once, in order, on an in-process challenger of `layout`: its commitment, and the roots each
    family's coins were released after."""
    live = Ch.LiveChallenger(AC.SCHEME, 0, layout, random.Random(1).randbytes)
    roots: dict[str, list[bytes]] = {}
    for label, after, n, _want in uses:
        for _ in range(n):
            live.open(label, len(roots.get(label, ())), after)
            roots.setdefault(label, []).append(after)
    return live.commitment, roots


def exactly(layout: list, uses: list) -> str | None:
    commitment, roots = released(layout, uses)
    return CC.exactly_reason(commitment, roots, [CC.Use(label, after, n) for label, after, n, _ in uses])


def test_a_slots_families_are_the_coins_its_records_uses_read():
    """Each family holds the sum of its rounds' ceil(n/4): rep 0's four rounds of one F128 read four coins, not one."""
    uses = PF.session_uses(3, record(LINKS, PLAN))
    got = V2L.slot_families(3, (LINKS, PLAN))
    assert got == families_of(uses) == [["slot/3/session", 5], ["slot/3/circuit/rep0", 2 + 3 + 4],
                                        ["slot/3/circuit/rep1", 2 + 2 + 128]]


def test_the_window_layout_releases_each_use_once_and_one_coin_short_refuses_the_last():
    uses = PF.session_uses(0, record(LINKS, PLAN))
    layout = V2L.window_layout(AC.SCHEME, AC.families({0: 4}), V2L.slot_families(0, (LINKS, PLAN)))
    assert layout[0] == [AC.FAMILY, AC.COINS_PER_RANK + 1] and exactly(layout, uses) is None
    short = [[label, n - (label == "slot/0/circuit/rep1")] for label, n in layout]
    with pytest.raises(Ch.CoinError, match="no slot/0/circuit/rep1 coin 131"):
        released(short, uses)


def test_a_slot_over_several_schedules_holds_each_familys_largest():
    got = V2L.slot_families(0, (8, [("a/rep0", 4)]), (18, [("a/rep0", 1), ("a/rep0", 1)]), (4, [("b/rep0", 9)]))
    assert got == [["slot/0/session", 5], ["slot/0/a/rep0", 2], ["slot/0/b/rep0", 3]]
    assert V2L.slot_families(0, (0, [("a/rep0", 1)])) == [["slot/0/a/rep0", 1]]


def test_what_is_no_layout_is_refused():
    with pytest.raises(ValueError, match="the link points' family"):
        V2L.slot_families(0, (4, [("session", 1)]))
    with pytest.raises(ValueError, match="at least one schedule"):
        V2L.slot_families(0)
    one = V2L.slot_families(0, (LINKS, PLAN))
    with pytest.raises(ValueError, match="labels are distinct"):
        V2L.window_layout(AC.SCHEME, AC.families({0: 4}), one, one)
    with pytest.raises(ValueError, match=f"at most {Ch.MAX_COINS} coins"):
        V2L.window_layout(AC.SCHEME, AC.families({0: 4}), V2L.slot_families(0, (4 * Ch.MAX_COINS, [])))


@pytest.fixture(scope="module")
def fx(tmp_path_factory) -> Path:
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        root = ci.fetch_sessions(FIXTURE, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(["lake", "build", "flock-firewall"], cwd=V2L.FIREWALL.parents[3], check=True, capture_output=True)
    return root


def _flags(fx: Path) -> list[str]:
    st = fx / "stage"
    return ["--statement", PF.STATEMENT, "--circuit", str(st / "circuit.txt"), "--public", str(st / "pub-32.bin"),
            "--partition", str(st / "partition.json"), "--program", str(st / "program.json")]


@pytest.mark.slow
def test_the_pinned_sessions_schedule_is_its_uses_family_for_family(fx):
    """At the session's own draw, its statement's schedule gives each family exactly the coins v2 reads of its record:
    ``slot/0/session`` and the table's two reps, ``slot/0/circuit/rep0`` and ``rep1``. The population's schedule (no
    draw, all 32 units) holds each family at least as large. An in-process challenger of the window's layout releases
    every use once and shows exactly the uses. Slow for three m = 25 setups and the firewall's build."""
    text = (fx / "session" / "session.json").read_text()
    uses = PF.session_uses(0, text)
    slot = V2L.slot_families(0, V2L.schedule(_flags(fx), json.loads(text)["unit_draw"]))
    assert slot == families_of(uses), slot
    assert [label for label, _ in slot] == ["slot/0/session", "slot/0/circuit/rep0", "slot/0/circuit/rep1"]
    whole = dict(V2L.slot_families(0, V2L.schedule(_flags(fx))))
    assert all(n <= whole[label] for label, n in slot), (slot, whole)
    assert exactly(V2L.window_layout(AC.SCHEME, AC.families({0: 4}), slot), uses) is None
