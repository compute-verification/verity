"""The rules `run.sh` states and the partition files an audit reads, held to what a stream takes, without running the
script:

* the rule the script's weights tie arm states to its records audit is the rate of every unit of the tie, which names
  the tie's own design, work over every one of its units;
* that arm takes a tie's pins as given or refused only on `weights-tie pins`' own answer, never on a traceback;
* each partition file (`partitions`: this benchmark's, the one-stage audit's and each one the script names) states a
  rate a stream takes or is listed as still stating a fixed design (`FIXED`), which an audit on a challenger's coins
  (`a5.py --challenger`) refuses at the rule check and a served window's audit from its records
  (`records_audit.py audit`) refuses by name. The list only shrinks."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from verity.commitments.periods.population import population_program
from verity.protocols.proof.spot.verifier import service as S
from verity_vllm.pipeline import serving_rows as SRW

BENCH = ((Path(__file__).resolve().parents[3] / "benchmarks" / "prove_path"))
REPO = BENCH.parents[1]
RUN_SH = REPO / "benchmarks" / "prove_path" / "run.sh"
#: where the partition files are kept: beside this benchmark's drivers and beside the one-stage audit's
KEPT = (BENCH, REPO / "benchmarks" / "one_stage")
#: a partition file's key for the rule it states
KEY = "law"
#: a window of one template's stratum: its template and its proof blocks
STANDIN, UNITS = "SiluMul_v2{I=256}", 8
RATE = "a rate (rate:NUM/DEN, stratified-rate:NUM/DEN or work-rate:NUM/DEN)"
#: the partition files that still state a fixed design (``subset:K``, ``stratified:K`` or ``work:K``), by path from the
#: root. A stream refuses such a text as its rule, so an audit on a challenger's coins refuses the file, and so does a
#: served window's audit from its records. Each one's rate is its K over the proof blocks its scope registers, a count
#: only its window gives. The list only shrinks: a file leaves it in the change that makes its text a rate, and none
#: joins it.
FIXED = frozenset({
    "benchmarks/one_stage/glm53_all_families_tp8_step16.json",
    "benchmarks/one_stage/glm53_prefill_tp8_step0.json",
    "benchmarks/one_stage/glm53_prefill_tp8_step0_core.json",
    "benchmarks/one_stage/glm53_sparse_mla_tp8_step16.json",
    "benchmarks/one_stage/llama32_1b_rope_d64_l0.json",
    "benchmarks/prove_path/llama32_1b_silu_mul_v2_i8192_l0_step0.json",
    "benchmarks/prove_path/llama32_1b_tg_l0_step24.json",
})


def rule_of(text: object) -> tuple[S.Selection | None, str | None]:
    """``text`` as a stream's rule, as `a5.stream_of` makes it of what the audit states, or why a stream refuses it: a
    text that is no rate, or a Bernoulli design, which `Stream.register` refuses. -> (the rule, None) or (None, why)."""
    try:
        rule = S.SamplingDesign(text)
    except ValueError as e:
        return None, str(e)
    return (None, S.BERNOULLI) if rule.text.startswith("bernoulli") else (rule, None)


def partitions() -> dict[str, object]:
    """The rule each partition file states, by the file's path from the root: the JSON objects with ``templates`` and a
    rule among the files kept beside the drivers (`KEPT`) and each file the script names."""
    named = (REPO / p for p in re.findall(r"\bbenchmarks/[\w./-]+\.json\b", RUN_SH.read_text()))
    out = {}
    for path in sorted({*(p for d in KEPT for p in d.glob("*.json")), *(p for p in named if p.is_file())}):
        doc = json.loads(path.read_text())
        if isinstance(doc, dict) and {"templates", KEY} <= set(doc):
            out[path.relative_to(REPO).as_posix()] = doc[KEY]
    return out


def test_the_weights_ties_arm_states_the_rate_of_every_unit_of_the_tie():
    """The weights tie's arm (``PP_WEIGHTS_TIE=1``) hands `records_audit.py audit` its stream's rule as text. The tie
    registers its own design, work over every one of its N units, and a stream refuses that count as its rule, so the arm
    states the rate that names it, ``work-rate:N/N`` with N the units its verifier's tie names: a rate a stream takes,
    whose sampling design over N proof blocks is work with K = N, every unit sampled, at any N. A rate of fewer would name
    another design than the tie's window registers, and leave units unproved."""
    arm = RUN_SH.read_text().split('if [ -n "${PP_WEIGHTS_TIE:-}" ]; then', 1)[1].split("\nfi\n", 1)[0]
    rules = re.findall(r"records_audit\.py audit [^\n]*--sampling-design (\S+)", arm)
    assert rules == ["work-rate:$units/$units"], (
        f"{RUN_SH.name}'s weights tie arm no longer states the rate of every unit of the tie to its records audit "
        f"(`--{KEY} work-rate:$units/$units`): found {rules}; a fixed count is refused by name, and a rate of fewer units "
        "is not the tie's design")
    assert rule_of("work:$units".replace("$units", "613"))[1], "a stream takes a fixed count as its rule"
    for units in (1, UNITS, 613):
        text = rules[0].replace("$units", str(units))
        rule, why = rule_of(text)
        assert rule is not None, (text, why)
        program, query = population_program([(SRW.template(STANDIN).fn, units)])
        names = S.selection_sampling_design(rule, units, S.selection_sizes(rule, program, query))
        assert names == {KEY: "work", "k": units}, (text, names)


#: `weights-tie pins`' outcomes: its status, its line and its stderr, whether it wrote the pins, and whether the tie
#: takes it as its answer
PINS = {
    "pinned": (0, '{"pinned": true, "weights": 3}', "", True, True),
    "refused": (1, '{"pinned": false, "why": "the tie window holds no root of it"}', "", False, False),
    "traceback": (1, "", "Traceback (most recent call last):\nRuntimeError: died\n", False, False),
    "refused, then a traceback": (1, '{"pinned": false, "why": "x"}', "Traceback (most recent call last):\n", False,
                                  False),
    "killed": (137, "", "", False, False),
}


def test_the_weights_ties_arm_takes_only_pins_own_answer_and_a_traceback_is_no_refusal(tmp_path):
    """The tie passes only on status 0, its ``{"pinned": true, ...}`` line and the pins. A refusal, a traceback or a
    kill is not that, and the arm says PINS ERROR."""
    arm = RUN_SH.read_text().split('if [ -n "${PP_WEIGHTS_TIE:-}" ]; then', 1)[1].split("\nfi\n", 1)[0]
    fn = re.search(r"^  pins_answered\(\) \{.*?^  \}$", arm, re.M | re.S)
    assert fn and "pins_answered $pinned " in arm and "PINS ERROR" in arm, (
        f"{RUN_SH.name}'s weights tie arm no longer judges `weights-tie pins` by `pins_answered`")
    for case, (status, line, err, wrote, want) in PINS.items():
        d = tmp_path / case.replace(" ", "-").replace(",", "")
        d.mkdir()
        (d / "verdict").write_text(f"{line}\n" if line else "")
        (d / "err").write_text(err)
        if wrote:
            (d / "pins").write_text("{}")
        got = subprocess.run(["bash", "-c", f'{fn.group(0)}\npins_answered {status} "$1/verdict" "$1/err" "$1/pins"',
                              "-", str(d)], check=False).returncode == 0
        assert got == want, (case, got)


def test_each_partition_file_states_a_rate_or_is_listed_as_still_stating_a_fixed_design():
    """A partition file's text is the rule an audit states to its stream, on a challenger's coins or from a served
    window's records, so each one states a rate a stream takes, or is in `FIXED` and that audit is refused at the rule
    check. A file outside the list that states a fixed design fails, and so does a stale entry: a listed file that states
    a rate, or that is gone."""
    files = partitions()
    assert files, f"no partition file beside {RUN_SH.name}: `partitions` reads none"
    fixed = {name for name, text in files.items() if rule_of(text)[1]}
    unlisted = {name: files[name] for name in sorted(fixed - FIXED)}
    assert not unlisted, (f"a partition file states a rule a stream refuses, so an audit on a challenger's coins refuses "
                          f"it: state {RATE} in it; FIXED only shrinks ({unlisted})")
    stale = {name: files.get(name, "no such partition file") for name in sorted(FIXED - fixed)}
    assert not stale, (f"FIXED lists a file that no longer states a fixed design: delete its entry, and where "
                       f"{RUN_SH.name}'s header says the arm reading it is refused at the rule check, correct it ({stale})")
