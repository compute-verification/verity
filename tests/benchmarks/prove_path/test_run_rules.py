"""The rules `run.sh` states and the partition files an audit reads, held to what a stream takes, without running the
script:

* the rule the script's weights tie arm states to its records audit is the rate of every unit of the tie, which names
  the tie's own design, work over every one of its units;
* that arm tells its records audit what each tie that must be refused forges, a registered read or a unit, and no other
  arm of the script audits a window as forging a read;
* each partition file (`partitions`: this benchmark's, the one-stage audit's and each one the script names) states a
  rate a stream takes or is listed as still stating a fixed design (`FIXED`), which an audit on a challenger's coins
  (`prove_path.py audit`, `a5.py --challenger`) refuses at the rule check and a served window's audit from its records
  (`records_audit.py audit`) refuses by name. The list only shrinks."""
from __future__ import annotations

import json
import re
from pathlib import Path

from verity.commitments.periods.population import population_program
from verity.protocols.proofs.spot.verifier import service as S
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
    arm = RUN_SH.read_text().split('elif [ -n "${PP_WEIGHTS_TIE:-}" ]; then', 1)[1].split("\nelse\n", 1)[0]
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


def test_the_weights_ties_arm_tells_the_audit_what_each_refused_tie_forges():
    """The weights tie's arm audits two ties that must be refused and tells `records_audit.py audit` what each forges,
    since the audit passes a refused window on that kind's check alone. ``weights-tie-other`` serves another
    checkpoint's words as the registered one's: a registered read that opens no pinned root, in a window of no false
    unit (``--forged-read``). ``weights-tie-forged`` forges its ends' digests too: false units (``--forged``). The
    honest tie is told neither. `tie` sets what the audit is told from its own arguments, each in turn, and hands it to
    the audit, and no other arm of the script names ``--forged-read``."""
    def code(text: str) -> str:
        """``text``'s lines but its comments', each without a comment after its code."""
        return "\n".join(x.split(" #", 1)[0] for x in text.splitlines() if not x.lstrip().startswith("#"))

    script = code(RUN_SH.read_text())
    arm = script.split('elif [ -n "${PP_WEIGHTS_TIE:-}" ]; then', 1)[1].split("\nelse\n", 1)[0]
    told = dict(re.findall(r"^\s*(--[\w-]+)\) forged=(\S+);", arm, re.M))
    ties = re.findall(r"\btie (weights-tie[\w-]*)((?: --[\w-]+)*) \|\| rc=1", arm)
    assert told and ties and re.search(r'for x in "\$@"; do\s+case \$x in', arm), (
        f"{RUN_SH.name}'s weights tie arm no longer sets what its audit is told as this test reads it (a `case` over "
        f"`tie`'s arguments in turn, each `--ARG) forged=FLAG;`, and `tie NAME [--ARG ...] || rc=1`): {told}, {ties}")

    def audited(args: str) -> str | None:
        flag = None
        for x in args.split():
            flag = told.get(x, flag)
        return flag
    assert {name: audited(args) for name, args in ties} == {
        "weights-tie": None, "weights-tie-other": "--forged-read", "weights-tie-forged": "--forged"}, (told, ties)
    audits = re.findall(r"records_audit\.py audit [^\n]*", arm.replace("\\\n", " "))
    assert len(audits) == 1 and " $forged " in audits[0] and "--forged" not in audits[0], (
        f"{RUN_SH.name}'s weights tie arm hands its one records audit what `tie` set, and no flag of its own: {audits}")
    assert script.count("--forged-read") == arm.count("--forged-read") == 1, (
        f"{RUN_SH.name} audits a window as forging a registered read (`--forged-read`) outside its weights tie's "
        "`--served`: say in its header which arm passes on that refusal, and hold it here")


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
