"""The rules `run.sh` states and the partition files an audit reads, held to what a stream takes, without running the
script:

* each rule the script states for its stand-in, the honest run's and the forged run's, under ``--zk`` and under
  ``PP_PLAIN=1``, is a rate (a stream refuses a fixed design as its rule), and over the stand-in's window, one template's
  stratum of eight proof blocks a rank, its sampling design is the one the window registers under it: half the proof
  blocks for the honest run, every one for the forged run, and a stratified design under ``PP_PLAIN=1``;
* each partition file (`partitions`: this benchmark's, the one-stage audit's and each one the script names) states a
  rate a stream takes or is listed as still stating a fixed design (`FIXED`), which an audit on a challenger's coins
  (`prove_path.py audit`, `a5.py --challenger`) refuses at the rule check. The list only shrinks."""
from __future__ import annotations

import json
import re
from pathlib import Path

from verity.commitments.windows.population import population_program
from verity.protocols.spot.verifier import service as S
from verity_vllm.pipeline import serving_rows as SRW

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parents[1]
RUN_SH = BENCH / "run.sh"
#: where the partition files are kept: beside this benchmark's drivers and beside the one-stage audit's
KEPT = (BENCH, REPO / "benchmarks" / "one_stage")
#: the script's variable for the kind of its stand-in's rules, and a partition file's key for the rule it states
VAR, KEY = "LAW", "law"
#: the stand-in's window a rank (`tests.pipeline.served_tp --world 2 --I 256`): its one template and its proof blocks
STANDIN, UNITS = "SiluMul_v2{I=256}", 8
RATE = "a rate (rate:NUM/DEN, stratified-rate:NUM/DEN or work-rate:NUM/DEN)"
#: the partition files that still state a fixed design (``subset:K``, ``stratified:K`` or ``work:K``), by path from the
#: root. A stream refuses such a text as its rule, so an audit on a challenger's coins refuses the file. The list only
#: shrinks: a file leaves it in the change that makes its text a rate, and none joins it.
FIXED = frozenset({
    "benchmarks/one_stage/glm53_all_families_tp8_step16.json",
    "benchmarks/one_stage/glm53_moe_w13_tp8_step16.json",
    "benchmarks/one_stage/glm53_moe_w2_tp8_step16.json",
    "benchmarks/one_stage/glm53_prefill_tp8_step0.json",
    "benchmarks/one_stage/glm53_prefill_tp8_step0_core.json",
    "benchmarks/one_stage/glm53_silu_mul_v2_tp8.json",
    "benchmarks/one_stage/glm53_silu_mul_v2_tp8_step16.json",
    "benchmarks/one_stage/glm53_silu_mul_v2_tp8_step16_stratified.json",
    "benchmarks/one_stage/glm53_sparse_mla_tp8_step16.json",
    "benchmarks/one_stage/llama32_1b_rope_d64_l0.json",
    "benchmarks/one_stage/llama32_1b_silu_mul_v2_tp2.json",
    "benchmarks/prove_path/glm53_silu_mul_v2_i1536_step16_all.json",
    "benchmarks/prove_path/llama32_1b_silu_mul_v2_i8192_l0_step0.json",
    "benchmarks/prove_path/llama32_1b_silu_mul_v2_i8192_l0_step1.json",
    "benchmarks/prove_path/llama32_1b_silu_mul_v2_i8192_l0_step24.json",
    "benchmarks/prove_path/llama32_1b_tg_l0_step1.json",
    "benchmarks/prove_path/llama32_1b_tg_l0_step24.json",
})


def rule_of(text: object) -> tuple[S.Selection | None, str | None]:
    """``text`` as a stream's rule, as `a5.stream_of` makes it of what the audit states, or why a stream refuses it: a
    text that is no rate, or a Bernoulli design, which `Stream.register` refuses. -> (the rule, None) or (None, why)."""
    try:
        rule = S.Law(text)
    except ValueError as e:
        return None, str(e)
    return (None, S.BERNOULLI) if rule.text.startswith("bernoulli") else (rule, None)


def stated() -> list[tuple[bool, bool, str]]:
    """The rules the script's stand-in states, each as (under ``PP_PLAIN=1``, the forged run's, its text): the kind the
    script's variable holds, first under ``--zk`` and then under ``PP_PLAIN=1``, with the rate the honest run hands
    `served_tp` and the one the forged run writes to its partition file."""
    text = RUN_SH.read_text()
    kinds = re.findall(rf"\b{VAR}=([\w-]+)", text)
    honest = re.findall(rf"--{KEY} \${VAR}:([\w/]+)", text)
    forged = re.findall(rf'\\"{KEY}\\": \\"\${VAR}:([\w/]+)\\"', text)
    assert (len(kinds), len(honest), len(forged)) == (2, 1, 1), (
        f"{RUN_SH.name} no longer states its stand-in's rules as this test reads them ({VAR}=KIND twice, --{KEY} ${VAR}:RATE "
        f"and \"{KEY}\": \"${VAR}:RATE\" once each): found {kinds}, {honest}, {forged}; state them so, or teach `stated` the "
        "new form")
    return [(plain, is_forged, f"{kind}:{arg}")
            for plain, kind in zip((False, True), kinds, strict=True) for is_forged, arg in ((False, honest[0]), (True, forged[0]))]


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


def test_each_rule_the_stand_in_states_is_a_rate_whose_design_its_window_registers():
    """`prove_path.py audit` states its partition file's text as the stream's rule, so a fixed design there ends the audit
    at the rule check, before any session opens. Each of the stand-in's four rules is a rate, and over a rank's window the
    sampling design the stream names for it, which its `register` holds the registration to, is the one the window
    registers: the serving rows read the partition file's text as a rate of that kind and resolve it over the window's
    program, by the steps of `serving_rows.population`. The honest run samples half a rank's proof blocks and the forged
    run every one, so its forged unit is sampled; a plain session's design is stratified."""
    rules = stated()
    refusals = {text: why for _plain, _forged, text in rules if (why := rule_of(text)[1])}
    assert not refusals, (f"{RUN_SH.name}'s stand-in states a rule a stream refuses, so its audit ends at the rule check: "
                          f"state {RATE} instead ({refusals})")
    program, query = population_program([(SRW.template(STANDIN).fn, UNITS)])
    for plain, forged, text in rules:
        rule, _why = rule_of(text)
        names = S.selection_law(rule, UNITS, S.selection_sizes(rule, program, query))
        read = SRW.law_of(text)
        assert read == {KEY: names[KEY], "rate": text}, (text, read)
        registers, table = S.params_law(read["rate"], program, query, SRW.work_table_of(read, {STANDIN: UNITS}))
        assert registers == names and table is None, (text, registers, names, table)
        assert names["k"] == (UNITS if forged else UNITS // 2), (text, names)
        assert not plain or names[KEY] == "stratified", (text, names)


def test_each_partition_file_states_a_rate_or_is_listed_as_still_stating_a_fixed_design():
    """A partition file's text is the rule an audit on a challenger's coins states, so each one states a rate a stream
    takes, or is in `FIXED` and that audit is refused at the rule check. A file outside the list that states a fixed
    design fails, and so does a stale entry: a listed file that states a rate, or that is gone."""
    files = partitions()
    assert files, f"no partition file beside {RUN_SH.name}: `partitions` reads none"
    fixed = {name for name, text in files.items() if rule_of(text)[1]}
    unlisted = {name: files[name] for name in sorted(fixed - FIXED)}
    assert not unlisted, (f"a partition file states a rule a stream refuses, so an audit on a challenger's coins refuses "
                          f"it: state {RATE} in it; FIXED only shrinks ({unlisted})")
    stale = {name: files.get(name, "no such partition file") for name in sorted(FIXED - fixed)}
    assert not stale, (f"FIXED lists a file that no longer states a fixed design: delete its entry, and where "
                       f"{RUN_SH.name}'s header says the arm reading it is refused at the rule check, correct it ({stale})")
