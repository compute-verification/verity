"""``verity bench numerical JOB`` (``verity_numerical/__main__.py``): every job is one of the package's command-line modules,
started with the rest of the arguments, and a name that is no job is refused with the list."""
from verity_numerical import __main__ as jobs
from verity_numerical.bench import (
    cell,
    fixture_sets,
    generate,
    instances,
    judge,
    latency,
    placement,
    subcircuit_catalog,
    unit_costs,
    unit_costs_menu,
    unit_costs_padding,
    unit_costs_prover,
)
from verity_numerical.checker import search, sweep
from verity_numerical.checker.v2 import search as search_v2

MODULES = {"generate": generate, "fixture_sets": fixture_sets, "instances": instances, "subcircuit_catalog": subcircuit_catalog,
           "unit_costs": unit_costs, "unit_costs_padding": unit_costs_padding, "unit_costs_prover": unit_costs_prover,
           "unit_costs_menu": unit_costs_menu, "latency": latency, "judge": judge, "cell": cell, "placement": placement, "sweep": sweep,
           "search": search, "search_v2": search_v2}


def test_every_job_names_a_module_of_the_package_that_has_a_main():
    assert set(jobs.JOBS) == set(MODULES)
    for name, (module, what) in jobs.JOBS.items():
        assert MODULES[name].__name__ == f"verity_numerical.{module}" and callable(MODULES[name].main) and what


def test_a_job_runs_with_the_rest_of_the_arguments(capsys):
    assert generate.main(["--list"]) == 0
    direct = capsys.readouterr().out
    assert jobs.main(["generate", "--list"]) == 0
    assert capsys.readouterr().out == direct != ""


def test_no_job_prints_every_job_and_a_name_that_is_none_is_refused_with_them(capsys):
    assert jobs.main([]) == 2
    listed = capsys.readouterr().out
    assert all(f"  {name} " in listed for name in jobs.JOBS)
    assert jobs.main(["--help"]) == 0 and capsys.readouterr().out == listed
    assert jobs.main(["nonesuch"]) == 2
    out = capsys.readouterr()
    assert out.out == "" and "no job 'nonesuch'" in out.err and listed in out.err
