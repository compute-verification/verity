"""circuit-check: the property checks every circuit passes whenever it is built or changed (a Definition, a subcircuit template,
a served row's Program, their Boolean lowerings).

    checks    the checks and the report (`verity-circuit-check/v1`)
    targets   what a target string names, and the catalog the suite runs over
    vectors   the edge and random input vectors
    cli       ``circuit-check TARGET... | --all [--out report.json]``

The suite (`tools/verity/circuit_check/tests`) runs it over every registered Definition and template; a failure is a correctness or
invariant violation, named; redundant and dead gates are warnings with counts.
"""
