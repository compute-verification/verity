"""circuit-check: the property checks every circuit passes whenever it is built or changed (a Definition, a subcircuit template,
a served row's Program, their Boolean lowerings).

    checks    the checks and the report (`verity-circuit-check/v1`)
    targets   what a target string names, and the catalog the suite runs over
    vectors   the edge and random input vectors
    structural  a ``flock-ir-tmpl`` template held to its Definition's flat lowering (``--structural``)
    ambient   each lowering one text in a plain and a perturbed interpreter, reading no environment, clock or draw (``--ambient``)
    cli       ``circuit-check TARGET... | --all | --structural UNIT.tmpl TARGET | --ambient TARGET... [--out report.json]``

The suite (`tools/verity/circuit_check/tests`) runs it over every registered Definition and template; a failure is a correctness or
invariant violation, named; redundant and dead gates are warnings with counts.
"""
