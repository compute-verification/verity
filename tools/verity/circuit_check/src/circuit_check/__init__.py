"""circuit-check: the property checks every circuit passes whenever it is built or changed (a Definition, a subcircuit template,
a served workload's Program, their Boolean lowerings).

    checks    the checks and the report (`verity-circuit-check/v1`)
    bodies    a composite's evaluation and its non-constant gates, body by body
    targets   what a target string names, and the catalog the suite runs over
    sidecars  each Definition module's bindings and pins, in sidecar files beside it
    vectors   the edge and random input vectors
    structural  a ``flock-ir-tmpl`` template held to its Definition's flat lowering (``--structural``)
    ambient   each lowering one text in a plain and a perturbed interpreter, reading no environment, clock or draw (``--ambient``)
    cache     the result cache: a target's report reused while everything its check read is unchanged
    units     the catalog bindings' records, each reused while what its build read is unchanged (``--catalog-units``)
    known     the failures `main` already has, each with its cause (``known/``)
    entries   the shared lists' layout, one file per entry (``known/``, ``pins/``)
    cli       ``circuit-check TARGET... | --all | --structural UNIT.tmpl TARGET | --ambient TARGET... | --batch [--out report.json]``

The suite (`tools/verity/circuit_check/tests`) runs it over every registered Definition and template; a failure is a correctness or
invariant violation, named; redundant and dead gates are warnings with counts.
"""
