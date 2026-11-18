# `research.pods`: the machines

These are the machines `research` runs work on, and what keeps each one bounded. A machine has a name in a registry
(one file per machine in the notes repository), and `research run --on NAME` reaches it by that name. A rented pod
carries a lease and a budget line from the moment it is created, and a guard on the control pod enforces both, so
nothing that bounds spend depends on an agent VM, which can be suspended at any moment, or on one process staying
alive. The Nebius nodes are set up from the files under `nebius/`.

How a run reaches a machine is in `tools/research/README.md` ("Remote runs").

## Try it

~~~sh
research pods registry
~~~

This prints every machine `research run --on` accepts, a line each, with where its entry comes from. It reads only the
notes repository. It exits 1 when an entry is refused, and that entry's line says why.

## `research pods`

~~~text
research pods registry [--json]             every machine `research run --on` accepts, and where its entry comes from
research pods register NAME --host ADDR --project P    bind a name to an ssh machine; refused unless it answers over ssh
research pods unregister NAME               drop the name
research pods env NAME [NAME ...]           `export <NAME>_HOST=<host>` for each, for `eval` only: it prints addresses
research pods ssh TARGET [-- CMD ...]       reach a machine by its name
research pods sync TARGET [SRC]             put a worktree's files on it (what git would show, nothing ignored)
research pods lease TARGET                  its lease line
research pods extend TARGET --hours H       the lease ends no sooner than now + H; it is never shortened
research pods check-part TARGET [--json]    whether it is the reference part; exit 3 when it is not
research pods guard status                  the budgets guard: `guard` runs it, `guard stop` ends it
research pods unpreserved TARGET            the Attempts recorded there that the store has not preserved
research pods drain TARGET                  terminate it only when nothing on it is unpreserved
~~~

`create` makes a RunPod pod with its lease, and any other verb (`list`, `terminate`) goes to `runpod.py` unchanged.
`create` needs `--max-hours` and a budget line that covers the name, and makes at most one live pod per `--name`, so a
retry loop can't duplicate pods: exit 4 when one exists (`--allow-duplicate` overrides), 5 for RunPod's explicit
no-stock reply, the one failure to answer with another spec, 6 when no budget line lets it, and 2 for any other failure.
