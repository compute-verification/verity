# `unread.py`'s fixture

A tree of four Lake packages laid out as the repository's are: `Security` (`Definitions`), `Security/Proofs` and
`Security/Properties` (each with `srcDir = ".."`), and one code package, `code`. `test_unread.py` builds it in a copy and
checks what the closure keeps: the guarantee's proof and what it reads, the properties, the code, a notation, a
declaration a tool names and a module of checks it names by path (`tools/reader.toml`), `@[defeq]` theorems, a `Coe` instance (a property uses the coercion,
whose term names `Core.unwrap` and not the instance), and a module that gives another's theorem `simp`.
Nothing reads `Definitions.Dead` (imported only through the aggregator `Definitions.Gone`) or `Proofs.Flock.Spare`, and
nothing imports `Definitions.Orphan`. This README names `Core.unusedHelper`, which keeps nothing.
