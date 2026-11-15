# `unread.py`'s fixture

A tree of four Lake packages laid out as the repository's are: `security` (`Definitions`), `security/Proofs` and
`security/Properties` (each with `srcDir = ".."`), and one code package, `code`. `test_unread.py` builds it in a copy and
checks what the closure keeps: the guarantee's proof and what it reads, the properties, the code, a notation, a
declaration a tool names (`tools/reader.toml`), a module of checks a test names by path (`tools/checks.py`), `@[defeq]`
theorems, a `Coe` instance (a property uses the coercion,
whose term names `Core.unwrap` and not the instance), and a module that gives another's theorem `simp`.
Nothing reads `Definitions.Dead` (through the aggregator `Definitions.Gone`, and an unread theorem of
`Proofs.Flock.Main` uses it) or `Proofs.Flock.Spare` (which the aggregator `Proofs.Flock` opens), so of the three wholly
unread modules only `Definitions.Gone` is deletable alone; nothing imports `Definitions.Orphan`. This README names
`Core.unusedHelper`, which keeps nothing.
