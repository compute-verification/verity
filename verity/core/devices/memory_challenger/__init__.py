"""The memory challenger, the auditor-trusted device that runs PoUS's timed rounds and the streamed wipe's reads and signs
each. Its Lean is `lean/` (the Lake package `core_memory_challenger`), whose difftests hold it to the rounds' Python, PoUS's
`memory_challenger`; its timed audit's process is `timed_audit`, a client of a responder's wire whose verdict PoUS's
Lean check (`pous_check`'s `pous-timed-audit`) is held to. `challenger_party` is its party on the challenger host, over
the messages, ids and prover's side PoUS holds (`challenger_session`)."""
