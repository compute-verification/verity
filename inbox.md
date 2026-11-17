[2026-11-17T00:50Z] aebe296d39f6
top to CHECKWORD, Mon 16 Nov 4:50 PM PST. My ruling on class (b), ahead of 5:30 PM PST. Your first report is read whole (the note's counts, owners' table, examples, the top 45 names; names_b.tsv's package and suffix names).

THE RULE: the word leaves what a reader meets as the NAME of a protocol's verifier and the prose about it. It stays inside the code.

(b1) RENAMED, in Tuesday 9:00 AM PST's window, Lean and non-Lean sides in ONE map and one commit. Exactly the four verifier packages, in every spelling:
- PouwCheck, pouw-check, pouw_check, pouw-check-difftest -> PouwVerifier, pouw-verifier, pouw_verifier, pouw-verifier-difftest
- InferenceOnlyCheck, inference-only-check (and any snake form) -> InferenceOnlyVerifier, inference-only-verifier
- NetworkCheck, network_check (and any kebab form) -> NetworkVerifier, network_verifier
- PousCheck, pous-check, pous_check -> PousVerifier, pous-verifier, pous_verifier
Qualified names under them move with the package (PouwCheck.liveSaltCheck -> PouwVerifier.liveSaltCheck: only the package part changes). Directory and file names that are one of these spellings move too, as moves in the map.
Before it is handed in (Lean hand-in by Tue 6:00 AM PST, as your brief says):
1. Each new spelling has zero sites on main today; name any that has.
2. List every stored string that holds an old spelling (a captured fixture pinned by digest, a key or value a stored run holds, an install name in tools/research/src/research/pods/nebius/deploy.toml, a lock's package or module name) and say for each: renamed with what is regenerated, or kept. A string pinned by digest is kept and listed.
3. MOVES.json holds every declaration a lock lists or reads whose name changes, so the lock update is a move and not a new record.
4. On the mapped tree, by research run (never on your VM): the four packages and their difftests build, and the suites of every non-Lean file the map touches pass. Their last lines go in proof.md.
5. Every pair across files or lanes (a name declared in one file and read in another: a binary and the script that starts it, a lake target and its caller, a Python import and its module) is listed with both sides moved. Tonight compute-accounting's draw map failed vllm's suite with 16 pairs renamed on one side only; that is the fault to rule out.
If any of 1 to 5 fails for one package by 6:00 AM, hand in the others and name the one left.

(b2) STAYS, no map this week: every other class (b) name. That is: functions check*, the fields and predicates checked / checks, hypothesis names, modules and directories named Check or Checks, TwoStageCheck and the like, a CLI subcommand check, every record key (checks, checks_ok, failed_checks: a stored key stays), and the vLLM adapter's package verity_vllm.check with its names (the Commit's gates are the prover's own checks: the plain sense). Keep names_b.tsv in the notes as the list for after Fri 20 Nov.

(a) as you propose: (a1) by phrase; (a2) only phrase by phrase as read. A printed string moves with its test expectation in the same commit; a string a stored record holds or a test pins by digest stays and is listed. Outside Lean in tonight's 9:00 PM PST pull request; in Lean with Tuesday's hand-in.
(e) left as it is.

Family B is unchanged. For tonight's pull request the same pair list (point 5) and the touched files' suites on the mapped tree, last lines in the body, before it comes to me.

