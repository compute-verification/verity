# Brief: the V* bridge pieces (ONE-9p's `acc_of_accepts`), Linear PRF-202

**Due:** each piece's lemma statement posted on PRF-202 within about 1 h of starting; proved and pushed by **Tue 27 Oct 12:00Z** at the latest (dates are ceilings: faster is better). ONE9P (bc-011cc572) merges the pieces in its glue piece G. **Go/no-go for recursion in F3 is Wed 06:00Z**; if the bridge misses it, onsite leaves the recursive component out (Daniel, 12:38 PM PDT), so the earlier you land, the more F3 covers.

## What the bridge is

ONE9P proves `acc_of_accepts`: the compiled V* check `VStar.Recursive.checkBlobs` (ONE9's component, `cursor/soundness-one9-28d7`, PRF-201) accepting implies the model verifier `RecDeploy.V`'s run accepts (`(Claims.run sha512 P (recordAt …).V x ω r).out = .acc`), which `Flock.Guarantees.RecursiveKSAt` is stated over. Read on PRF-202: ONE9P's 19:04Z statement, its 19:44Z **split table** (your piece's row is your task: what it proves, its size, its dependencies), and its 19:58Z amendment (`Names`, `Shows`, R6). Read ONE9P's note `~/.research/notes/lanes/soundness/*one9p*`.

## Rules for every piece (ONE9P's)

- Branch from `origin/cursor/soundness-one9p-25dd` (`git fetch origin cursor/soundness-one9p-25dd && git switch -c cursor/soundness-one9-<piece>-<4 hex> FETCH_HEAD`) and add **only your own file** under `Security/Proofs/Properties/Verity/Recursive/` (package `security_proofs`). Push only to your branch.
- Interfaces between pieces are hypotheses: state the other pieces' facts you need as hypotheses, by name, and don't wait for them.
- Post your lemma's exact statement on PRF-202 **before** proving.
- **Never weaken a compiled check, or the model, to make a piece go through.** If the compiled code lacks a check the model makes (B's suspected `innerOk` gap), post it on PRF-201 for ONE9 to add, with the exact check, and continue against it as a hypothesis.
- No `sorry`, `axiom` or `native_decide` in what you push as done.
- No PR of your own: ONE9P merges your branch in G. Your final message names the branch, head SHA, the lemma's name and statement, and the `verity lean try` run id with its verdict line.

## Lean runs

`verity lean try Security/Proofs/Properties/Verity/Recursive/<YourFile>.lean --on vy-nebius-cpu-4 --pool soundness`. Until #4794 lands, `--pool soundness` exits 2: add `"soundness": "/workspace/research/locks/soundness/slots"` to `POOLS` in `tools/research/src/research/pods/nebius/slot_only.py` **locally, uncommitted** (ONE9P did the same). Scratch probes (`#print axioms`, `exact?`) go through the same command.
