# Inbox for worker CARVE (top's branch; read before you open the pull request)

## Wed 11 Nov 10:14 PM PST, from top

1. **The Configuration row gets one clause from soundness** (Linear MEM-279, 10:12 PM PST): Lean's `Composite` is a configuration's list of protocols and the ties between their named values, without where a party runs. Write it as true on `main` today (the Lean name is `Composite`; its rename is a later change and is not yours to announce).
2. **The moves wait for the merge of the remote branch.** In `AGENTS.md`'s 'Target, not yet' wording, name the targets only; give no date and no order of work.
3. **Nothing else changes in your brief.** No objection to the words has come; soundness agrees with them.

## Wed 11 Nov 10:31 PM PST, from top

4. **Leave `verity/protocols/remote/verifier/composition/README.md` untouched** (its line 3 says "composite protocol"). The directory holds a Lake package, and the queue counts a Lake package's whole directory as a Lean input, so one changed README there makes the pull request wait behind every Lean batch. The same goes for any file under `security/` and any directory that holds a `lakefile.toml`, `lakefile.lean` or `lean-audit.json`. In the body, list each site you leave for this reason under "Left for proofs' word move", with its path and line. proofs takes them (MEM-279, 10:22 PM PST).
5. **proofs asked (MEM-279, 10:28 PM PST) that three sites be in your pull request,** because they must not land after the move: `ARCHITECTURE.md:174` ("It is not the configuration that assigns roles to machines"), `GLOSSARY.md:253`, and `tests/test_retired_terms.py:101` (it retires "onsite deployment, remote deployment" in favour of "composite protocol"; the entry must now retire "composite protocol" too and point at "configuration"). If any of the three is not in your diff, add it.
6. **Point 2 above is out of date but changes nothing you write:** lean withdrew its hold at 10:21 PM PST. Still give no date and no order of work.
