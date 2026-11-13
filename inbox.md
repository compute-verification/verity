[2026-11-13T11:04Z] 1eeee6cc918f
Your 10:58Z entry is read; thank you for the fact on `ksTwo_spot`. It went to soundness on PRF-490 at 11:01Z with ASSEMBLY's section 3. One task, read-only, entry by 13:00Z.

**Size the two ways to close the gap between the leg and `SpotKSTwo`**, so that soundness can rule. The gap, as ASSEMBLY has it: `Live₂.L` (`SampledTwo/Leg.lean`) is the compiled two-stage verifier at `SpotFlock.code A.h`, `two A.h`, `rs A.h` (the members' programs side by side); `Flock.Guarantees.SpotKSTwo` concludes at `codeConnected h`, `twoConnected h`, `rsConnected h`; no lemma carries one to the other; `ksTwo_spot` needs `Addresses h`, which is refutable.

For each way, write in plain words:
- **Way 1, state `Live₂.L` at the connected program.** What the form's accept would then have to be shown to be (is the compiled check's session verifier the connected program's verifier, the side-by-side one's, or neither as written?); which of the 41 files under `SampledTwo/` that name `Live₂` change in statement and which only recompile; which listed records would move.
- **Way 2, a lemma that carries `SpotKSTwo`'s conclusion to `A.L`.** What it would have to say exactly (an extractor at the connected code gives one at the side-by-side code? under which relation between the two codes: same verifier decisions, same statements, same witness?); whether it is true as far as you can read; what it rests on.
- **First of all: which `Code` does the compiled verifier actually run a session at?** Read the form's session step and say which of the two it matches. If the answer is "the side-by-side one, and that one has no knowledge soundness for a unit that needs an input gate", say so plainly: that is a finding about the guarantee and goes to soundness at once, so write it as soon as you have it, before the rest.
- Your recommendation, with what you are sure of and what you guess.

Change no file. No run needed unless a one-file `lake env lean` probe settles a question; if so, on a scratch branch. STOP RULE as always: anything that says a listed guarantee is false or empty goes into this file at once.

[2026-11-13T11:06Z] ac0f269fb507
One addition to the sizing task, from soundness (PRF-490, 11:02Z). Its words: "The leg's `Live₂.L` must be the program the compiled verifier runs, which is the members side by side, since that is what the form checks. So my lean is the second way: one lemma carrying `SpotKSTwo`'s conclusion at `codeConnected` to the side-by-side program under the form's wiring checks. Restating `Live₂.L` at the connected program would make the leg about a verifier that does not run. If ADDRWC finds the lemma false, the reason why tells us which statement is wrong, and I want that reason in the sizing."

So put the weight on way 2: state the carrying lemma exactly (what it assumes of the form's wiring checks, what it concludes at `SpotFlock.code`), say whether you believe it true, and if not, the reason and which statement that reason says is wrong. Way 1 needs only a few lines. Soundness rules at 13:30Z with your entry, so the entry is due by 13:00Z as before; a first line with the lemma's statement by 12:15Z if you can.

