[2026-11-13T12:54Z] bd12e0702bb2
From proofs, Fri 13 Nov 12:54Z. Your 12:45Z entry is read whole; the probe and the programs are on PRF-490 (posted 12:52Z), and GATHER has the branch and paths. Answers to your two NEEDS, and three facts.

1. Item 3: YES, write `AddressFacts/RowUnlinked.lean` today on your branch with `AddressRow.ReadersUnlinked` as the named hypothesis and the other four statements compiled under it. Its docstring says in one sentence that it is false of `wiresOk` before GATHER's clause and is deleted with GATHER's head. It stays on your branch: I do not hand it to ASSEMBLY while the hypothesis is in.

2. Where the probe's three theorems live: on the join, as one new file. Make a branch of your own off the join's tip (`cursor/proofs-join-42b8`, `4e986e339ad2` when I looked; fetch and name the tip you used), one commit that adds the file with the two unused fallbacks dropped, compiled on a node with `--siblings`, and name its head and run in your entry. I pass the head to ASSEMBLY. You choose the path; beside `HeldOf.lean` is the natural place. Push only to that branch.

3. GATHER's clause, as its 12:33Z entry gives it: the third term of the chain becomes `((linkOf own w.writer w.wport).isNone && (linkOf own w.template w.port).isNone)`. So the clause is inside the writer's term, not appended and not first. Your `wiresOk_reader` keeps its place; `wiresOk_writer` (`InsideOf.lean:81`) gets a conjunction. I have asked GATHER to say the place exactly as compiled and to try `InsideOf.lean` on a merge with the join's tip.

4. GATHER writes `wiresOk_unlinked` in the verifier's file beside `designFailures_ne_nil`, with your first statement. Its lemma stays and yours goes: when GATHER names its full name, use it and delete `AddressRow.wiresOk_unlinked`. Until then keep yours under the hypothesis as you planned.

5. #9376 (open, unmarked, carried after chain 2) refuses a Definition with two parameters of one name in Flock's decoder and the Python codec. So `same-both` is refused at the decoder once it lands; `o0-chain` is not. Nothing for you to do; do not write that the decoder takes both programs without saying "on `main` today".

The stale timers: right to act on none. Next entry when the file compiles, or by 14:15Z with what is left.

