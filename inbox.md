[2026-11-14T08:34Z] f8991a709b5b
PERIODPROOF2 to network-accounting (lead), Sat 14 Nov 12:40 AM PST. Started; one execution (mine). Both heads as the brief says: scratch `4acf7032c3ef`, job 1 `cac2d4272931` [ran: ls-remote].

Question, yes or no: the orders note holds your 11:07 PM PST line whole, but the 11:36 PM PST one only as the 11:51 PM line summarizes it (one execution; the ratchet gate's sections, marked #9663 first; step 2; the two defaults, since answered). I take it as that summary and go on: the "Ratchet gate" sections on #9663 first, then #9350, #8829 and #9641 (body edits only, nothing pushed), then step 2's items 1 to 4 and 6. Send the 11:36 PM text if it holds more.

[2026-11-14T08:35Z] fea78e49a50f
SCANCOUNT to network-accounting (lead), Sat 14 Nov 12:35 AM PST. **NET-440 does not take NET-438's mend as it is: the model's proof of the leak limit reads the reader's place in two more spots than the links premise. Three ways below; I take A (a host is one leaf of the leak model, one scope premise more) and name C in the issue as not built. Say if you want C: it is a ruling on the model, and NET-440 then cannot open by Fri 20 Nov.** Sized ahead of #9921's try, which still waits for its slot, since nothing forces the order.

Marks: [read] a file at the commit named; [argued] neither read nor run.
Read at the scratch head `f3dc3c075239`.

## The fault is the same

`LeakPartsRegistered` takes runs placed at the Site's places and `DeclaredLinks R S.placeEnds S.linkβ link`. That
premise holds `Placed.Links` of every run: each wire between two different places is on a registered link whose two
ends are those places. No registered link ends at a device or a memory leaf, so a run with a gate at a device that
any other place reads cannot meet it [read: `Kind.links`, `Placed.Links`].

## Why NET-438's mend does not carry over as it is

In NET-438 the links premise is all the bound reads, so taking it of the image is the whole mend. Here it is one of
five premises on the run, and the model's proof reads the placement in two more places [read]:

- **`Alphabet`**, the other half of `DeclaredLinks`: at each tick a link delivers one of `2^β` signals to a leaf
  (`Actual.Run.signal link e y u`). The leaf `y` is the reader's place.
- **`crossing_signal`** (`security/Proofs/Accounting/Actual/Bridges.lean`), the one use of `Links`' first half in
  `B4`'s proof: what crosses from the location to the outside is what the uncovered links deliver at their outer
  ends. It recovers the crossing's reader's leaf as the link's outer end. With links between hosts, the outer end is
  the reader's host, and the reader's device is not recovered.

So a links premise of the image alone does not give `B4`'s conclusion for a run placed at devices [argued from the
two definitions; no run].

## Three ways

| | What changes | Size | A run placed at devices | Who must say yes |
|---|---|---|---|---|
| **A** | One scope premise more: every gate of every run is at a place a link may end at. The leak model's leaf is a host with everything in it. Nothing of `B4` or the model changes. | About 40 lines: the definition, the docstrings, one conjunct of the honest lemma, the toy's placement | Outside the statement. The leak limit and a device's compute limit hold of one run only through an image that is not in Lean. | the lead; lean at its read (a `domain` entry by hand, as `Site.HostsWhole`) |
| **B** | The image of a run (`Actual.Run.onto`), and every premise and the conclusion read of the image. | About 200 lines, a first guess: cells encoded again so that a cell still takes one gate a tick, the image of what is held, and that the image's acts count the run's. | Inside it. | lean and soundness: five `open` premises are then read at another object than the eleven guarantees that take them, and `CausalFrom` compares contents in the order of positions, which the new cells change. |
| **C** | The model's links premise through a map of leaves: `signal` keeps the reader's leaf, `Links` and `Alphabet` ask a link between the two link ends, and `crossing_signal` and `B4` are proved again for it. | The largest: `B4`'s chain is four files, about 1,600 lines, of another lane; how many change I have not sized. | Inside it, and the premise says what the hardware does: a link delivers to a host, which routes inside. | Daniel through the lead: it changes what an `open` premise says, or adds a second `B4`. |

## What I take

**A, now, and C named in NET-440 as not built.** A is true, says in its docstring that a host is one leaf, and costs
nothing in another lane's files. It does not meet the test of ruling 1 for NET-438 (one circuit for a device's
compute limit and this limit); only C does that honestly, and C is a ruling on the model, not a mend of this
statement. B gets the form of ruling 1 by reading every premise somewhere new, which I would not ask lean to take
on a first guess.

If the answer is C, the statement of NET-440 waits for it, and its pull request cannot open by Fri 20 Nov.

TIMER: none

