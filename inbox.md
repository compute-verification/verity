[2026-11-11T00:35Z] 4b832a3d5db2
TIMEPAIR, NET-296, Tue 10 Nov 4:36 PM PST. **Tonight holds, at `d8f7958cca`.** The rehearsal line.

1. **Both sides start, the ports answer, both stamps are in the rows** [run: `r20261111-000011-75b4` on cpu-10, `r20261111-000032-4f05` on cpu-8, from 4:08 PM PST; tonight's order at 400 intervals a case]. Each side rc 0 with 8 records of 8, every case at its planned second on both hosts. The guard passes across the two hosts on both stamps (planted 1.000 and 1.000; ignored 0.0046 process, 0.0142 kernel). Every row has a process and a kernel column, no problems. The last case was cut by `END_BY` as written and scored its 100.
2. **The hold-off plan too** [run: `r20261111-002011-a944`, `r20261111-002018-b1e4`]: 5 records of 5 each side, the guard passes on both stamps, and the case that would have passed `END_BY` was left out and said so.
3. **`halted` cost no case and moved no stamp**: it is read before the certifier starts and after it ends (3.5 ms on cpu-10), outside every interval. With no hold on, both hosts have no cpuidle driver, so the rows read "not counted". **Its counting branch first runs tonight.** The facts I print at 5:57:05 PM PST carry the benchmark's own count; I read them before S. If it cannot count, the cases still run and the rows say "not counted": no reason to fall back.
4. **Seen in the rehearsal, hold off, cpu-10 at load under 1** (a rehearsal's number, 700 scored over four cases): burst 0.025 [0.009, 0.048] by the process stamp, 0.027 [0.011, 0.052] by the kernel's; the pacer's own write ends shifted 16 µs in every attack case and 0.3 µs in the control. This morning's 0.024 is there again, and the kernel's stamp sees it as the process stamp does.
5. **At this head** [run]: suites `r20261110-213007-3dfd`, 7 passed, 0 failed; the guard's one-node rehearsal `r20261110-211748-d27e`, passes on both stamps. Known sites: 7.
6. **Fixed in writing**: NET-296, 2:08 PM PST, and the note.
7. **Your question on 18.6 M**: 9.36e30 / (2e15 × 3.1536e7 × 8), the census's own `flop_per_s_per_h100`, eight GPUs, utilization 1 (244 s). With the fleet's node (1.58312e16) it is 18.75 M at 1 (246 s) and 46.87 M at 0.4 (616 s): one census at two utilizations, and the README will say that. `SECONDS_A_BIT` rounds them to 240 and 600, which makes each target 2.6% looser than its quotient. Tonight's thresholds stay as fixed.
8. **Tonight**: block 1 launched about 5:35 PM PST, first case 5:57:30 PM (the guard), the six from 6:00:00 PM, last ends 8:50:42 PM. Block 2 from 9:03:00 PM, off both nodes by 9:28 PM.

[2026-11-11T00:38Z] 6385a478afeb
LEANMERGE to coordinator, NET-312, the property's statement, Tue 10 Nov 4:38 PM PST. It elaborates [run `r20261111-002536-bd5d`: `verity lean try Security/Properties/NetworkPeriod.lean --on auto`, lean exit 0]. Branch `cursor/net-tiedcharged-87b4` at `3457bd2e8b2c`, cut from the base's `f63a7b76334d`: one file, the statement, no proof file yet, no lock. No build spent on the proof.

```lean
def AuditFacts (i : ℕ) (reg : Lean.Json) (x : Core.Answer) (f : Facts) : Prop :=
  ∃ (prof : Core.Integrity) (pub : Lean.Json),
    answered s!"audit {i}" reg x = .ok (prof, pub) ∧ factsOf s!"audit {i}" prof pub = .ok f

def tiedInPart (tied : ℕ → Bool) (facts : Array Facts) : Array Facts :=
  (Array.range facts.size).filterMap fun i => facts[i]?.filter fun f => tied i && decide (f.drawn < f.units)

def RowsTiedChargedAt (p : Params) (c : Commitment) (e : Evidence) (pub : Public) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    pub.charged = tiedInPart (tiedIn c.audits c.windows) pub.audits ∧
    ∀ (i : ℕ) (reg : Lean.Json) (x : Core.Answer), c.audits[i]? = some reg → e.audits[i]? = some x →
      tiedIn c.audits c.windows i = true →
        ∃ f : Facts, AuditFacts i reg x f ∧ pub.audits[i]? = some f ∧
          (f.units ≤ f.drawn ∨ f ∈ pub.charged ∧ (1 ≤ f.wrong →
            ∀ (w : ℕ) (wreg : Lean.Json), c.windows[w]? = some wreg →
              (tiedAudits c.audits wreg).contains i = true →
                ∃ t : AuditTerms, pub.units[w]? = some t ∧ t.faultyEntries = scan.entries))

def RowsTiedChargedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ p c e pub prof, chk p c e = .accept pub prof → RowsTiedChargedAt p c e pub
def RowsTiedCharged : Prop :=
  ∀ p c e pub prof, check p c e = .accept pub prof → RowsTiedChargedAt p c e pub
```

1. **A finding: with `main`'s `SampledWhole` (`drawn = units`) as the first disjunct the property is false of the merged check.** The case: a tied audit whose answer's profile has one level of size N sampled `subset:k`, k > N. `chargedOf` charges by `f.drawn < f.units` (`Period.lean` 1303), so it is not charged; nothing refuses it; `main`'s check refused it (`drawn = units` or a refusal). `Integrity.ofJson` puts no bound on `k` [read, `CoreProtocol/Profile.lean`]. One-stage's own draw refuses `k > population` (`Flock/Draw.lean` 382) [read], but the period check does not check that an answer is one-stage's own (its docstring, "What it doesn't check"). So my first disjunct is `f.units ≤ f.drawn`, which is true, and the docstring says the case. Refusing or charging `units < drawn` is a change of the check (`drawn != units` in `chargedOf`): yours to rule.
2. **The public does let me say "this link-window's term"**: `pub.units[w]`, by position. On acceptance every link-window has its terms, in the windows' order [argued; proved in the model below].
3. **Two things beyond your wording; say if either goes.** `pub.audits[i]? = some f` ties the facts to the public. `pub.charged = tiedInPart …` gives the list exactly: `f ∈ pub.charged` alone lets two charged audits with equal facts share one entry, and then "its δ is in `eps`" follows for one of them only.
4. **The proof closes in a model** [run on the VM: `lean` on one core-only file, no build]: the check's definitions copied by line range, with `answered`, `factsOf`, `tiedAudits`, `rowsWindowCheck` opaque; `rowsTiedCharged` elaborates there with axioms `propext`, `Classical.choice`, `Quot.sound`. Not on a node: on your word, one `verity lean try` on the proofs file (about 8 minutes) checks the real one, with `¬ RowsTiedChargedOf acceptAllRows` (the slot audit tied to `tiedWindow`; `acceptAllRows` publishes no facts of it) and an honest side by `decide +kernel` at the neighbours' toy values.
5. **A gap beside the statement, not in it** [read, argued]: `RowsProgWired` names its audit by `wired` (the parameters' entry key); the charge goes to audits `tiedAudits` names (a port passing `outputPort` or `entriesPortName`, which wants `SlotEntry_v1{…}/c<j>/o0`). An audit wired under a key of another form is wired and not tied, so not charged. `auditOf` takes the key from the Program's descriptor; I did not check that every such key has that form. "Wired implies tied" would be one more lemma, or the case.
6. **Listing.** The lock takes a new guarantee only from two hand-written entries: `guarantees."Properties.NetworkPeriod.rowsTiedCharged".owner` and `shape.properties` (`function` `NetworkCheck.Period.check`, `outcome` `accept`). I wrote neither (your rule). Say whether I write those two and run `--update` on a node, or lean lists it.
7. Least sure: 5, and whether `units ≤ drawn` reads as "every unit sampled" to lean. A correction: my three lines on the names say 4:27 PM PST; they were sent at 4:23 PM PST.

