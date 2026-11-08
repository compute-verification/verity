module

public import Proofs.Flock.SpotFlock.Rows
public import Proofs.Flock.Claims.Bridge.Transports
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.FitS
public import Proofs.Flock.Soundness.Discharge.Hm.Port

/-!
# The frame opens the session's rows

`Reveals h dec` (`SpotFlock.Opens`) asks the reveal to hold, for each gate a checked unit's correctness reads, an opening
of the decoded value there that passes the deployed check. `FrameIs h L` is what the auditor's frame (`h.frame`, whose
`reveal`, `at_`, `ownAt`, `place`, `bits`, `salt` and `π` are the code's) does toward that, one opening at a time, at a
session layout `L`. At each such gate whose row the session reads (`L.rowOf … = some (j, i, p)`: table `j`'s instance
`i` in the unit's block `β`, at its circuit's port `p`), the reveal holds an opening

* at the gate, of the decoded value there (`decOf L`), under the roots `Reveals` names;
* whose row is port `p`'s row message (`Hidden.Any.rowMsg`) of the bytes the session witness's hidden values hold at
  the row's message columns (`Typed.msgCol`), and whose salt is the bytes they hold at its salt columns (`Hm.saltCol`),
  where table `j`'s column `y` in block `β` is the hidden value at `hidAt` (`SpotFlock.Rows`);
* that the frame checks at a registered read `reg` of port `p`, among those `h.registered` gives at the run's roots and
  the interior registration's: at `reg`'s root and the position the session header gives for the row, among ports
  whose first with that root is `reg`, through the path the header gives for the row.

`opens_frame`: an opening the frame checks so passes the deployed check where its row's `b ‖ c` opens `reg` at that
position through its path (`Flock.Registered.opens`), which `opens_of_stmtOk` gives for the table's row once the
opening's `b ‖ c` is the table row's. `leaf_commitString`: the deployed check's hm96 leaf is Flock's tree leaf of the
commit string `b ‖ c`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.NetworkCertifier.Bytes (bs)

/-- **hm96's leaf is Flock's tree leaf of the commit string**: the leaf of a row's bytes `r` under the salt `y` is
`Hm96.Default512.treeLeaf` of `b ‖ c`, the commit string of `r` under `y`. -/
theorem leaf_commitString (r y : List UInt8) :
    bytes (FlockSoundness.Binding.digBytes (FlockSoundness.Refine.H512 (FlockSoundness.Binding.hm96Input r y))) =
      _root_.Flock.Hm96.Default512.treeLeaf (bytes (FlockSoundness.Binding.commitString r y)) := by
  rw [bytes_digBytes, FlockSoundness.Refine.H512_val]
  show _root_.Flock.Sha512.hash ⟨(FlockSoundness.Refine.K512.leafPrefix ++
      FlockSoundness.Binding.commitString r y).toArray⟩ =
    _root_.Flock.Sha512.hash (_root_.Flock.Hm96.Default512.leafPrefix ++ bytes (FlockSoundness.Binding.commitString r y))
  congr 1

variable (h : Held)

/-- **A frame opening passes the deployed check where its row's `b ‖ c` opens the registered read it is checked at**:
the frame checks it at `reg`'s root and position `pos`, among ports whose first with that root is `reg`, and Flock's
registered read of the commit string of its row and salt there, through its path, passes. -/
theorem opens_frame {roots : List Msg} {o : Opening Bool} {reg : _root_.Flock.Registered.Port} {pos : ℕ}
    (hat : h.frame.at_ roots o.pos = some (bs reg.root, pos))
    (hfind : (h.frame.ownAt roots o.pos).find? (fun P => decide (bs P.root = bs reg.root)) = some reg)
    (hop : _root_.Flock.Registered.opens reg
      (bytes (FlockSoundness.Binding.commitString (SamplingDeploy.row h.frame.place h.frame.bits o) (h.frame.salt o)))
      pos (bytes (h.frame.π o)) = true) :
    (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 roots o = true :=
  (opens_sha512 (code h) roots o).2 ⟨(bs reg.root, pos), reg, hat, hfind, by rw [leaf_commitString]; exact hop⟩

/-- **The frame opens the session's rows**, at a session layout `L`. At the statements `Reveals` reads, the interior
registration `it`, the session statement's setup `(ins, spec, sts)` and parse `S`, the registered reads `own` that
`h.registered` gives at the run's roots and `it`'s, and the checked unit's block `β`: for each gate `g` the unit's
correctness reads whose row is table `j`'s (`st`) instance `i` at port `p`, the reveal holds an opening

* at `g`, of the decoded value there, under the interior roots of the unit's replay block where `g` is one of its
  interior gates, and under the run's roots otherwise;
* whose row is port `p`'s row message of the bytes the session witness's hidden values hold at the row's message
  columns, and whose salt is the bytes they hold at its salt columns;
* that the frame checks at a registered read `reg` of port `p`: at `reg`'s root and the position the session header
  gives for the row, among ports whose first with that root is `reg`, through the path the header gives for it. -/
def FrameIs (L : Layout) : Prop :=
  ∀ x s Pt d (m₀ m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) (rest : List (Msg × Out)) wF u b ρ rb it ins spec sts S own β,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ xS c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS wF = true →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    interiorsOf m₁ = some it → FlockDeploy.setupOf true xS = some (ins, spec, sts) →
    FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S →
    (FlockDeploy.jsonOf "registered reads" (h.registered s.roots it.roots) >>= _root_.Flock.Registered.ofJson) =
      .ok own →
    L.blockOf x m₁ xS u = some β →
    ∀ g j i p (st : _root_.Flock.Stmt), s.prog.Needs b g → L.rowOf x m₁ xS u g = some (j, i, p) →
      sts[j]? = some st →
      ∃ q ∈ h.frame.reveal s u (seenAt m₀ m₁ xS c₀ c₁ rest) wF,
        q.2.pos = g ∧ q.2.val = decOf L x m₁ xS wF u g ∧
        q.1 = (if g ∈ rb ∧ s.prog.Boundary rb g = false then iroots x (seenAt m₀ m₁ xS c₀ c₁ rest) ρ
          else s.roots) ∧
        SamplingDeploy.row h.frame.place h.frame.bits q.2 =
          FlockSoundness.Discharge.Hidden.Any.rowMsg st.c.ports[p]!
            (FlockSoundness.Discharge.Hm.bytesOf (fun y => wF.hid.getD (hidAt sts.size S.kLog j β y) false)
              (2 * st.c.ports[p]!.words) (FlockSoundness.Typed.msgCol st.c i p)) ∧
        h.frame.salt q.2 =
          FlockSoundness.Discharge.Hm.bytesOf (fun y => wF.hid.getD (hidAt sts.size S.kLog j β y) false) 192
            (FlockSoundness.Discharge.Hm.saltCol st.c i p) ∧
        ∃ reg ∈ own, ∃ t positions paths,
          (List.range st.c.ports.size).find? (fun p' => st.c.ports[p']!.name == reg.name) = some p ∧
          st.pub.tables = some t ∧ _root_.Flock.Registered.entry st.pub reg = .ok (positions, paths) ∧
          h.frame.at_ q.1 g =
            some (bs reg.root, positions.getD (_root_.Flock.Registered.ref t st.c.ports.size p (β * st.c.g + i)) 0) ∧
          (h.frame.ownAt q.1 g).find? (fun P => decide (bs P.root = bs reg.root)) = some reg ∧
          bytes (h.frame.π q.2) =
            paths.getD (_root_.Flock.Registered.ref t st.c.ports.size p (β * st.c.g + i)) ByteArray.empty

/-- **`Reveals`' opening at a gate whose row the frame opens**: at the statements `FrameIs` reads, a gate the checked
unit's correctness reads, whose row is table `j`'s (`st`) instance `i` at port `p`, where that instance is one of the
statement's and the table's row there is the commit string of the row message and salt bytes the hidden values hold,
has the opening `Reveals` asks for: it passes the deployed check, at the gate, of the decoded value, under the roots
`Reveals` names. The registered read it passes is `opens_of_stmtOk`'s, at the run's roots and the interior
registration's. -/
theorem opens_of_frameIs (L : Layout) (hF : FrameIs h L) :
    ∀ x s Pt d (m₀ m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) (rest : List (Msg × Out)) wF u b ρ rb it ins spec sts S own β,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ xS c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS wF = true →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    interiorsOf m₁ = some it → FlockDeploy.setupOf true xS = some (ins, spec, sts) →
    FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S →
    (FlockDeploy.jsonOf "registered reads" (h.registered s.roots it.roots) >>= _root_.Flock.Registered.ofJson) =
      .ok own →
    L.blockOf x m₁ xS u = some β →
    ∀ g j i p (st : _root_.Flock.Stmt), s.prog.Needs b g → L.rowOf x m₁ xS u g = some (j, i, p) →
      sts[j]? = some st → β * st.c.g + i < st.pub.n →
      (∀ t, st.pub.tables = some t →
        _root_.Flock.Registered.tableRow st.pub t p (_root_.Flock.Registered.ref t st.c.ports.size p (β * st.c.g + i)) =
          bytes (FlockSoundness.Binding.commitString
            (FlockSoundness.Discharge.Hidden.Any.rowMsg st.c.ports[p]!
              (FlockSoundness.Discharge.Hm.bytesOf (fun y => wF.hid.getD (hidAt sts.size S.kLog j β y) false)
                (2 * st.c.ports[p]!.words) (FlockSoundness.Typed.msgCol st.c i p)))
            (FlockSoundness.Discharge.Hm.bytesOf (fun y => wF.hid.getD (hidAt sts.size S.kLog j β y) false) 192
              (FlockSoundness.Discharge.Hm.saltCol st.c i p)))) →
      ∃ q ∈ (code h).reveal s u (seenAt m₀ m₁ xS c₀ c₁ rest) wF,
        (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true ∧ q.2.pos = g ∧
          q.2.val = decOf L x m₁ xS wF u g ∧
          q.1 = if g ∈ rb ∧ s.prog.Boundary rb g = false then iroots x (seenAt m₀ m₁ xS c₀ c₁ rest) ρ else s.roots := by
  intro x s Pt d m₀ m₁ xS c₀ c₁ rest wF u b ρ rb it ins spec sts S own β hx hP h2 hin hok hu hck hb hρ hrb hit hset hS
    hown hβ g j i p st hg hr hst hn hbc
  obtain ⟨q, hq, hpos, hval, hroot, hrow, hsalt, reg, hreg, t, positions, paths, hp, ht, hent, hat, hfind, hπ⟩ :=
    hF x s Pt d m₀ m₁ xS c₀ c₁ rest wF u b ρ rb it ins spec sts S own β hx hP h2 hin hok hu hck hb hρ hrb hit hset hS
      hown hβ g j i p st hg hr hst
  obtain ⟨it', ins', spec', sts', own', hit', hset', -, hown', hall⟩ := opens_of_stmtOk h hx hok
  rw [hit] at hit'
  cases hit'
  rw [hset] at hset'
  cases hset'
  rw [hown] at hown'
  cases hown'
  obtain ⟨t', idx, ht', -, hregs⟩ := hall st (Array.mem_iff_getElem?.2 ⟨j, hst⟩)
  rw [ht] at ht'
  cases ht'
  obtain ⟨p', positions', paths', hp', hent', hi⟩ := hregs reg hreg
  rw [hp] at hp'
  cases hp'
  rw [hent] at hent'
  cases hent'
  obtain ⟨-, -, hop⟩ := hi (β * st.c.g + i) hn
  refine ⟨q, hq, ?_, hpos, hval, hroot⟩
  subst hpos
  refine opens_frame h hat hfind ?_
  rw [hrow, hsalt, hπ, ← hbc t ht]
  exact hop

end SpotFlock
