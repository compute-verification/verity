module

public import Proofs.Flock.SpotFlock.RowsOpen
public import Proofs.Flock.Claims.Bridge.Transports

/-!
# The frame opens the session's rows

`FrameRows h L` (`SpotFlock.RowsOpen`), from which `reveals_of_frameRows` gives `Reveals`, asks the reveal to hold, for
each value a checked unit needs, an opening of the witness's bit at the layout's wire that passes wherever the
registered read of the session row holding it passes. `FrameIs h L` says what the auditor's frame (`h.frame`: its
`reveal`, how the code reads openings out of a Flock witness, and its `at_`, `ownAt`, `place`, `bits`, `salt` and `π`,
the deployed check's) does to that end, one opening at a time. At the statements and layout data `FrameRows` reads, the
opening at gate `g`, of the witness's bit at the wire's column, under the roots `Reveals` names:

* has as row and salt row `q` of unit `n` of table `j` as the witness holds it (`InnerRows.rowBytes`,
  `InnerRows.salt`);
* is checked at the registered read `reg` that the table's port `q` is named like: at `reg`'s root and the position the
  session header gives for the row, among ports whose first with that root is `reg`, through the path the header gives
  for it.

`frameRows_of_frameIs`: `FrameIs` gives `FrameRows`. Such an opening passes the deployed check wherever its row's
`b ‖ c` opens `reg` there (`opens_frame`), and its `b ‖ c` is the commit string `FrameRows` reads
(`InnerRows.commitString_data`). `leaf_commitString`: the deployed check's hm96 leaf is Flock's tree leaf of `b ‖ c`.

`sessionOpensTwo_of_frameIs`: so `SessionOpensTwo` at SPOT[Flock]'s two-stage code follows from two premises on the
registration, `Embeds` and `FrameIs` (`sessionOpensTwo_of_rows`).
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

/-- **The frame opens the session's rows**, at a session layout `L`: at the statements `Reveals` reads and the
session's setup, with registered ports `own`, for each gate `g` the unit's correctness reads, the layout's wire for `g`
is column `y` of table `j` in the unit's block `β`, and the reveal holds an opening at `g` of the witness's bit there,
under the roots `Reveals` names, whose row and salt are row `q` of a unit `n` of the table as the witness holds it, and
which the frame checks at the registered read `reg` that the table's port `q` is named like: at `reg`'s root and the
position the session header gives for the row, among ports whose first with that root is `reg`, through the path the
header gives for it. -/
def FrameIs (L : Layout) : Prop :=
  ∀ x s Pt d (m₀ m₁ m₂ : Msg) (c₀ c₁ : FlockDeploy.Coin) (rest : List (Msg × Out)) wF u b ρ rb ins spec sts own,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts) → ins.registered = some own →
    ∀ g, s.prog.Needs b g → ∃ β j y n q reg, L.blockOf x m₁ (unitStmt h m₂ u) u = some β ∧ ∃ hj : j < sts.size,
      y < 2 ^ sts[j].c.kLog ∧ L.wire x m₁ (unitStmt h m₂ u) u g = j * 2 ^ sts[j].c.kLog + y ∧ n < sts[j].pub.n ∧
      reg ∈ own ∧
      (List.range sts[j].c.ports.size).find? (fun q => sts[j].c.ports[q]!.name == reg.name) = some q ∧
      ∃ o ∈ h.frame.reveal s u (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF, o.2.pos = g ∧
        o.2.val = _root_.Flock.InnerRows.bitAt { ins.read with sts } wF j β y ∧
        o.1 = (if g ∈ rb ∧ s.prog.Boundary rb g = false then iroots x (seenAt m₀ m₁ m₂ c₀ c₁ rest) ρ
          else s.roots) ∧
        SamplingDeploy.row h.frame.place h.frame.bits o.2 =
          _root_.Flock.InnerRows.rowBytes { ins.read with sts } wF j n q ∧
        h.frame.salt o.2 = (_root_.Flock.InnerRows.salt { ins.read with sts } wF j n q).data.toList ∧
        ∀ t positions paths, sts[j].pub.tables = some t →
          _root_.Flock.Registered.entry sts[j].pub reg = .ok (positions, paths) →
          h.frame.at_ o.1 g =
            some (bs reg.root, positions.getD (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) 0) ∧
          (h.frame.ownAt o.1 g).find? (fun P => decide (bs P.root = bs reg.root)) = some reg ∧
          bytes (h.frame.π o.2) = paths.getD (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) ByteArray.empty

/-- **`FrameRows` from `FrameIs`**: an opening whose row and salt are the session row's, which the frame checks at the
row's registered read, passes the deployed check wherever that read opens the row's commit string. -/
theorem frameRows_of_frameIs (L : Layout) (hF : FrameIs h L) : FrameRows h L := by
  intro x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb ins spec sts own hx hP h2 hin hok hu hck hb hρ hrb hset hown g hg
  obtain ⟨β, j, y, n, q, reg, hβ, hj, hy, hw, hn, hreg, hq, o, ho, hpos, hval, hroot, hrow, hsalt, hchk⟩ :=
    hF x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb ins spec sts own hx hP h2 hin hok hu hck hb hρ hrb hset hown g hg
  refine ⟨β, j, y, n, q, reg, hβ, hj, hy, hw, hn, hreg, hq, o, ho, hpos, hval, hroot,
    fun t positions paths ht hent hop => ?_⟩
  obtain ⟨hat, hfind, hπ⟩ := hchk t positions paths ht hent
  subst hpos
  have hb : ∀ B : ByteArray, bytes B.data.toList = B := fun B => bytes_bs B
  refine opens_frame h hat hfind ?_
  rw [hrow, hsalt, _root_.Flock.InnerRows.commitString_data, hb, hπ]
  exact hop

/-- **`SessionOpensTwo` at SPOT[Flock]'s two-stage code from a layout** where the session statement holds each checked
unit's gates (`Embeds`) and the frame opens the session's rows (`FrameIs`). -/
theorem sessionOpensTwo_of_frameIs (L : Layout) (he : Embeds h L) (hF : FrameIs h L) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo_of_rows h L he (frameRows_of_frameIs h L hF)

end SpotFlock
