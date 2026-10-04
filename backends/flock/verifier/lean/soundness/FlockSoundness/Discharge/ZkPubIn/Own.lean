import FlockSoundness.Discharge.ZkPubIn.Reg
import FlockSoundness.Discharge.ZkReg.Own

/-!
# The verifier's own registered reads, from R, with public inputs

`ZkReg.Own` at a `verify --zk --registered R --public-inputs F` session (`rports` R's ports, `own` F's bytes), on the
public plan's layout and rows (`layShHP`, `rsShHP`):

* `ownReadsHP rports`: at row `(u, q)`, R's port named as the circuit's port `q`, at the position it gives for `u`'s
  unit in the population's header (`ZkReg.ownIsRead`, `ZkReg.ownPortAt`, `ZkReg.ownReadAt`); the registrant's openings
  (`wv`, `ws`, `wpath`, `committed`) are parameters.
* `drawnOwnHP`: `ownReadsHP` at the rows the units of a draw `S` hold (`layShHP_pos_unit`: a row a unit holds is its
  own).
* `ownAtDrawsHP_of_check`: at a live outcome `(R, S, wr S R lo)` that `verify --zk --registered R --public-inputs F`
  accepts, `drawnOwnHP … S` is the verifier's own at the record's draw (`OwnReadsHP`), from the coin server's record
  (`RecordsLiveZKP`) and its draw proving `S` (`DrawOkZKHP`'s first conjunct): the check reads the units
  `Registered.tableUnits` reads, at the draw `dj S`.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Refine ZkLink ZkExec ZkBind ZkHidden HiddenExec HiddenDec DecodeZL ZkReg
open FlockSoundness.Discharge.Exec (Inputs recordDraw)

variable {n : ℕ} {I : Inputs} {own : ByteArray} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀)) {Reg : Type}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : Layout.scopeOk c = true) (hU : Built c.unit) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
  {pub : Public} (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)

/-- **The reads the verifier builds from R** (`rports`) over the population `pub`, on the public plan's layout, at the
registrant's openings `wv`, `ws`, `wpath` (`committed`: they open R's root at R's positions). -/
def ownReadsHP (rports : Array Registered.Port)
    (wv : (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).Pos →
      (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).Val)
    (ws : (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).Pos → List UInt8)
    (wpath : (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).Pos → ByteArray)
    (committed : ∀ p, ownIsRead rports c pub p →
      Flock.Registered.opensLeaf (ownPortAt rports c p)
        ((Binding.ExplicitLeaf.hm96 (rsShHP dj y₀ ht hc hscope hU hpi hpub).enc
          (rsShHP dj y₀ ht hc hscope hU hpi hpub).enc_inj).leaf (wv p) (ws p)).1 (ownReadAt rports c pub p)
        (wpath p) = true) :
    FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
      (rsShHP dj y₀ ht hc hscope hU hpi hpub) where
  isRead := ownIsRead rports c pub
  read_row p h := by
    rcases p with _ | _
    · rfl
    · exact h.elim
  port := ownPortAt rports c
  readAt := ownReadAt rports c pub
  wv := wv
  ws := ws
  wpath := wpath
  committed := committed

/-- **`ownReadsHP` at the rows the units of `S` hold** (the bridge's `restrict _ (drawnReads _ S)`). -/
def drawnOwnHP (rports : Array Registered.Port)
    (wv : (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).Pos →
      (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).Val)
    (ws : (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).Pos → List UInt8)
    (wpath : (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).Pos → ByteArray)
    (committed : ∀ p, ownIsRead rports c pub p →
      Flock.Registered.opensLeaf (ownPortAt rports c p)
        ((Binding.ExplicitLeaf.hm96 (rsShHP dj y₀ ht hc hscope hU hpi hpub).enc
          (rsShHP dj y₀ ht hc hscope hU hpi hpub).enc_inj).leaf (wv p) (ws p)).1 (ownReadAt rports c pub p)
        (wpath p) = true)
    (S : Finset (Fin n)) :
    FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
      (rsShHP dj y₀ ht hc hscope hU hpi hpub) where
  isRead p := ownIsRead rports c pub p ∧
    ∃ u ∈ S, ∃ g ∈ ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).io u,
      (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).pos g = p
  read_row p h := (ownReadsHP dj y₀ ht hc hscope hU hpi hpub rports wv ws wpath committed).read_row p h.1
  port := ownPortAt rports c
  readAt := ownReadAt rports c pub
  wv := wv
  ws := ws
  wpath := wpath
  committed p h := committed p h.1

/-- **A row a unit holds is its own, on the public plan's layout**: a committed wire of unit `u` at row `(u', q)` has
`u' = u` (`rowOfH_wire_unitP`). -/
theorem layShHP_pos_unit {u u' : Fin n} {q : Fin c.ports.size}
    {g : Fin ((keyProgP c hU n pi).circuit (Layout.outsOf c hU n)).N}
    (hg : g ∈ ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).io u)
    (hpos : (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).pos g = .inl (u', q)) : u' = u := by
  obtain ⟨x, rfl⟩ := (Const.Prog.instC (keyProgP c hU n pi) (Layout.outsOf c hU n) u).exists_wire_of_mem_io hg
  rcases hk : rowOfH c hU n (Layout.outsOf c hU n)
      ((Const.Prog.instC (keyProgP c hU n pi) (Layout.outsOf c hU n) u).wire x) with _ | ⟨u'', q'', k⟩
  · have h2 : (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).pos
        ((Const.Prog.instC (keyProgP c hU n pi) (Layout.outsOf c hU n) u).wire x) =
        .inr ((Const.Prog.instC (keyProgP c hU n pi) (Layout.outsOf c hU n) u).wire x) :=
      (Layout.ofRows_pos_none _ hk : (keyReadersHP (sitesShHP_parse (Reg := Reg) dj y₀ ht hc hscope) pi hpi own
        (Layout.outsOf c hU n) (fun _ => pub)).pos _ = _)
    rw [h2] at hpos
    cases hpos
  · have h2 : (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).pos
        ((Const.Prog.instC (keyProgP c hU n pi) (Layout.outsOf c hU n) u).wire x) = .inl (u'', q'') :=
      (Layout.ofRows_pos_some _ hk : (keyReadersHP (sitesShHP_parse (Reg := Reg) dj y₀ ht hc hscope) pi hpi own
        (Layout.outsOf c hU n) (fun _ => pub)).pos _ = _)
    rw [h2] at hpos
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Sum.inl.inj hpos)
    exact rowOfH_wire_unitP _ x hk

/-- **The verifier's own reads at an accepted live outcome, with public inputs.** At a live outcome `(R, S, wr S R lo)`
that `verify --zk --registered R --public-inputs F` accepts (`rports` R's ports, `own` F's bytes), where the coin
server records the draw `dj S` (`RecordsLiveZKP`) and that draw proves `S` (`DrawOkZKHP`'s first conjunct), the reads R
gives at the rows the drawn units hold (`drawnOwnHP`) are the verifier's own at the record's draw (`OwnReadsHP`). -/
theorem ownAtDrawsHP_of_check (rports : Array Registered.Port) (hTags : TagsOkH I)
    {wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript} (hRec : RecordsLiveZKP I own dj wr)
    {S : Finset (Fin n)} (hS : ∀ D, Flock.Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved)
    (R : Reg) (lo : LiveOut mPts) (hacc : AcceptsZKRP I rports own (wr S R lo))
    (wv : (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).Pos →
      (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).Val)
    (ws : (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).Pos → List UInt8)
    (wpath : (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).Pos → ByteArray)
    (committed : ∀ p, ownIsRead rports c pub p →
      Flock.Registered.opensLeaf (ownPortAt rports c p)
        ((Binding.ExplicitLeaf.hm96 (rsShHP dj y₀ ht hc hscope hU hpi hpub).enc
          (rsShHP dj y₀ ht hc hscope hU hpi hpub).enc_inj).leaf (wv p) (ws p)).1 (ownReadAt rports c pub p)
        (wpath p) = true) :
    OwnReadsHP rports c pi pub (wr S R lo)
      (drawnOwnHP dj y₀ ht hc hscope hU hpi hpub rports wv ws wpath committed S).isRead
      (drawnOwnHP dj y₀ ht hc hscope hU hpi hpub rports wv ws wpath committed S).port
      (drawnOwnHP dj y₀ ht hc hscope hU hpi hpub rports wv ws wpath committed S).readAt := by
  rintro p ⟨hp, u₀, hu₀, g, hg, hpos⟩
  obtain ⟨u, q, rfl, hmem, hfind, hreads⟩ := ownReads_shape hp
  have hu : u ∈ S := (layShHP_pos_unit dj y₀ ht hc hscope hU hpi (own := own) (pub := pub) hg hpos) ▸ hu₀
  obtain ⟨st, -, hck⟩ := check_of_acceptsHP hTags hacc
  obtain ⟨-, idx, -, hidx, -⟩ := check_tables hck
  obtain ⟨hd, -⟩ := hRec S R lo (acceptsZKP_of_RP hacc)
  have hidx' : Registered.tableUnits I.publicFile (some (dj S)) I.count 0 = .ok idx := by
    rw [← hd]; exact hidx
  obtain ⟨header, units, D, hh, hun, hD, -⟩ := Registered.tableUnits_one hTags.2.2.2 hidx'
  rw [loadPublic_header (loadPublic_nilP hpub)] at hh
  cases hh
  have hpop : popUnits pub = units := by
    show ((Registered.unitIndices pub.header).toOption).getD #[] = units
    rw [hun]; rfl
  rw [hpop] at hreads
  exact ⟨u, q, dj S, D, units, rfl, hmem, hfind, hd, hD, hS D hD u hu, hun, hreads⟩

end FlockSoundness.Discharge.ZkPubIn
