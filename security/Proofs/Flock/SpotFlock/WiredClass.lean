module

public import Proofs.Flock.SpotFlock.WiredRegX
public import Proofs.Flock.SpotFlock.OpensWhole
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.Claims.Sampling.Deployed

/-!
# What an accepted session reveals at the wired program holds the classes of its key whole

At the members' programs wired by `ω` (`codeWired`), a session reveals, for each gate `i` a proof block needs side by
side (`needsSide`), the frame's opening of `i` at the gate `ω i`, under the roots the wired program opens `ω i` under
(`revealWired`). This file proves that such a reveal holds the classes of a key of the gates whole, as `code h`'s does
(`reveal_key_of_passes`), without asking that every opening it holds passes:

* `needsWired`, `gateKeyWired`: the gates a proof block needs at the wired program (the `ω` image of `needsSide`), and
  a gate's key there (the proof blocks that need it, the replay block that holds it and whether it is on that block's
  boundary in the wired program);
* `mem_revealWired_pos`: each opening of the reveal is at a gate the proof block needs, under that gate's roots;
* `opens_of_needsWired`: at an accepted session of a checked unit, under `BuiltWired` and `AddressesWired`, every gate
  the proof block needs has an opening in the reveal that passes the commitment's check (`CheckedRun.reveals_wired`);
* `rootsAtWired_key`: gates with one key open under the same roots;
* `reveal_key_wired`: the three together. An opening of the reveal has, at every gate with its gate's key, an opening
  of the same reveal at that gate, under the same roots, that passes;
* `revealsLe_wired`: the wired code's reveal count bounds what a session reveals.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart)

variable (h : Held) (ω : ℕ → ℕ) (keep : ℕ → Bool)

/-- **The gates unit `u`'s proof block needs opened at the wired program**: `ω` of each gate it needs side by side
(`needsSide`), the positions its reveal opens (`revealWired`). -/
def needsWired (s : AuditStatement) (u : ℕ) : Finset ℕ := ((needsSide h s u).map ω).toFinset

/-- **A gate's key at the wired program**, at the wired partition `Pt` and design `d`: the proof blocks that need it
opened there (`needsWired`), the replay block that holds it (`gateBlock`), and whether it is on that replay block's
boundary in the wired program. -/
noncomputable def gateKeyWired (s : AuditStatement) (Pt : Partition) (d : Design.Two) (g : ℕ) :
    Finset ℕ × Option ℕ × Bool :=
  ((Finset.range Pt.blocks.length).filter fun v => g ∈ needsWired h ω s v, gateBlock d.part g,
    (gateBlock d.part g).any fun ρ => ((bitsProgs h.ms).wired ω keep).Boundary (d.part.blocks.getD ρ []) g)

/-- **Each opening the reveal at the wired program holds is at a gate the proof block needs there, under the roots
the wired program opens that gate under.** -/
theorem mem_revealWired_pos {f : Code} {s : AuditStatement} {u : ℕ} {v : Seen} {wF : Claims.Flock.Wit}
    {q : List Msg × Opening Bool} (hq : q ∈ revealWired h ω keep f s u v wF) :
    q.2.pos ∈ needsWired h ω s u ∧ q.1 = rootsAtWired h ω keep s u v q.2.pos := by
  unfold revealWired at hq
  obtain ⟨⟨i, o⟩, hio, rfl⟩ := List.mem_map.1 hq
  exact ⟨List.mem_toFinset.2 (List.mem_map.2 ⟨i, (List.of_mem_zip hio).1, rfl⟩), rfl⟩

/-- A unit that needs a gate opened at the wired program is one of the members' partition's. -/
theorem lt_of_mem_needsWired {s : AuditStatement} {u g : ℕ} (hg : g ∈ needsWired h ω s u) :
    u < (membersPart h.ms).blocks.length := by
  obtain ⟨i, hi, -⟩ := List.mem_map.1 (List.mem_toFinset.1 hg)
  exact Reveal.lt_of_mem_needs (partition h) (s := { s with prog := bitsProgs h.ms }) rfl hi

/-- Gates with one key at the wired program are needed by the same units there. -/
theorem gateKeyWired_needs {s : AuditStatement} {Pt : Partition} {d : Design.Two}
    (hlen : Pt.blocks.length = (membersPart h.ms).blocks.length) {g g' : ℕ}
    (hk : gateKeyWired h ω keep s Pt d g = gateKeyWired h ω keep s Pt d g') {u : ℕ}
    (hg : g ∈ needsWired h ω s u) : g' ∈ needsWired h ω s u := by
  have h1 : ((Finset.range Pt.blocks.length).filter fun v => g ∈ needsWired h ω s v) =
      (Finset.range Pt.blocks.length).filter fun v => g' ∈ needsWired h ω s v :=
    congrArg Prod.fst hk
  have hmem : u ∈ (Finset.range Pt.blocks.length).filter fun v => g ∈ needsWired h ω s v :=
    Finset.mem_filter.2 ⟨Finset.mem_range.2 (lt_of_lt_of_eq (lt_of_mem_needsWired h ω hg) hlen.symm), hg⟩
  rw [h1] at hmem
  exact (Finset.mem_filter.1 hmem).2

/-- Gates with one key at the wired program lie in one replay block. -/
theorem gateKeyWired_block {s : AuditStatement} {Pt : Partition} {d : Design.Two} {g g' : ℕ}
    (hk : gateKeyWired h ω keep s Pt d g = gateKeyWired h ω keep s Pt d g') :
    gateBlock d.part g = gateBlock d.part g' :=
  congrArg (fun k : Finset ℕ × Option ℕ × Bool => k.2.1) hk

/-- **Gates with one key at the wired program lie on one side of each replay block's boundary there**, where the
replay blocks are disjoint. -/
theorem gateKeyWired_side {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hnd : d.part.gates.Nodup)
    {g g' : ℕ} (hk : gateKeyWired h ω keep s Pt d g = gateKeyWired h ω keep s Pt d g') {ρ : ℕ} {rb : List ℕ}
    (hrb : d.part.blocks[ρ]? = some rb) (hg : g ∈ rb ∧ ((bitsProgs h.ms).wired ω keep).Boundary rb g = false) :
    g' ∈ rb ∧ ((bitsProgs h.ms).wired ω keep).Boundary rb g' = false := by
  have hgb : gateBlock d.part g = some ρ := gateBlock_of_mem hnd hrb hg.1
  have hgb' : gateBlock d.part g' = some ρ := (gateKeyWired_block h ω keep hk).symm.trans hgb
  have h3 : ((gateBlock d.part g).any fun ρ => ((bitsProgs h.ms).wired ω keep).Boundary (d.part.blocks.getD ρ []) g) =
      ((gateBlock d.part g').any fun ρ => ((bitsProgs h.ms).wired ω keep).Boundary (d.part.blocks.getD ρ []) g') :=
    congrArg (fun k : Finset ℕ × Option ℕ × Bool => k.2.2) hk
  simp only [hgb, hgb', Option.any_some] at h3
  obtain ⟨hr', hg'⟩ := gateBlock_some hgb'
  have hrbeq : d.part.blocks.getD ρ [] = rb := Option.some.inj (hr'.symm.trans hrb)
  rw [hrbeq] at h3 hg'
  exact ⟨hg', h3.symm.trans hg.2⟩

/-- **Gates with one key at the wired program open under the same roots there**, at a unit of the wired partition
whose replay block the wired design names. -/
theorem rootsAtWired_key {s : AuditStatement} {Pt : Partition} {d : Design.Two}
    (hP : partitionWired h ω s = some Pt) (hd : twoWired h ω s = some d) (hnd : d.part.gates.Nodup)
    (hrep : ∀ v < Pt.blocks.length, (replayOf Pt d.part v).isSome) {u : ℕ} (hu : u < Pt.blocks.length) (v : Seen)
    {g g' : ℕ} (hk : gateKeyWired h ω keep s Pt d g = gateKeyWired h ω keep s Pt d g') :
    rootsAtWired h ω keep s u v g = rootsAtWired h ω keep s u v g' := by
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hrep u hu)
  obtain ⟨rb, hrb⟩ := replay_block hρ
  have hP' : partitionWired h ω { s with prog := (bitsProgs h.ms).wired ω keep } = some Pt := hP
  have hd' : twoWired h ω { s with prog := (bitsProgs h.ms).wired ω keep } = some d := hd
  exact Reveal.rootsAt_same_side (partitionWired h ω) (twoWired h ω) (iroots []) hP' hd' hρ hrb
    ⟨gateKeyWired_side h ω keep hnd hk hrb, gateKeyWired_side h ω keep hnd hk.symm hrb⟩

/-- **The reveal at the wired program, at SPOT's own frame there**: for each gate `i` the proof block needs side by
side, the address's opening of `i` at position `ω i`, under the roots the wired program opens `ω i` under. -/
theorem revealWired_eq (hB : BuiltWired h ω keep) (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) :
    revealWired h ω keep h.frame s u v wF = (needsSide h s u).map fun i =>
      (rootsAtWired h ω keep s u v (ω i),
        ({ addressOp h (addressMapWired h ω keep) { s with prog := bitsProgs h.ms } u v wF i [] with pos := ω i } :
          Opening Bool)) := by
  obtain ⟨c₀, hc⟩ := hB
  have hG : h.frame.reveal { s with prog := bitsProgs h.ms } u v wF = (needsSide h s u).map fun i =>
      (rootsAtWired h ω keep { s with prog := bitsProgs h.ms } u v (ω i),
        ({ addressOp h (addressMapWired h ω keep) { s with prog := bitsProgs h.ms } u v wF i [] with pos := ω i } :
          Opening Bool)) := by
    rw [hc]
    rfl
  unfold revealWired
  rw [hG, map_zip_map_self] <;> rfl

/-- **The reveal at the wired program reads only a transcript's messages**, at SPOT's own frame there. -/
theorem revealWired_out (hB : BuiltWired h ω keep) (s : AuditStatement) (u : ℕ) (ms : List (Msg × Out))
    (o o' : Out) (wF : Claims.Flock.Wit) :
    revealWired h ω keep h.frame s u ⟨ms, o⟩ wF = revealWired h ω keep h.frame s u ⟨ms, o'⟩ wF := by
  have hroots : ∀ g, rootsAtWired h ω keep s u ⟨ms, o⟩ g = rootsAtWired h ω keep s u ⟨ms, o'⟩ g := fun _ => rfl
  have hop : ∀ i, addressOp h (addressMapWired h ω keep) { s with prog := bitsProgs h.ms } u ⟨ms, o⟩ wF i [] =
      addressOp h (addressMapWired h ω keep) { s with prog := bitsProgs h.ms } u ⟨ms, o'⟩ wF i [] := fun _ => rfl
  rw [revealWired_eq h ω keep hB, revealWired_eq h ω keep hB] <;> simp only [hroots, hop]

/-- **Every gate a checked unit's proof block needs at the wired program has an opening in the reveal that passes**,
at every run: under `BuiltWired` and `AddressesWired`, where `ω` rewires the members' programs, at a statement in the
two-stage scope, a unit `u` the two stages check, whose session's view at the run accepted, and a witness the proof
system's check accepts at the session's statement (`CheckedRun.reveals_wired`). -/
theorem opens_of_needsWired (hB : BuiltWired h ω keep) (hA : AddressesWired h ω keep)
    (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} (hx : parse h x = some s) (hP : partition h s = some Pt) (hd : two h s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω) (ζ : Ω) (r : Fin ((V h).k x) → FlockDeploy.Coin) {u : ℕ}
    (hu : u ∈ (code h).checked x r) {xS : Msg} {vS : Seen}
    (hsess : session h u (run FlockDeploy.sha512 P (V h) x ζ r).seen = some (xS, vS)) (hacc : vS.out = .acc)
    {wF : Claims.Flock.Wit}
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS wF = true) {g : ℕ}
    (hg : g ∈ needsWired h ω s u) :
    ∃ q ∈ revealWired h ω keep h.frame s u (run FlockDeploy.sha512 P (V h) x ζ r).seen wF,
      (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true ∧ q.2.pos = g := by
  obtain ⟨m₀, m₁, m₂, rest, hm, hok, hxS⟩ := session_acc_run h FlockDeploy.sha512 P x ζ r hsess hacc
  rw [hxS] at hck
  obtain ⟨st, hst⟩ : ∃ st, decode x = some st := by
    cases hdx : decode x with
    | none => simp [SpotFlock.parse, hdx] at hx
    | some st => exact ⟨st, rfl⟩
  have hum : u ∈ h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (stageSrc (stageCoins h x r)) :=
    (congrArg (u ∈ ·) (checked_eq h x hst r)).mp hu
  have hlt : u < Pt.blocks.length := by
    have := mem_checked_lt hum
    rwa [units_base h hx hP] at this
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hin.2.2.2.1 u hlt)
  obtain ⟨rb, hrb⟩ := replay_block hρ
  obtain ⟨ins, spec, sts, hset⟩ :
      ∃ ins spec sts, FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts) := by
    have hs := sessionOk_of_stmtOkAt h (stmtOkAt_unit h hx hok hum)
    simp only [sessionOk, Bool.and_eq_true] at hs
    obtain ⟨ins, spec, sts, hset, -⟩ := zkOk_sound hs.1.1.1
    exact ⟨ins, spec, sts, hset⟩
  have hr : CheckedRun.Ok h ⟨x, s, m₀, m₁, m₂, r ⟨0, k_pos h x⟩, stageCoins h x r, rest, wF, u, Pt.blocks[u], ρ, rb,
      ins, sts⟩ :=
    ⟨Pt, d, spec, hx, hP, hd, hin, hok, hum, hck, List.getElem?_eq_getElem hlt, hρ, hrb, hset⟩
  obtain ⟨i, hi, rfl⟩ := List.mem_map.1 (List.mem_toFinset.1 hg)
  obtain ⟨q, hq, hop, hpos, -, -⟩ := CheckedRun.reveals_wired hr hB hA hW ((CheckedRun.mem_needsSide hr).1 hi)
  have hseen : (run FlockDeploy.sha512 P (V h) x ζ r).seen =
      ⟨(seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest).msgs,
        (run FlockDeploy.sha512 P (V h) x ζ r).out⟩ :=
    congrArg (fun l => (⟨l, (run FlockDeploy.sha512 P (V h) x ζ r).out⟩ : Seen)) hm
  refine ⟨q, ?_, hop, hpos⟩
  rw [hseen, revealWired_out h ω keep hB s u _ _ .acc wF]
  exact hq

/-- **An accepted session's reveal at the wired program holds the classes of the wired key whole**: under
`BuiltWired` and `AddressesWired`, where `ω` rewires the members' programs, at a statement in the two-stage scope with
its design `d`, a unit `u` the two stages check, whose session's view at the run accepted, and a witness the proof
system's check accepts at the session's statement, an opening of the reveal at the wired program has, at every gate
`g'` with its gate's key there (at the wired partition and design), an opening of the same reveal at `g'`, under the
same roots, that passes the commitment's check. -/
theorem reveal_key_wired (hB : BuiltWired h ω keep) (hA : AddressesWired h ω keep)
    (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep) {x : Msg} {s : AuditStatement} {d : Design.Two}
    (hx : parse h x = some s) (hd : two h s = some d) (hin : InScopeTwo (membersPart h.ms) d s)
    (hnd : (d.part.cut ω).gates.Nodup)
    (hrep : ∀ v < ((membersPart h.ms).cut ω).blocks.length,
      (replayOf ((membersPart h.ms).cut ω) (d.part.cut ω) v).isSome)
    {Ω : Type} (P : Prover Ω) (ζ : Ω) (r : Fin ((V h).k x) → FlockDeploy.Coin) {u : ℕ}
    (hu : u ∈ (code h).checked x r) {xS : Msg} {vS : Seen}
    (hsess : session h u (run FlockDeploy.sha512 P (V h) x ζ r).seen = some (xS, vS)) (hacc : vS.out = .acc)
    {wF : Claims.Flock.Wit}
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS wF = true) :
    ∀ p ∈ revealWired h ω keep h.frame s u (run FlockDeploy.sha512 P (V h) x ζ r).seen wF, ∀ g' : ℕ,
      gateKeyWired h ω keep s ((membersPart h.ms).cut ω) ({ d with part := d.part.cut ω } : Design.Two) p.2.pos =
          gateKeyWired h ω keep s ((membersPart h.ms).cut ω) ({ d with part := d.part.cut ω } : Design.Two) g' →
        ∃ p' ∈ revealWired h ω keep h.frame s u (run FlockDeploy.sha512 P (V h) x ζ r).seen wF,
          p'.1 = p.1 ∧ p'.2.pos = g' ∧ (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 p'.1 p'.2 = true := by
  intro p hp g' hk
  obtain ⟨hpn, hp1⟩ := mem_revealWired_pos h ω keep hp
  have hlen : ((membersPart h.ms).cut ω).blocks.length = (membersPart h.ms).blocks.length :=
    Partition.length_blocks_cut (membersPart h.ms) ω
  have hg' : g' ∈ needsWired h ω s u := gateKeyWired_needs h ω keep hlen hk hpn
  obtain ⟨q, hq, hop, hpos⟩ := opens_of_needsWired h ω keep hB hA hW hx rfl hd hin P ζ r hu hsess hacc hck hg'
  have hq1 := (mem_revealWired_pos h ω keep hq).2
  have hPw : partitionWired h ω s = some ((membersPart h.ms).cut ω) := rfl
  have hdw : twoWired h ω s = some ({ d with part := d.part.cut ω } : Design.Two) :=
    congrArg (Option.map fun d : Design.Two => ({ d with part := d.part.cut ω } : Design.Two)) hd
  refine ⟨q, hq, ?_, hpos, hop⟩
  rw [hq1, hp1, hpos]
  exact (rootsAtWired_key h ω keep hPw hdw hnd hrep (lt_of_lt_of_eq (lt_of_mem_needsWired h ω hpn) hlen.symm) _
    hk).symm

/-- **The wired code's reveal count bounds what a session reveals**: the reveal at the wired program is no longer
than the list of gates the proof block needs side by side (`length_revealWired_le`), which the side-by-side count
bounds (`Reveal.length_needs_le`). -/
theorem revealsLe_wired : RevealsLe (codeWired h ω keep) := by
  intro x s hx u v wF
  have hx' : (parse h x).map
      (fun s : AuditStatement => ({ s with prog := (bitsProgs h.ms).wired ω keep } : AuditStatement)) = some s := hx
  obtain ⟨s₀, hs₀, rfl⟩ := Option.map_eq_some_iff.1 hx'
  have e : ({ s₀ with prog := bitsProgs h.ms } : AuditStatement) = s₀ :=
    congrArg (fun p : Prog => ({ s₀ with prog := p } : AuditStatement)) (parse_prog h hs₀).symm
  have hn : (needsSide h { s₀ with prog := (bitsProgs h.ms).wired ω keep } u).length ≤
      Reveal.reveals (partition h) (parse h) x := by
    show (Reveal.needs (partition h) { s₀ with prog := bitsProgs h.ms } u).length ≤ _
    rw [e]
    exact Reveal.length_needs_le (partition h) (parse h) hs₀ u
  exact (length_revealWired_le h ω keep h.frame { s₀ with prog := (bitsProgs h.ms).wired ω keep } u v wF).trans hn

end SpotFlock
