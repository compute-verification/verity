module

public import Proofs.Flock.SessionLoop.Concrete.Scope
public import Proofs.Flock.SessionLoop.Concrete.Regions
public import Proofs.Flock.Claims.ZK.MaskedTables

/-! # The concrete statement is `Live`

`live_concreteStmt`: where the verifier's two fixed nets build (`FixedNetsBuild`), the concrete statement is in the
masked session's scope with a witness (`Masked.live_of`), so its session has one table (`jOf_concreteStmt`). The witness
`wC` is the trace `Trace.zC` bit for bit. The deployed relation reads the statement's one table (`relS_relation`), which
the trace satisfies, its rows, pin and public regions included (`Trace.satisfies_regions_stW`).
-/

@[expose] public section

namespace FlockDeploy

open Flock FlockSoundness RecDeploy.VStarNets FlockVBridge.Names
open Properties.Flock.Nets (sparseOf FixedNetsBuild)

/-- The concrete statement's one table. -/
noncomputable def stC : Stmt :=
  Concrete.stW (HmNets.shaLaid ()) (HmNets.hmLaid ()) concreteHash (sparseOf (HmNets.shaLaid ()).ra)
    (sparseOf (HmNets.shaLaid ()).rb) (sparseOf (HmNets.hmLaid ()).ra) (sparseOf (HmNets.hmLaid ()).rb) Concrete.s1C
    Concrete.s3C (Concrete.s4Of concreteHash) Concrete.s6C Concrete.helloC

/-- The concrete statement, parsed: its inputs' readable fields and its one table. -/
noncomputable def pC : _root_.Flock.Readable.Parsed :=
  { Concrete.rC (HmNets.shaLaid ()) (HmNets.hmLaid ()) concreteHash with sts := #[stC] }

/-- **The concrete witness**: the trace `Trace.zC`, one hidden value per bit. -/
noncomputable def wC : _root_.Claims.Flock.Wit :=
  ⟨List.ofFn fun a : Fin (2 ^ stC.m) => Concrete.Trace.zIn (a.val % 2 ^ 22)⟩

theorem parseBytes_concreteStmt (hn : FixedNetsBuild) :
    _root_.Flock.Readable.parseBytes (readOf true) concreteStmt = some pC := by
  have hp := fixedNets_parse hn
  have hpar := Concrete.parse_rC (HmNets.shaLaid ()) (HmNets.hmLaid ()) concreteHash (sparseOf (HmNets.shaLaid ()).ra)
    (sparseOf (HmNets.shaLaid ()).rb) (sparseOf (HmNets.hmLaid ()).ra) (sparseOf (HmNets.hmLaid ()).rb) Concrete.s1C
    Concrete.s3C (Concrete.s4Of concreteHash) Concrete.s6C Concrete.helloC rfl (sha_lines hn) (hm_lines hn) hp.1.2
    hp.2.2 Concrete.hc1_of Concrete.hc2_of (fun c h1 h2 => Concrete.check_of hp.1.1 hp.2.1 c h1 h2) Concrete.hu1_of
    (Concrete.hu2_of _ _ _ _ _ _ _ _) (Concrete.hd_of _) Concrete.hid_of Concrete.hhello_of
  unfold _root_.Flock.Readable.parseBytes readOf
  rewrite [concreteStmt_ins hn]
  exact hpar

theorem zOf_wC : _root_.Flock.Readable.zOf 1 wC 0 stC.m = Concrete.Trace.zC stC.m := by
  funext a
  simp only [_root_.Flock.Readable.zOf, wC, Nat.mul_one, Nat.add_zero, List.getD_eq_getElem?_getD,
    List.getElem?_ofFn, a.isLt, dite_true, Option.getD_some, _root_.Flock.Readable.toZ, Concrete.Trace.zC]

/-- **The concrete statement is in the deployed relation** with the trace as its witness. -/
theorem rel_concreteStmt (hn : FixedNetsBuild) :
    _root_.Claims.Flock.R (deployZk _root_.Flock.Readable.relation).parse concreteStmt wC := by
  have ht : Concrete.tz.typed = false := rfl
  have hp := fixedNets_parse hn
  have hset := Concrete.setup_stW (HmNets.shaLaid ()) (HmNets.hmLaid ()) concreteHash
    (sparseOf (HmNets.shaLaid ()).ra) (sparseOf (HmNets.shaLaid ()).rb) (sparseOf (HmNets.hmLaid ()).ra)
    (sparseOf (HmNets.hmLaid ()).rb) Concrete.s1C Concrete.s3C (Concrete.s4Of concreteHash) Concrete.s6C
    Concrete.helloC rfl (sha_lines hn) (hm_lines hn) hp.1.2 hp.2.2 Concrete.hc1_of Concrete.hc2_of
    (fun c h1 h2 => Concrete.check_of hp.1.1 hp.2.1 c h1 h2) Concrete.hu1_of (Concrete.hu2_of _ _ _ _ _ _ _ _)
    (Concrete.hd_of _) Concrete.hid_of Concrete.hhello_of
  have L : Layout stC := setupH_layout ht hset
  have hS := Concrete.Trace.satisfies_regions_stW (HmNets.shaLaid ()) (HmNets.hmLaid ()) concreteHash
    (sparseOf (HmNets.shaLaid ()).ra) (sparseOf (HmNets.shaLaid ()).rb) (sparseOf (HmNets.hmLaid ()).ra)
    (sparseOf (HmNets.hmLaid ()).rb) Concrete.s1C Concrete.s3C (Concrete.s4Of concreteHash) Concrete.s6C
    Concrete.helloC ht hset L hn
  refine ⟨_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation pC, ?_, ?_⟩
  · change (_root_.Flock.Readable.parseBytes (readOf true) concreteStmt).map
      (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation) = _
    rw [parseBytes_concreteStmt hn]
    rfl
  refine (_root_.Flock.Readable.relS_relation pC wC).2 fun c hc => ?_
  have h1 : pC.sts.size = 1 := rfl
  obtain rfl : c = 0 := by omega
  have hz : _root_.Flock.Readable.zOf pC.sts.size wC 0 pC.sts[0]!.m = Concrete.Trace.zC stC.m := zOf_wC
  refine ⟨⟨rfl, rfl, (L.model []).hk6, (L.model []).hk, (L.model []).pin.isLt⟩, ?_⟩
  rw [hz]
  exact (hS : (L.model (Refine.regionsOf stC)).Satisfies (Concrete.Trace.zC stC.m))

/-- **The concrete statement is `Live`** where the fixed nets build. -/
theorem live_concreteStmt (hn : FixedNetsBuild) : _root_.Claims.ZKMal.Masked.Live concreteStmt :=
  _root_.Claims.ZKMal.Masked.live_of (concreteStmt_scope hn) (rel_concreteStmt hn)

/-- **The concrete statement's masked session has one table.** -/
theorem jOf_concreteStmt (hn : FixedNetsBuild) : _root_.Claims.ZKMal.Masked.jOf concreteStmt = 1 := by
  unfold _root_.Claims.ZKMal.Masked.jOf
  rw [ite_eq_left (live_concreteStmt hn)]
  unfold _root_.Claims.ZKMal.Masked.stsOf
  rw [concreteStmt_setupOf hn]
  exact concreteSetup_tables

end FlockDeploy
