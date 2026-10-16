import Proofs.Flock.Claims.Complete.Json.Session
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.FailClosed.PublicTies
import Proofs.Flock.Soundness.Discharge.Exec.Setup

/-!
# What the record needs of the statement, from its setup

`stmtOk`: at every good case, the session spec names no keyed coin tree, a non-empty `Σ`, a printable draw that decodes
to the statement's instances (`DrawOk`), and the statement's draw is the one it was set up at: `StmtOk`, from the case's
setup alone, under the JSON library's specification (`Claims.JsonSpec`).

* The plain verifier's coins are the OS's or a seed's (`Raw.ins`: only `--zk` keys a coin tree), so the hello's `coins`
  entry is absent or a string, and `CoinTree.ofHello` reads none (`ofHello_none`). The scope check (`ProvedScope.check`)
  reads no coins, so it is `Raw.ins`, not the scope, that keeps a keyed coin tree out of the plain verifier's statements.
* `Σ` is a SHA-512 digest (`setupH_stmt`, `Hidden.setupHidden_spec'`).
* The public header is the population's with the draw put in (`HmRow.drawn`), so its `unit_draw` is the draw
  (`unitDraw_of`), which decodes to the statement's instances (`PublicTies.instances`).
* The draw is the statement's parsed JSON (`jsonOf`) and stratified (`ProvedScope.drawLaw`), and a stratified draw that
  decodes has exactly the keys it reads, each a string, a natural number or an array of them (`good_of_stratified`), so
  it is printable.
-/

namespace FlockComplete.JsonRT

open Flock FlockDeploy
open Lean (Json JsonNumber)
open FlockSoundness FlockSoundness.Refine

open private stratumOfJson from Flock.Draw

/-! ## The statement's inputs -/

/-- What `Raw.ins` reads: the masked verifier or not, the plain verifier's coins (the OS's or a seed's), and the draw. -/
theorem Raw.ins_read {w : Raw} {s : Ins} (h : w.ins = .ok s) :
    s.read.zk = w.zk ∧ (w.zk = false → s.read.I.coins = .os ∨ s.read.I.coins = .seed) ∧
      w.draw.mapM (jsonOf "draw") = .ok s.read.draw := by
  unfold Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact ⟨rfl, fun hz => by simp_all, by assumption⟩)

/-- The draw is JSON that `Json.parse` read. -/
theorem draw_parse {w : Raw} {draw : Option Json} (h : w.draw.mapM (jsonOf "draw") = .ok draw) {d : Json}
    (hd : draw = some d) : ∃ t, Json.parse t = .ok d := by
  subst hd
  cases hw : w.draw with
  | none => rw [hw] at h; cases h
  | some b =>
    rw [hw] at h
    simp only [Option.mapM_some] at h
    cases hj : jsonOf "draw" b with
    | error e => rw [hj] at h; cases h
    | ok d' =>
      rw [hj] at h
      cases h
      unfold jsonOf at hj
      split at hj
      · cases hj
      · rename_i t _
        cases hp : Json.parse t with
        | error e => rw [hp] at hj; cases hj
        | ok j => rw [hp] at hj; cases hj; exact ⟨t, hp⟩

/-- A statement of the plain verifier decodes to inputs that `Raw.ins` read. -/
theorem insOf_false {x : Claims.Msg} {s : Ins} (h : insOf false x = some s) :
    ∃ w : Raw, w.zk = false ∧ w.ins = .ok s := by
  unfold insOf at h
  obtain ⟨w, -, h⟩ := Option.bind_eq_some_iff.1 h
  by_cases hz : w.zk ≠ false
  · simp [hz] at h
  · refine ⟨w, not_not.1 hz, ?_⟩
    simp only [hz, ↓reduceIte] at h
    cases hi : w.ins with
    | error e => rw [hi] at h; cases h
    | ok s' => rw [hi] at h; cases h; rfl

/-! ## Objects of a known key count -/

/-- An object with `L.length` keys, the keys of `L` among them, has no other key. -/
theorem key_mem {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {L : List String} (hL : L.Nodup)
    (hin : ∀ k ∈ L, k ∈ m) (hs : m.size = L.length) {k : String} (hk : k ∈ m) : k ∈ L := by
  by_contra hn
  have hsub : k :: L ⊆ m.keys := by
    intro x hx
    rcases List.mem_cons.1 hx with rfl | hx
    · exact (Std.TreeMap.Raw.mem_keys hm).2 hk
    · exact (Std.TreeMap.Raw.mem_keys hm).2 (hin x hx)
  have hle := (List.subperm_of_subset (List.nodup_cons.2 ⟨hn, hL⟩) hsub).length_le
  rw [Std.TreeMap.Raw.length_keys hm, hs] at hle
  simp at hle

theorem mem_of_getObjVal {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {k : String} {v : Json}
    (h : (Json.obj m).getObjVal? k = .ok v) : k ∈ m ∧ m.get? k = some v := by
  simp only [Json.getObjVal?] at h
  cases hg : m.get? k with
  | none => rw [hg] at h; cases h
  | some a =>
    rw [hg] at h
    cases h
    refine ⟨(Std.TreeMap.Raw.mem_iff_isSome_getElem? hm).2 ?_, rfl⟩
    rw [← Std.TreeMap.Raw.get?_eq_getElem?, hg]
    rfl

theorem getObjVal_of_get {m : Std.TreeMap.Raw String Json compare} {k : String} {v : Json} (h : m.get? k = some v) :
    (Json.obj m).getObjVal? k = .ok v := by
  simp only [Json.getObjVal?, h]
  rfl

theorem obj_of_getObjVal {j : Json} {k : String} {v : Json} (h : j.getObjVal? k = .ok v) :
    ∃ m, j = .obj m := by
  cases j with
  | obj m => exact ⟨m, rfl⟩
  | _ => cases h

/-! ## The `J` readers' values -/

theorem getNat_ok {j : Json} {n : ℕ} (h : j.getNat? = .ok n) : j = .num ⟨Int.ofNat n, 0⟩ := by
  unfold Json.getNat? at h
  split at h
  · cases h; rfl
  · cases h

theorem asNat_ok {j : Json} {c : String} {n : ℕ} (h : J.asNat j c = .ok n) : j = .num ⟨Int.ofNat n, 0⟩ := by
  unfold J.asNat at h
  split at h
  · rename_i _ hg; cases h; exact getNat_ok hg
  · cases h

theorem asNat_good {j : Json} {c : String} {n : ℕ} (h : J.asNat j c = .ok n) : Good j := by
  rw [asNat_ok h]; exact .num _

theorem get_ok {j : Json} {k c : String} {v : Json} (h : J.get j k c = .ok v) : j.getObjVal? k = .ok v := by
  unfold J.get at h
  split at h
  · rename_i _ hg; cases h; exact hg
  · cases h

theorem nat_ok {j : Json} {k c : String} {n : ℕ} (h : J.nat j k c = .ok n) :
    j.getObjVal? k = .ok (.num ⟨Int.ofNat n, 0⟩) := by
  unfold J.nat at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  rw [get_ok hv, asNat_ok h]

theorem str_ok {j : Json} {k c : String} {s : String} (h : J.str j k c = .ok s) : j.getObjVal? k = .ok (.str s) := by
  unfold J.str at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  rw [get_ok hv]
  unfold J.asStr at h
  split at h
  · rename_i _ hg
    cases h
    unfold Json.getStr? at hg
    split at hg
    · cases hg; rfl
    · cases hg
  · cases h

theorem arr_ok {j : Json} {k c : String} {a : Array Json} (h : J.arr j k c = .ok a) :
    j.getObjVal? k = .ok (.arr a) := by
  unfold J.arr at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  rw [get_ok hv]
  unfold J.asArr at h
  split at h
  · rename_i _ hg
    cases h
    unfold Json.getArr? at hg
    split at hg
    · cases hg; rfl
    · cases hg
  · cases h

/-- Every element `mapM` maps returned. -/
theorem mapM_ok_mem {α β : Type} {f : α → V β} {a : Array α} {b : Array β} (h : a.mapM f = .ok b) :
    ∀ x ∈ a, ∃ y, f x = .ok y := by
  intro x hx
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
  obtain ⟨hs, hb⟩ := Discharge.Exec.array_mapM_ok_getElem h
  exact ⟨_, hb i hi (hs ▸ hi)⟩

/-! ## What the setup checked -/

/-- A drawn session's setup checked its draw's law (`ProvedScope.drawLaw`). -/
theorem Ins.setup_drawLaw {s : Ins} {p : SessionSpec × Array Stmt} (h : s.setup = .ok p)
    (hd : s.read.draw.isSome = true) : ProvedScope.drawLaw s.read.zk s.read.draw = .ok () := by
  unfold Ins.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | assumption

theorem hmOut_loadPublic {tags : Tags} {c : Circuit} {f : ByteArray} {pop : Public}
    (h : HmOut.loadPublic tags c f = .ok pop) : HmRow.loadPublic tags { c with outPorts := #[] } f = .ok pop := by
  unfold HmOut.loadPublic at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    assumption

set_option maxHeartbeats 4000000 in
/-- What `Stmt.setupH` set up: its public file loaded (at a draw, `HmRow.drawn` of it) and passed `HmRow.checkPublic`,
its instances, Σ and its hello. -/
theorem setupH_stmt {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) :
    (∃ pop, HmRow.loadPublic tags st.c pf = .ok pop ∧ ∀ d, draw = some d → HmRow.drawn st.c pop d = .ok st.pub) ∧
      HmRow.checkPublic st.c st.pub = .ok () ∧ st.n = st.pub.n ∧
      st.sigma = Sha512.hash (tags.sigmaTag.toUTF8 ++ st.c.sha ++ st.digest ++ st.pub.sha) ∧
      Flock.helloOf tags st.sigma (PT_LOCAL + st.nbl) coins = .ok st.hello := by
  cases draw <;> unfold Stmt.setupH at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨⟨_, by assumption, fun d hd => by first | (cases hd; done) | (cases hd; assumption)⟩, by assumption, rfl,
      rfl, by assumption⟩

open Discharge in
/-- **What a table's setup set up** (`setupOne`, either output mode): the public file a public-file load returned, at a
draw that file's `HmRow.drawn`; the header's draw decodes to the statement's instances; a SHA-512 Σ; and the hello. -/
theorem setupOne_stmt {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}
    (h : HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st) :
    (∃ c pop, HmRow.loadPublic tags c pf = .ok pop ∧ ∀ d, draw = some d → HmRow.drawn c pop d = .ok st.pub) ∧
      (∀ d, st.pub.header.getObjVal? "unit_draw" = .ok d →
        ∃ u, Draw.UnitDraw.ofJson d = .ok u ∧ u.proved.size = st.n) ∧
      0 < st.sigma.size ∧ Flock.helloOf tags st.sigma (PT_LOCAL + st.nbl) coins = .ok st.hello := by
  rcases Bool.eq_false_or_eq_true tags.hiddenOutputs with hh | hh
  · rw [HiddenExec.setupOne_hidden hh] at h
    have S := Hidden.setupHidden_spec' h
    obtain ⟨pop, hl, -, hd⟩ := S.pub
    have T := FailClosed.ties_of_check (hidden := true) (Or.inr ⟨rfl, S.checkPublic⟩)
    refine ⟨⟨_, pop, hmOut_loadPublic hl, hd⟩, fun d hd' => ?_, by rw [S.sigma, Sha512.hash_size]; omega, S.hello⟩
    obtain ⟨u, hu, -, hn⟩ := T.instances d hd'
    exact ⟨u, hu, by rw [S.n, hn]⟩
  · rw [HiddenExec.setupOne_public hh] at h
    obtain ⟨⟨pop, hl, hd⟩, hc, hn, hs, hhello⟩ := setupH_stmt h
    have T := FailClosed.ties_of_check (hidden := false) (Or.inl ⟨rfl, hc⟩)
    refine ⟨⟨_, pop, hl, hd⟩, fun d hd' => ?_, by rw [hs, Sha512.hash_size]; omega, hhello⟩
    obtain ⟨u, hu, -, hn'⟩ := T.instances d hd'
    exact ⟨u, hu, by rw [hn, hn']⟩

end FlockComplete.JsonRT
