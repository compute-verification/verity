import Proofs.Flock.Claims.Complete.Json.Session
import Proofs.Flock.Claims.Complete.Json.Canon
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.FailClosed.PublicTies
import Proofs.Flock.Soundness.Discharge.Exec.Setup

/-!
# What the record needs of the statement, from its setup

`stmtOk`: at every good case, the session spec names no keyed coin tree, a non-empty `Σ`, a printable ASCII draw that
decodes to the statement's instances (`DrawOk`), and the statement's draw is the one it was set up at: `StmtOk`, from the
case's setup alone, with no premise on the JSON library: the verifier keeps what it parsed rebuilt (`Flock.norm`), and the coin
tree its coins name.

* The plain verifier's coins are the OS's or a seed's (`Raw.ins`: only `--zk` keys a coin tree), and the statement
  keeps the coins' tree (`Stmt.coinTree`, `setupOne_stmt`), so it names none (`tree?_none`). The scope check
  (`ProvedScope.check`) reads no coins, so it is `Raw.ins`, not the scope, that keeps a keyed coin tree out of the plain
  verifier's statements.
* `Σ` is a SHA-512 digest (`setupH_stmt`, `Hidden.setupHidden_spec'`).
* The public header is the population's, rebuilt (`loadPublic_header`), with the draw put in (`HmRow.drawn`), so its
  `unit_draw` is the draw (`unitDraw_of`), which decodes to the statement's instances (`PublicTies.instances`).
* The draw is the statement's parsed JSON (`jsonOf`) and stratified or, at hidden outputs, a subset (the setup's
  `ProvedScope.drawLaw`, `Ins.setup_drawLaw`, `law_of_lawWhy`), and a stratified or subset draw that decodes has exactly
  the keys it reads, each a string, a natural number or an array of them (`good_stratified`, `good_subset`), so it is
  printable.
-/

namespace FlockComplete.JsonRT

open Flock FlockDeploy
open Lean (Json JsonNumber)
open FlockSoundness FlockSoundness.Refine

open private stratumOfJson from Flock.Draw

/-! ## The statement's inputs -/

/-- What `Raw.ins` reads: the masked verifier or not, the plain verifier's coins (the OS's or a seed's), and the
draw. -/
theorem Raw.ins_read {w : Raw} {s : Ins} (h : w.ins = .ok s) :
    s.read.zk = w.zk ∧ (w.zk = false → s.read.I.coins = .os ∨ s.read.I.coins = .seed) ∧
      ∃ draw, w.draw.mapM (jsonOf "draw") = .ok draw ∧ s.read.draw = draw.map Flock.norm := by
  refine ⟨Raw.ins_zk h, ?_, Raw.ins_draw h⟩
  unfold Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact fun hz => by simp_all)

/-- The draw is well formed: the verifier rebuilt it (`Flock.norm`, `wf_norm`). -/
theorem draw_wf {w : Raw} {draw : Option Json}
    (h : ∃ dr, w.draw.mapM (jsonOf "draw") = .ok dr ∧ draw = dr.map Flock.norm) {d : Json} (hd : draw = some d) :
    Claims.Json.Wf d := by
  obtain ⟨dr, -, rfl⟩ := h
  obtain ⟨d₀, -, rfl⟩ := Option.map_eq_some_iff.1 hd
  exact wf_norm d₀

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
    (hd : s.read.draw.isSome = true) :
    ProvedScope.drawLaw s.read.zk s.read.tags false s.read.draw = .ok () := by
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
      Flock.helloOf tags st.sigma (PT_LOCAL + st.nbl) coins = .ok st.hello ∧ st.coinTree = coins.tree? := by
  cases draw <;> unfold Stmt.setupH at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨⟨_, by assumption, fun d hd => by first | (cases hd; done) | (cases hd; assumption)⟩, by assumption, rfl,
      rfl, by assumption, rfl⟩

set_option maxHeartbeats 4000000 in
/-- `Stmt.setupHidden`'s statement keeps the coins' tree. -/
theorem setupHidden_coinTree {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : st.coinTree = coins.tree? := by
  cases draw <;> unfold Stmt.setupHidden at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl

open Discharge in
/-- **What a table's setup set up** (`setupOne`, either output mode): the public file a public-file load returned, at a
draw that file's `HmRow.drawn`; the header's draw decodes to the statement's instances; a SHA-512 Σ; and the hello. -/
theorem setupOne_stmt {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}
    (h : HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st) :
    (∃ c pop, HmRow.loadPublic tags c pf = .ok pop ∧ ∀ d, draw = some d → HmRow.drawn c pop d = .ok st.pub) ∧
      (∀ d, st.pub.header.getObjVal? "unit_draw" = .ok d →
        ∃ u, Draw.UnitDraw.ofJson d = .ok u ∧ u.proved.size = st.n) ∧
      0 < st.sigma.size ∧ Flock.helloOf tags st.sigma (PT_LOCAL + st.nbl) coins = .ok st.hello ∧
      st.coinTree = coins.tree? := by
  rcases Bool.eq_false_or_eq_true tags.hiddenOutputs with hh | hh
  · rw [HiddenExec.setupOne_hidden hh] at h
    have hct := setupHidden_coinTree h
    have S := Hidden.setupHidden_spec' h
    obtain ⟨pop, hl, -, hd⟩ := S.pub
    have T := FailClosed.ties_of_check (hidden := true) (Or.inr ⟨rfl, S.checkPublic⟩)
    refine ⟨⟨_, pop, hmOut_loadPublic hl, hd⟩, fun d hd' => ?_, by rw [S.sigma, Sha512.hash_size]; omega, S.hello, hct⟩
    obtain ⟨u, hu, -, hn⟩ := T.instances d hd'
    exact ⟨u, hu, by rw [S.n, hn]⟩
  · rw [HiddenExec.setupOne_public hh] at h
    obtain ⟨⟨pop, hl, hd⟩, hc, hn, hs, hhello, hct⟩ := setupH_stmt h
    have T := FailClosed.ties_of_check (hidden := false) (Or.inl ⟨rfl, hc⟩)
    refine ⟨⟨_, pop, hl, hd⟩, fun d hd' => ?_, by rw [hs, Sha512.hash_size]; omega, hhello, hct⟩
    obtain ⟨u, hu, -, hn'⟩ := T.instances d hd'
    exact ⟨u, hu, by rw [hn, hn']⟩

/-! ## The public header's draw -/

set_option maxHeartbeats 4000000 in
/-- The public file's header is a rebuilt object (`Flock.norm` of what `Json.parse` read). -/
theorem loadPublic_header {tags : Tags} {c : Circuit} {f : ByteArray} {pop : Public}
    (h : HmRow.loadPublic tags c f = .ok pop) : (∃ j, pop.header = Flock.norm j) ∧ ∃ m, pop.header = .obj m := by
  unfold HmRow.loadPublic at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hpj := ‹(pure _ : V Json) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq] at hpj
    subst hpj
    exact ⟨⟨_, rfl⟩, obj_of_getObjVal (str_ok (k := "format") (by assumption))⟩

set_option maxHeartbeats 4000000 in
/-- `HmRow.drawn` puts the draw in the population's header. -/
theorem drawn_header {c : Circuit} {pub pub' : Public} {d : Json} {m : Std.TreeMap.Raw String Json compare}
    (hm : m.WF) (hp : pub.header = .obj m) (h : HmRow.drawn c pub d = .ok pub') :
    pub'.header.getObjVal? "unit_draw" = .ok d := by
  unfold HmRow.drawn at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    dsimp only
    rw [hp]
    simp only [Json.setObjVal!]
    apply getObjVal_of_get
    rw [Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_insert hm.insert.insert.insert,
      Std.TreeMap.Raw.getElem?_insert hm.insert.insert, Std.TreeMap.Raw.getElem?_insert hm.insert,
      Std.TreeMap.Raw.getElem?_insert hm]
    simp

/-- **The statement's header names its draw**: a draw's statement over a loaded public file has that draw as its
`unit_draw`. -/
theorem unitDraw_of {tags : Tags} {c : Circuit} {pf : ByteArray}
    {pop pub : Public} {d : Json} (hl : HmRow.loadPublic tags c pf = .ok pop) (hd : HmRow.drawn c pop d = .ok pub) :
    pub.header.getObjVal? "unit_draw" = .ok d := by
  obtain ⟨⟨j, hj⟩, m, hm⟩ := loadPublic_header hl
  have hw := FlockDeploy.wf_norm j
  rw [← hj, hm] at hw
  cases hw with
  | obj hwf _ => exact drawn_header hwf hm hd

set_option maxHeartbeats 4000000 in
/-- `HmRow.drawn` prints the header it puts the draw in (`publicDigest`'s `canon`), so the draw is ASCII. -/
theorem drawn_ascii {c : Circuit} {pub pub' : Public} {d : Json} {m : Std.TreeMap.Raw String Json compare}
    (hm : m.WF) (hp : pub.header = .obj m) (h : HmRow.drawn c pub d = .ok pub') : Ascii d := by
  have hu := drawn_header hm hp h
  unfold HmRow.drawn at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hc := ‹HmRow.publicDigest _ _ = Except.ok _›
    unfold HmRow.publicDigest at hc
    obtain ⟨_, hs, -⟩ := ebind_ok hc
    have ha := ascii_of_canon hs
    try dsimp only at hu
    rw [hp] at hu ha
    simp only [Json.setObjVal!] at hu ha
    exact ascii_getObjVal hm.insert.insert.insert.insert ha hu

/-- **A draw's statement over a loaded public file has an ASCII draw**: the setup prints the header it is in. -/
theorem ascii_of_drawn {tags : Tags} {c : Circuit} {pf : ByteArray} {pop pub : Public} {d : Json}
    (hl : HmRow.loadPublic tags c pf = .ok pop) (hd : HmRow.drawn c pop d = .ok pub) : Ascii d := by
  obtain ⟨⟨j, hj⟩, m, hm⟩ := loadPublic_header hl
  have hw := FlockDeploy.wf_norm j
  rw [← hj, hm] at hw
  cases hw with
  | obj hwf _ => exact drawn_ascii hwf hm hd

/-! ## The coins name no coin tree -/

theorem good_nil : ∀ p ∈ ([] : List (String × Json)), Good p.2 := fun _ h => by cases h

theorem good_cons {k : String} {v : Json} {l : List (String × Json)} (hv : Good v) (hl : ∀ p ∈ l, Good p.2) :
    ∀ p ∈ (k, v) :: l, Good p.2 := by
  intro p hp
  rcases List.mem_cons.1 hp with rfl | hp
  · exact hv
  · exact hl p hp

theorem good_append {l₁ l₂ : List (String × Json)} (h₁ : ∀ p ∈ l₁, Good p.2) (h₂ : ∀ p ∈ l₂, Good p.2) :
    ∀ p ∈ l₁ ++ l₂, Good p.2 := by
  intro p hp
  rcases List.mem_append.1 hp with hp | hp
  · exact h₁ p hp
  · exact h₂ p hp

theorem good_arr_strs (a : Array String) : Good (Json.arr (a.map Json.str)) :=
  good_arr_of fun x hx => by
    obtain ⟨y, -, rfl⟩ := Array.mem_map.1 hx
    exact .str _

/-- **The plain verifier's coins name no coin tree**: the OS's coins or a seed's (`Coins.tree?`), which the statement
keeps as its `coinTree`. -/
theorem tree?_none {coins : Coins} (hc : coins = .os ∨ coins = .seed) : coins.tree? = none := by
  rcases hc with rfl | rfl <;> rfl

/-! ## A stratified draw that decodes is printable -/

theorem getObj_ok {j : Json} {m : Std.TreeMap.Raw String Json compare} (h : j.getObj? = .ok m) : j = .obj m := by
  cases j with
  | obj m' => cases h; rfl
  | _ => cases h

theorem getStr_ok {j : Json} {s : String} (h : j.getStr? = .ok s) : j = .str s := by
  cases j with
  | str s' => cases h; rfl
  | _ => cases h

theorem isOk_ok {j : Json} {k : String} (h : (j.getObjVal? k).isOk = true) : ∃ v, j.getObjVal? k = .ok v := by
  cases e : j.getObjVal? k with
  | ok v => exact ⟨v, rfl⟩
  | error _ => rw [e] at h; cases h

/-- **An object of known keys is printable**: one with `L.length` keys, each of `L` holding a printable value. -/
theorem good_obj_of {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {L : List String} (hL : L.Nodup)
    (hs : m.toArray.size = L.length) (hk : ∀ k ∈ L, ∃ v, (Json.obj m).getObjVal? k = .ok v ∧ Good v) :
    Good (.obj m) := by
  refine .obj hm fun k v hv => ?_
  have hv' := getObjVal_of_get hv
  have hkL : k ∈ L := key_mem hm hL (fun k hk' => by
      obtain ⟨v, hv, -⟩ := hk k hk'
      exact (mem_of_getObjVal hm hv).1)
    (by rw [← Std.TreeMap.Raw.size_toArray hm]; exact hs) (mem_of_getObjVal hm hv').1
  obtain ⟨v', hv'', hg⟩ := hk k hkL
  rw [hv'] at hv''
  cases hv''
  exact hg

theorem good_of_asNat_mapM {a : Array Json} {c : String} {b : Array ℕ} (h : a.mapM (J.asNat · c) = .ok b) :
    Good (.arr a) :=
  .arr fun x hx => by
    obtain ⟨_, hy⟩ := mapM_ok_mem h x hx
    exact asNat_good hy

/-- A stratum's `[lo, hi)` range is a pair of natural numbers. -/
theorem good_range {p : Array Json} (hs : p.size = 2) {c₀ c₁ : String} {lo hi : ℕ} (h₀ : J.asNat p[0]! c₀ = .ok lo)
    (h₁ : J.asNat p[1]! c₁ = .ok hi) : Good (.arr p) := by
  refine .arr fun x hx => ?_
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
  have : i = 0 ∨ i = 1 := by omega
  rcases this with rfl | rfl
  · rw [← getElem!_pos p 0 hi]; exact asNat_good h₀
  · rw [← getElem!_pos p 1 hi]; exact asNat_good h₁

set_option maxHeartbeats 4000000 in
/-- **A stratum that decodes is printable**: an object of exactly `template`, `units` and `k`, a string, ranges of two
natural numbers, and a natural number. -/
theorem good_stratum {j : Json} (hw : Wf j) {s : Draw.Stratum} (h : stratumOfJson j = .ok s) : Good j := by
  unfold stratumOfJson at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hu : ∀ r ∈ _, Good r := fun r hr => by
      obtain ⟨y, h⟩ := mapM_ok_mem ‹Array.mapM _ _ = Except.ok _› r hr
      try dsimp only at h
      try simp only [throw_bind'] at h
      repeat' walk_step
      all_goals
        exact good_range (by simpa using ‹¬ (_ != 2) = true›) ‹J.asNat _[0]! _ = _› ‹J.asNat _[1]! _ = _›
    have hj := getObj_ok ‹j.getObj? = _›
    subst hj
    cases hw with
    | obj hm _ =>
    have hc := ‹¬ (_ != 3 || _) = true›
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not, List.any_cons, List.any_nil, Bool.or_false,
      Bool.not_eq_true', Bool.not_eq_false] at hc
    obtain ⟨hs, ht, hun, hk⟩ := hc
    refine good_obj_of hm (L := ["template", "units", "k"]) (by decide) (by simpa using hs) ?_
    simp only [List.mem_cons, List.mem_nil_iff, or_false]
    rintro key (rfl | rfl | rfl)
    · obtain ⟨v, hv⟩ := isOk_ok ht
      refine ⟨v, hv, ?_⟩
      have hs' := ‹(Json.obj _).getObjValAs? String "template" = _›
      simp only [Json.getObjValAs?, Json.getObjValD, hv, Except.toOption, Option.getD_some] at hs'
      rw [getStr_ok hs']
      exact .str _
    · exact ⟨_, ‹(Json.obj _).getObjVal? "units" = _›, .arr hu⟩
    · obtain ⟨v, hv⟩ := isOk_ok hk
      refine ⟨v, hv, ?_⟩
      have hn := ‹J.asNat _ "U2: a stratum's k" = _›
      simp only [hv, Except.toOption, Option.getD_some] at hn
      exact asNat_good hn

set_option maxHeartbeats 4000000 in
/-- **A stratified draw that decodes is printable**: an object of exactly `law`, `population`, `units`, `k` and
`strata`, a string, natural numbers, an array of them, and strata that decode. -/
theorem good_stratified {j : Json} (hw : Wf j) {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson j = .ok D) {K : ℕ}
    {strata : Array Draw.Stratum} (hl : D.law = .stratified K strata) : Good j := by
  unfold Draw.UnitDraw.ofJson at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
  all_goals
    have hp := ‹(pure _ : V Draw.Law) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq] at hp
  all_goals
    try simp only [reduceCtorEq] at hl
  all_goals
    have hj := getObj_ok ‹j.getObj? = _›
    subst hj
    cases hw with
    | obj hm hwv =>
    have hc := ‹¬ (_ != 5 || _ || _) = true›
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at hc
    have hss := ‹(Json.obj _).getObjVal? "strata" = Except.ok (Json.arr _)›
    refine good_obj_of hm (L := ["law", "population", "units", "k", "strata"]) (by decide) (by simpa using hc.1.1) ?_
    simp only [List.mem_cons, List.mem_nil_iff, or_false]
    rintro key (rfl | rfl | rfl | rfl | rfl)
    · exact ⟨_, str_ok ‹J.str _ "law" _ = _›, .str _⟩
    · exact ⟨_, nat_ok ‹J.nat _ "population" _ = _›, .num _⟩
    · exact ⟨_, arr_ok ‹J.arr _ "units" _ = _›, good_of_asNat_mapM ‹Array.mapM _ _ = Except.ok _›⟩
    · exact ⟨_, nat_ok ‹J.nat _ "k" _ = _›, .num _⟩
    · refine ⟨_, hss, .arr fun x hx => ?_⟩
      obtain ⟨_, hy⟩ := mapM_ok_mem ‹Array.mapM stratumOfJson _ = Except.ok _› x hx
      have hw' := hwv "strata" _ (mem_of_getObjVal hm hss).2
      cases hw' with
      | arr ha => exact good_stratum (ha x hx) hy

set_option maxHeartbeats 4000000 in
/-- **A subset draw that decodes is printable**: an object of exactly `law`, `population`, `units` and `k`, a string,
natural numbers and an array of them. -/
theorem good_subset {j : Json} (hw : Wf j) {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson j = .ok D) {k : ℕ}
    (hl : D.law = .subset k) : Good j := by
  unfold Draw.UnitDraw.ofJson at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
  all_goals
    have hp := ‹(pure _ : V Draw.Law) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq] at hp
  all_goals
    try simp only [reduceCtorEq] at hl
  all_goals
    have hj := getObj_ok ‹j.getObj? = _›
    subst hj
    cases hw with
    | obj hm hwv =>
    have hc := ‹¬ (_ != 4) = true›
    simp only [bne_iff_ne, ne_eq, not_not] at hc
    refine good_obj_of hm (L := ["law", "population", "units", "k"]) (by decide) (by simpa using hc) ?_
    simp only [List.mem_cons, List.mem_nil_iff, or_false]
    rintro key (rfl | rfl | rfl | rfl)
    · exact ⟨_, str_ok ‹J.str _ "law" _ = _›, .str _⟩
    · exact ⟨_, nat_ok ‹J.nat _ "population" _ = _›, .num _⟩
    · exact ⟨_, arr_ok ‹J.arr _ "units" _ = _›, good_of_asNat_mapM ‹Array.mapM _ _ = Except.ok _›⟩
    · exact ⟨_, nat_ok ‹J.nat _ "k" _ = _›, .num _⟩

/-- A draw a soundness claim draws by is stratified or a subset (a subset only where `lawWhy`'s form covers it). -/
theorem law_of_lawWhy {zk : Bool} {tags : Tags} {D : Draw.UnitDraw} (h : ProvedScope.lawWhy zk tags false D = none) :
    (∃ K strata, D.law = .stratified K strata) ∨ ∃ k, D.law = .subset k := by
  revert h
  cases hl : D.law with
  | stratified K s => exact fun _ => Or.inl ⟨K, s, rfl⟩
  | subset k => exact fun _ => Or.inr ⟨k, rfl⟩
  | _ => simp [ProvedScope.lawWhy, hl]

/-! ## The statement at a good case -/

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **What the record needs of the statement**, at every good case: no keyed coin tree, a non-empty `Σ`, a printable
draw that decodes to the statement's instances, and the statement's draw the one it was set up at. -/
def StmtOk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ c : Case H rel, c.spec.coinTree = none ∧ 0 < c.spec.sigma.size ∧ DrawOk c.spec ∧ c.spec.unitDraw = c.s.read.draw

/-- **The case's one table**: its statement is `setupOne`'s, at the statement's tags, coins and draw. -/
theorem _root_.FlockComplete.Case.one (c : Case H rel) :
    ∃ st, c.sts = #[st] ∧ c.spec = st.spec ∧
      Discharge.HiddenExec.setupOne c.s.read.tags c.s.read.I.circuitFile c.s.read.I.publicFile c.s.read.coins
        c.s.read.I.tables c.s.read.draw c.s.read.I.partition c.s.read.I.program = .ok st := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployPlain rel) rel (readOf false) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  have hi := (setupOf_some c.setup).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  obtain ⟨hc, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hset := Ins.setup_read (setupOf_some c.setup).2
  rw [← ht] at hset
  unfold Readable.Read.setup at hset
  rw [hpi] at hset
  rcases Discharge.HiddenExec.setupTables_okAny hset with ⟨-, st, h1, hs, hsts⟩ | ⟨hc1, hsz, -⟩
  · refine ⟨st, hsts, hs, ?_⟩
    rw [← ht]
    exact h1
  · obtain ⟨st0, hsts0, -⟩ := c.table
    rw [hsts0] at hsz
    simp only [List.size_toArray, List.length_cons, List.length_nil, zero_add] at hsz
    omega

/-- **What the record needs of the statement holds at every good case** (`StmtOk`): the plain verifier's coins name no
coin tree, Σ is a SHA-512 digest, the draw is printable and decodes to
the statement's instances, and the session's draw is the statement's. -/
theorem stmtOk : StmtOk H rel := by
  intro c
  obtain ⟨st, -, hspec, h1⟩ := c.one
  obtain ⟨⟨c', pop, hl, hd⟩, hinst, hsig, -, hct⟩ := setupOne_stmt h1
  obtain ⟨w, hwz, hw⟩ := insOf_false (setupOf_some c.setup).1
  obtain ⟨hzk, hcoins, hdraw⟩ := Raw.ins_read hw
  have hz : c.s.read.zk = false := hzk.trans hwz
  obtain ⟨d, hdd⟩ := Option.isSome_iff_exists.1 c.ready_parts.1
  have hu := unitDraw_of hl (hd d hdd)
  have hU : c.spec.unitDraw = some d := by
    rw [hspec]
    simp [Stmt.spec, hu, Except.toOption]
  refine ⟨?_, ?_, ?_, by rw [hU, hdd]⟩
  · rw [hspec]
    change st.coinTree = none
    rw [hct]
    refine tree?_none ?_
    simp only [Readable.Read.coins, hz, Bool.false_eq_true, ↓reduceIte]
    exact hcoins hwz
  · rw [hspec]
    exact hsig
  · unfold DrawOk
    rw [hU]
    dsimp only
    obtain ⟨D, hD, hn⟩ := hinst d hu
    have hdl := Ins.setup_drawLaw (setupOf_some c.setup).2 c.ready_parts.1
    rw [hdd, hz] at hdl
    simp only [ProvedScope.drawLaw, hD] at hdl
    have hlw : ProvedScope.lawWhy false c.s.read.tags false D = none := by
      cases e : ProvedScope.lawWhy false c.s.read.tags false D with
      | none => rfl
      | some _ => rw [e] at hdl; cases hdl
    have hg : Good d := by
      rcases law_of_lawWhy hlw with ⟨K, strata, hK⟩ | ⟨k, hk⟩
      · exact good_stratified (draw_wf hdraw hdd) hD hK
      · exact good_subset (draw_wf hdraw hdd) hD hk
    refine ⟨fun hn' => ?_, hg, ascii_of_drawn hl (hd d hdd), D, hD, by rw [hspec]; exact hn⟩
    rw [hn'] at hg
    rw [hn'] at hD
    cases hD

end FlockComplete.JsonRT
