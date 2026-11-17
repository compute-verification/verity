module

public import Properties.InferenceOnly.Opening
public import Properties.InferenceOnly.PinnedRead
public import Properties.Verity.Sha512
public import Flock.Hm96
public import Proofs.ResourceAccounting.ReadUnit
public import Proofs.ResourceAccounting.ReadUnit.BitsPorts
public import Proofs.Properties.InferenceOnly.RecordForm

/-! A pinned reader's unit returns its table's row (`pinnedReadBinds`), a lemma no lock lists; it is on the row
returned, not on the opened ports, reads no row the opening declares, and gives no file, record and transcript that meet
its hypotheses (`PinnedRead`), whose witnesses `PinnedReadHonest` and `PinnedReadNontrivial` are not proved. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Defn)
open ResourceAccounting

theorem gatesOf_some {N k : ℕ} {gate : ℕ → Option ℕ} {f : Fin k → Fin N} (h : gatesOf N k gate = some f)
    (i : Fin k) : gate i.val = some (f i).val := by
  unfold gatesOf at h
  by_cases hg : ∀ i : Fin k, ∃ g : Fin N, gate i.val = some g.val
  · rw [dite_eq_left hg] at h
    obtain rfl := Option.some.inj h
    exact Classical.choose_spec (hg i)
  · rw [dite_eq_right hg] at h
    cases h

theorem gatesOf_at {N k : ℕ} {gate : ℕ → Option ℕ} {f : Fin k → Fin N} (h : gatesOf N k gate = some f) (i : ℕ)
    (hi : i < k) : gate i = some (f ⟨i, hi⟩).val :=
  gatesOf_some h ⟨i, hi⟩

theorem gatesOf_eq {N k : ℕ} {gate : ℕ → Option ℕ} {f : Fin k → Fin N} (h : gatesOf N k gate = some f)
    (at_ : ℕ → Fin N) (col : ℕ → ℕ) (hat : ∀ b < k, ∀ g, gate b = some g → (at_ (col b)).val = g) :
    f = fun i => at_ (col i.val) :=
  funext fun i => Fin.ext (hat i.val i.isLt _ (gatesOf_some h i)).symm

theorem portsAtGates_spec {C : Verity.Circuit} {inG outG : String → ℕ → Option ℕ} {ro : Roles} {out : String}
    {count words : ℕ} {q : ReadUnit.Ports C} (h : portsAtGates C inG outG ro out count words = some q) :
    (∀ i : Fin 32, inG ro.key i.val = some (q.tok i).val) ∧
      (∀ i : Fin 512, inG ro.root i.val = some (q.root i).val) ∧
      (∀ i : Fin 512, inG ro.domain i.val = some (q.domain i).val) ∧
      (∀ i : Fin (16 * q.words), outG out i.val = some (q.h i).val) ∧
      outG ro.verdict 0 = some q.ok.val ∧
      ro.opened.length = 3 ∧
      (∀ i : Fin (16 * q.words), inG (ro.opened.getD 0 "") i.val = some (q.row i).val) ∧
      (∀ i : Fin (8 * Hash.Hm96.SALT_BYTES), inG (ro.opened.getD 1 "") i.val = some (q.salt i).val) ∧
      ∀ (l : Fin (Hash.Tree.depth q.count)) (i : Fin 512),
        inG (ro.opened.getD 2 "") (512 * l.val + i.val) = some (q.path l i).val := by
  unfold portsAtGates at h
  by_cases hlen : ro.opened.length = 3
  · rw [ite_eq_left hlen] at h
    obtain ⟨tok, htok, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨root, hroot, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨domain, hdomain, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨rowG, hrow, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨saltG, hsalt, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨pathG, hpath, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨hG, hh, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨okG, hok, h⟩ := Option.map_eq_some_iff.1 h
    subst h
    exact ⟨gatesOf_some htok, gatesOf_some hroot, gatesOf_some hdomain, gatesOf_some hh,
      gatesOf_at hok 0 Nat.one_pos, hlen, gatesOf_some hrow, gatesOf_some hsalt,
      fun l i => gatesOf_at hpath (512 * l.val + i.val) _⟩
  · rw [ite_eq_right hlen] at h
    cases h

theorem ports_congr {C : Verity.Circuit} {count words : ℕ} {position : ℕ → ℕ} {tok tok' : Fin 32 → Fin C.N}
    {root root' domain domain' : Fin 512 → Fin C.N} {row row' : Fin (16 * words) → Fin C.N}
    {salt salt' : Fin (8 * Hash.Hm96.SALT_BYTES) → Fin C.N}
    {path path' : Fin (Hash.Tree.depth count) → Fin 512 → Fin C.N} {h h' : Fin (16 * words) → Fin C.N}
    {ok ok' : Fin C.N} (h1 : tok = tok') (h2 : root = root') (h3 : domain = domain') (h4 : row = row')
    (h5 : salt = salt') (h6 : path = path') (h7 : h = h') (h8 : ok = ok') :
    (ReadUnit.Ports.mk count words position tok root domain row salt path h ok : ReadUnit.Ports C) =
      ReadUnit.Ports.mk count words position tok' root' domain' row' salt' path' h' ok' := by
  subst h1 h2 h3 h4 h5 h6 h7 h8
  rfl

theorem portsAtGates_eq {C : Verity.Circuit} {inG outG : String → ℕ → Option ℕ} {ro : Roles} {out : String}
    {count words : ℕ} {q : ReadUnit.Ports C} (h : portsAtGates C inG outG ro out count words = some q)
    (at_ : ℕ → Fin C.N) (ckey croot cdomain crow csalt ch : ℕ → ℕ) (cpath : ℕ → ℕ → ℕ) (cok : ℕ)
    (hkey : ∀ b < 32, ∀ g, inG ro.key b = some g → (at_ (ckey b)).val = g)
    (hroot : ∀ b < 512, ∀ g, inG ro.root b = some g → (at_ (croot b)).val = g)
    (hdomain : ∀ b < 512, ∀ g, inG ro.domain b = some g → (at_ (cdomain b)).val = g)
    (hrow : ∀ b < 16 * words, ∀ g, inG (ro.opened.getD 0 "") b = some g → (at_ (crow b)).val = g)
    (hsalt : ∀ b < 8 * Hash.Hm96.SALT_BYTES, ∀ g, inG (ro.opened.getD 1 "") b = some g → (at_ (csalt b)).val = g)
    (hpath : ∀ l < Hash.Tree.depth count, ∀ i < 512, ∀ g,
      inG (ro.opened.getD 2 "") (512 * l + i) = some g → (at_ (cpath l i)).val = g)
    (hh : ∀ b < 16 * words, ∀ g, outG out b = some g → (at_ (ch b)).val = g)
    (hok : ∀ g, outG ro.verdict 0 = some g → (at_ cok).val = g) :
    q = { count := count, words := words, position := fun r => r, tok := fun i => at_ (ckey i),
          root := fun i => at_ (croot i), domain := fun i => at_ (cdomain i), row := fun i => at_ (crow i),
          salt := fun i => at_ (csalt i), path := fun l i => at_ (cpath l i), h := fun i => at_ (ch i),
          ok := at_ cok } := by
  unfold portsAtGates at h
  by_cases hlen : ro.opened.length = 3
  · rw [ite_eq_left hlen] at h
    obtain ⟨tok, htok, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨root, hroot', h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨domain, hdomain', h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨rowG, hrow', h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨saltG, hsalt', h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨pathG, hpath', h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨hG, hh', h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨okG, hok', h⟩ := Option.map_eq_some_iff.1 h
    subst h
    exact ports_congr (gatesOf_eq htok at_ ckey hkey) (gatesOf_eq hroot' at_ croot hroot)
      (gatesOf_eq hdomain' at_ cdomain hdomain) (gatesOf_eq hrow' at_ crow hrow)
      (gatesOf_eq hsalt' at_ csalt hsalt)
      (funext fun l => funext fun i => Fin.ext (hpath l.val l.isLt i.val i.isLt _ (gatesOf_some hpath'
        ⟨512 * l.val + i.val, by have := l.isLt; have := i.isLt; omega⟩)).symm)
      (gatesOf_eq hh' at_ ch hh) (Fin.ext (hok _ (gatesOf_some hok' ⟨0, Nat.one_pos⟩)).symm)
  · rw [ite_eq_right hlen] at h
    cases h

theorem findIdx?_fst : ∀ (ws : List (String × ℕ)) (i : ℕ) (f : String), (ws.map (·.1))[i]? = some f →
    (ws.map (·.1)).Nodup → ws.findIdx? (·.1 == f) = some i
  | [], _, _, h, _ => by cases h
  | x :: ws, 0, f, h, _ => by
    have hx : x.1 = f := Option.some.inj h
    rw [List.findIdx?_cons, ite_eq_left (by rw [hx]; exact beq_self_eq_true f)]
  | x :: ws, i + 1, f, h, hnd => by
    have hnd' : x.1 ∉ ws.map (·.1) ∧ (ws.map (·.1)).Nodup := List.nodup_cons.1 hnd
    have hf : (ws.map (·.1))[i]? = some f := h
    have hne : ¬ (x.1 == f) = true := fun he =>
      hnd'.1 (eq_of_beq he ▸ List.mem_of_getElem? hf)
    rw [List.findIdx?_cons, ite_eq_right hne, findIdx?_fst ws i f hf hnd'.2]
    rfl

theorem fieldBit_at {c : Flock.Circuit} {ws : List (String × ℕ)} {f : String} {i β : ℕ}
    (hname : (ws.map (·.1))[i]? = some f) (hnd : (ws.map (·.1)).Nodup) (hupv : 0 < c.upv)
    (hβ : β < 16 * (ws.getD i default).2) :
    Spot.Bits.fieldBit c ws f β = some (0, ((ws.take i).map (·.2)).sum + β / 16, β % 16) := by
  have hw : β / 16 < (ws.getD i default).2 := by omega
  have hle : (ws.getD i default).2 ≤ c.upv * (ws.getD i default).2 := Nat.le_mul_of_pos_left _ hupv
  unfold Spot.Bits.fieldBit
  rw [findIdx?_fst ws i f hname hnd]
  show (if ((ws.getD i default).2 == 0 || decide (c.upv * (ws.getD i default).2 ≤ β / 16)) = true then none
    else some (β / 16 / (ws.getD i default).2, ((ws.take i).map (·.2)).sum + β / 16 % (ws.getD i default).2,
      β % 16)) = _
  have hc : ¬ ((ws.getD i default).2 == 0 || decide (c.upv * (ws.getD i default).2 ≤ β / 16)) = true := by
    simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq]
    omega
  rw [ite_eq_right hc, Nat.div_eq_of_lt hw, Nat.mod_eq_of_lt hw]

theorem rowGate_copyGate {V H : ℕ} {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) (hcheck : Flock.ReadNet.check V H tu.t = true)
    (hB : FlockSoundness.Built (Flock.HmOut.unrolled ct).unit) (n : ℕ) (u : Fin n)
    (hj : 0 < (Flock.HmOut.unrolled ct).upv) {d : Defn} {ws : List (String × ℕ)} (hws : paramWords d = ws) {i : ℕ}
    {port : String} {W P : ℕ} (hname : (ws.map (·.1))[i]? = some port) (hnd : (ws.map (·.1)).Nodup)
    (hW : (ws.getD i default).2 = W) (hP : ((ws.take i).map (·.2)).sum = P)
    (hfit : P + W ≤ 162 + H + 32 * Hash.Tree.depth V) {b col g : ℕ} (hb : b < 16 * W) (hcol : col = 16 * P + b)
    (h : rowGate (Flock.HmOut.unrolled ct) d n u.val port b = some g) :
    (Flock.Bridge.copyGate (.tmpl ct) hB n u hj (ReadUnit.colAt (FlockSoundness.Rows.ofNet hB) col)).val = g := by
  subst hws
  have hf : Spot.Bits.fieldBit (Flock.HmOut.unrolled ct) (paramWords d) port b = some (0, P + b / 16, b % 16) := by
    rw [fieldBit_at hname hnd hj (by rw [hW]; exact hb), hP]
  unfold rowGate at h
  rw [hf] at h
  have hc : col = 16 * (P + b / 16) + b % 16 := by omega
  exact (congrArg (fun x => (Flock.Bridge.copyGate (.tmpl ct) hB n u hj
      (ReadUnit.colAt (FlockSoundness.Rows.ofNet hB) x)).val) hc).trans
    (ReadUnit.copyGate_colAt_inGate ht hsz hcheck hB n u hj (p := P + b / 16) (t := b % 16) (by omega)
      (Nat.mod_lt _ (by decide)) h)

theorem outRowGate_copyGate (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) (u : Fin n)
    (hj : 0 < D.unroll.upv) {d : Defn} {ws : List (String × ℕ)} (hws : outWords d = ws) {i : ℕ} {out : String}
    {W P : ℕ} (hname : (ws.map (·.1))[i]? = some out) (hnd : (ws.map (·.1)).Nodup)
    (hW : (ws.getD i default).2 = W) (hP : ((ws.take i).map (·.2)).sum = P)
    {b g : ℕ} (hb : b < 16 * W) (h : outRowGate D.unroll d n u.val out b = some g) :
    (Flock.Bridge.copyGate D hU n u hj (ReadUnit.colAt (FlockSoundness.Rows.ofNet hU)
      (FlockSoundness.Discharge.Layout.outCol D.unroll (P + b / 16) (b % 16)))).val = g := by
  subst hws
  have hf : Spot.Bits.fieldBit D.unroll (outWords d) out b = some (0, P + b / 16, b % 16) := by
    rw [fieldBit_at hname hnd hj (by rw [hW]; exact hb), hP]
  unfold outRowGate at h
  rw [hf] at h
  exact ReadUnit.copyGate_colAt_outGate D hU n u hj h

theorem readRows_spec {d : Defn} {ro : Roles} {out : String} {count words : ℕ}
    (h : readRows d ro out count words = true) :
    paramWords d = [(ro.key, 2), (ro.root, 32), (ro.domain, 32), (ro.opened.getD 0 "", words),
        (ro.opened.getD 1 "", Hash.Hm96.SALT_BYTES / 2), (ro.opened.getD 2 "", 32 * Hash.Tree.depth count)] ∧
      outWords d = [(out, words), (ro.verdict, 1)] ∧
      [ro.key, ro.root, ro.domain, ro.opened.getD 0 "", ro.opened.getD 1 "", ro.opened.getD 2 ""].Nodup ∧
      [out, ro.verdict].Nodup := by
  unfold readRows at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩

theorem parseTmpl_of_parseStmt {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {ct : Flock.Circuit} (h : Flock.Desc.parseStmt tags bytes tables = .ok (.tmpl ct)) :
    Flock.HmOut.parseTmpl tags bytes tables = .ok ct := by
  unfold Flock.Desc.parseStmt at h
  split at h
  · split at h
    · next c hc => cases h; exact hc
    · cases h
  · unfold Flock.Desc.parse at h
    split at h <;> cases h

theorem portsOf_portsAt {V H : ℕ} {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {ct : Flock.Circuit} {tu : Flock.TmplUnit} (hparse : Flock.HmOut.parseTmpl tags bytes tables = .ok ct)
    (ht : ct.tmpl = some tu) (hcheck : Flock.ReadNet.check V H tu.t = true) (ho : ct.outNet = ct.unitNet)
    (hB : FlockSoundness.Built (Flock.HmOut.unrolled ct).unit) (n : ℕ) (u : Fin n) {d : Defn} {ro : Roles}
    {out : String} {q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n)}
    (h : portsOf ct hB n d u.val ro out V H = some q) :
    ∃ hj : 0 < (Flock.HmOut.unrolled ct).upv, q = ReadUnit.portsAt (.tmpl ct) hB n u hj V H := by
  have hsz : ct.unitNet < ct.nets.size := by
    obtain ⟨_, _, P, -, rfl, -, -⟩ := FlockSoundness.TmplValue.parseTmpl_facts hparse
    exact P.unitNet
  have hj : 0 < (Flock.HmOut.unrolled ct).upv := (ReadUnit.Gates.unrolled_leaves_of_parse hparse ho).1
  unfold portsOf at h
  obtain ⟨hrows, h⟩ := Option.ite_none_right_eq_some.1 h
  obtain ⟨hp, hw, hnp, hno⟩ := readRows_spec hrows
  have hS : Hash.Hm96.SALT_BYTES = 192 := rfl
  have key := fun (i : ℕ) (port : String) (W P : ℕ) =>
    @rowGate_copyGate V H ct tu ht hsz hcheck hB n u hj d _ hp i port W P
  have outk := fun (i : ℕ) (o : String) (W P : ℕ) =>
    @outRowGate_copyGate (.tmpl ct) hB n u hj d _ hw i o W P
  exact ⟨hj, portsAtGates_eq h
    (fun x => Flock.Bridge.copyGate (.tmpl ct) hB n u hj (ReadUnit.colAt (FlockSoundness.Rows.ofNet hB) x))
    (fun b => b) (fun b => Flock.ReadNet.ROOT_AT + b) (fun b => Flock.ReadNet.DOMAIN_AT + b)
    (fun b => Flock.ReadNet.ROW_AT + b) (fun b => Flock.ReadNet.saltAt H + b)
    (fun b => FlockSoundness.Discharge.Layout.outCol (Flock.HmOut.unrolled ct) (b / 16) (b % 16))
    (fun l i => Flock.ReadNet.pathAt H + 512 * l + i)
    (FlockSoundness.Discharge.Layout.outCol (Flock.HmOut.unrolled ct) H 0)
    (fun b hb g hg => key 0 ro.key 2 0 rfl hnp rfl rfl (by omega) (by omega) (by omega) hg)
    (fun b hb g hg => key 1 ro.root 32 2 rfl hnp rfl rfl (by omega) (by omega)
      (by unfold Flock.ReadNet.ROOT_AT; omega) hg)
    (fun b hb g hg => key 2 ro.domain 32 34 rfl hnp rfl rfl (by omega) (by omega)
      (by unfold Flock.ReadNet.DOMAIN_AT; omega) hg)
    (fun b hb g hg => key 3 (ro.opened.getD 0 "") H 66 rfl hnp rfl rfl (by omega) hb
      (by unfold Flock.ReadNet.ROW_AT; omega) hg)
    (fun b hb g hg => key 4 (ro.opened.getD 1 "") (Hash.Hm96.SALT_BYTES / 2) (66 + H) rfl hnp rfl
      (show 2 + (32 + (32 + (H + 0))) = 66 + H by omega) (by omega) (by omega)
      (by unfold Flock.ReadNet.saltAt; omega) hg)
    (fun l hl i hi g hg => key 5 (ro.opened.getD 2 "") (32 * Hash.Tree.depth V) (162 + H) rfl hnp rfl
      (show 2 + (32 + (32 + (H + (Hash.Hm96.SALT_BYTES / 2 + 0)))) = 162 + H by omega) (by omega) (by omega)
      (by unfold Flock.ReadNet.pathAt; omega) hg)
    (fun b hb g hg => (congrArg (fun x => (Flock.Bridge.copyGate (.tmpl ct) hB n u hj
        (ReadUnit.colAt (FlockSoundness.Rows.ofNet hB)
          (FlockSoundness.Discharge.Layout.outCol (Flock.HmOut.unrolled ct) x (b % 16)))).val)
        (Nat.zero_add (b / 16)).symm).trans (outk 0 out H 0 rfl hno rfl rfl hb hg))
    (fun g hg => outk 1 ro.verdict 1 H rfl hno rfl rfl (show 0 < 16 * 1 by omega) hg)⟩

theorem pinnedReadBinds : PinnedReadBinds := by
  intro F c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n consts u q Yc table h
  unfold PinnedRead PinnedReadAt at h
  unfold ReturnsRow
  obtain ⟨-, -, -, -, -, -, hstmt, ht, hcheck, ho, hlaid, hports, hone, hcorrect, hconsts, hok, htable⟩ := h
  have hparse := parseTmpl_of_parseStmt hstmt
  have hu : u.val = r.unit := by
    unfold laidOutAs at hlaid
    simp only [Bool.and_eq_true, beq_iff_eq] at hlaid
    exact hlaid.2
  rw [← hu] at hports
  obtain ⟨hj, hq⟩ := portsOf_portsAt hparse ht hcheck ho hB n u hports
  have hgates : ReadUnit.GatesRefine Verity.sha512 Verity.sha512 Flock.Hm96.Default512.key.data.toList consts
      (Flock.Bridge.partsOf (.tmpl ct) hB n) u q := by
    rw [hq]
    exact ReadUnit.gatesRefine_check hparse ht hcheck ho hB n consts u hone hj
  have hrows : ReadUnit.RegisteredRows consts (Flock.Bridge.partsOf (.tmpl ct) hB n) u q (q.rootOf Yc)
      (q.domainOf Yc) Yc :=
    ⟨rfl, rfl, hok, hconsts⟩
  have hb := ResourceAccounting.SecurityProofs.readUnitBinds Verity.sha512 Verity.sha512
    Flock.Hm96.Default512.key.data.toList consts (Flock.Bridge.partsOf (.tmpl ct) hB n) u q _ _ Yc hgates hcorrect
    hrows
  obtain ⟨hh, -⟩ := hgates.computes Yc hcorrect hconsts
  by_cases hlt : q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) < q.count
  · rcases hb (table _) (htable _ hlt) with h | h | h
    · left
      rw [h]
      unfold Hash.EmbeddingRead.h
      exact (ite_eq_left hlt).symm
    · exact .inr (.inl h)
    · exact .inr (.inr h)
  · left
    rw [hh]
    unfold Hash.EmbeddingRead.h
    exact (ite_eq_right hlt).trans (ite_eq_right hlt).symm

end InferenceOnlyCheck.Properties.Record.SecurityProofs
