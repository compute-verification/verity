import Proofs.Flock.Claims.Recursion.ResLowerDeployed
import Proofs.Flock.Claims.Recursion.FinalPiece

/-!
# A residual copy's row ports read its committed rows

At a staged algebra part, element `o` of row port `i` among the rows the copy of instance `n` reads
(`rowsRead ns nv (copyOf … (n / g) (n % g) 0)`, element `(ns.take i).sum + o`) is element `o` of the bytes of that port's
row at instance `n` (`rowHidden`'s forms, read off the witness), which is the row the part commits at that port
(`held_rowSlot`).

* `eraseDups_nodup`, `nodup_of_innerUnit`: `checkRepS`'s distinct names (`eraseDups` keeps the length) make V\*'s row
  port names a `Nodup` list.
* `port_at`: row port `i` of V\*'s part is the circuit's port `i`: its name finds it, its bits are its words', and the
  words before it are the row ports'.
* `rowCol_start`, `rowPortCol_eq`: the verifier's row column of port `i` is `16` times the row ports' words before it.
* `copy_rowElems`: the copy's row port elements are the committed row's, and row port `i` is named for the verifier's
  port `P[i]` (`checkPart_ok`).
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB
open FlockSoundness.Discharge.Hm (bitL)

/-- `eraseDups` keeps a list's length only where the list has no duplicates. -/
theorem eraseDups_nodup {α : Type} [BEq α] [LawfulBEq α] : ∀ (n : ℕ) (l : List α), l.length ≤ n →
    l.eraseDups.length ≤ l.length ∧ (l.eraseDups.length = l.length → l.Nodup)
  | _, [], _ => by simp
  | 0, _ :: _, h => by simp at h
  | n + 1, a :: as, h => by
    have hf : (as.filter fun b => !b == a).length ≤ as.length := List.length_filter_le ..
    obtain ⟨ih1, ih2⟩ := eraseDups_nodup n (as.filter fun b => !b == a) (by simp at h; omega)
    rw [List.eraseDups_cons, List.length_cons, List.length_cons]
    refine ⟨by omega, fun he => ?_⟩
    have hfl : (as.filter fun b => !b == a).length = as.length := by omega
    have hall := List.length_filter_eq_length_iff.1 hfl
    have hfe : (as.filter fun b => !b == a) = as := List.filter_eq_self.2 hall
    rw [hfe] at ih2 he
    refine List.nodup_cons.2 ⟨fun hm => ?_, ih2 (by omega)⟩
    have := hall a hm
    simp at this

/-- V\*'s `InnerRepCheck` unit's row ports have distinct names (`checkRepS`). -/
theorem nodup_of_innerUnit {S : VStar.RepS} {nm : String} {u : VStar.UnitC} (h : VStar.innerUnit S nm = .ok u) :
    (S.ports.map (·.1)).Nodup := by
  unfold VStar.innerUnit at h
  cases hc : VStar.checkRepS S with
  | error e => simp [hc, bind, Except.bind] at h
  | ok _ =>
    unfold VStar.checkRepS at hc
    by_cases hne : (S.ports.map (·.1)).eraseDups.length = (S.ports.map (·.1)).length
    · exact (eraseDups_nodup _ _ le_rfl).2 hne
    · have h1 : ((S.ports.map (·.1)).eraseDups.length != (S.ports.map (·.1)).length ||
          (S.ports.map (·.1)).contains "v" || S.ports.any (·.2 < 1)) = true := by
        simp only [Bool.or_eq_true]
        exact Or.inl (Or.inl (bne_iff_ne.2 hne))
      simp only [h1, ↓reduceIte] at hc
      contradiction

/-- **Row port `i` is the circuit's port `i`**: at an algebra part's statement, `checkParsed`'s rows are V\*'s
(`vstarRows`), the row ports first, so the circuit's port `i` carries row port `i`'s name and words, its name finds it
(the names are distinct, `nodup_of_innerUnit`), and the words before it are the row ports'. -/
theorem port_at {j : ℕ} {nm : String} {pt : VStar.Part} {c : Circuit} {u : VStar.UnitC}
    (hu : (VStar.Stmt.inner j nm pt).unit = .ok u)
    (hr : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      RecVStar.vstarRows (VStar.Stmt.inner j nm pt).sub u)
    {i : ℕ} (hi : i < pt.S.ports.length) :
    ∃ hc : i < c.ports.size, c.ports.findIdx? (·.name == pt.S.ports[i].1) = some i ∧
      c.ports[i].bits = 16 * VStar.portWordsOf pt.S.ports[i].2 ∧
      (c.ports.extract 0 i).foldl (· + ·.words) 0 = ((pt.S.ports.take i).map fun p => VStar.portWordsOf p.2).sum := by
  have hu' : VStar.innerUnit pt.S nm = .ok u := hu
  have hnd := nodup_of_innerUnit hu'
  obtain ⟨-, -, hin, -⟩ := FlockVBridge.OfVStar.innerUnit_ok hu'
  have hlen : (pt.S.ports.map (·.1)).length = (pt.S.ports.map fun p => VStar.portWordsOf p.2).length := by simp
  have hr' : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      ((pt.S.ports.map (·.1)).zip (pt.S.ports.map fun p => VStar.portWordsOf p.2)).map
        (fun p => (p.1, p.2, 16 * p.2)) ++
      (("v", VStar.portWordsOf pt.S.gf.nv, 16 * VStar.portWordsOf pt.S.gf.nv) ::
        (["out"].zip u.outPorts).map fun p => (p.1, (p.2 + 63) / 64 * 64, 16 * ((p.2 + 63) / 64 * 64))) := by
    rw [hr]
    have e1 : (VStar.Stmt.inner j nm pt).sub.inputs = pt.S.ports.map (·.1) ++ ["v"] := rfl
    have e2 : (VStar.Stmt.inner j nm pt).sub.outputs = ["out"] := rfl
    unfold RecVStar.vstarRows
    rw [e1, e2, hin, FlockVBridge.OfVStar.repPorts, List.zip_append hlen, List.map_append, List.append_assoc]
    rfl
  have hA : ((pt.S.ports.map (·.1)).zip (pt.S.ports.map fun p => VStar.portWordsOf p.2)).map
      (fun p => (p.1, p.2, 16 * p.2)) =
      pt.S.ports.map fun p => (p.1, VStar.portWordsOf p.2, 16 * VStar.portWordsOf p.2) := by
    rw [List.zip_map', List.map_map]
    rfl
  rw [hA] at hr'
  have hget : ∀ t (ht : t < pt.S.ports.length), c.ports.toList[t]?.map (fun p => (p.name, p.words, p.bits)) =
      some (pt.S.ports[t].1, VStar.portWordsOf pt.S.ports[t].2, 16 * VStar.portWordsOf pt.S.ports[t].2) := by
    intro t ht
    rw [← List.getElem?_map, hr', List.getElem?_append_left (by simpa using ht), List.getElem?_map,
      List.getElem?_eq_getElem ht]
    rfl
  have hcs : ∀ t, t < pt.S.ports.length → t < c.ports.size := fun t ht => by
    have h := hget t ht
    by_contra hge
    rw [List.getElem?_eq_none (by simpa using hge)] at h
    simp at h
  have hfield : ∀ t (ht : t < pt.S.ports.length), (c.ports[t]'(hcs t ht)).name = pt.S.ports[t].1 ∧
      (c.ports[t]'(hcs t ht)).words = VStar.portWordsOf pt.S.ports[t].2 ∧
      (c.ports[t]'(hcs t ht)).bits = 16 * VStar.portWordsOf pt.S.ports[t].2 := by
    intro t ht
    have h := hget t ht
    rw [List.getElem?_eq_getElem (by simpa using hcs t ht), Option.map_some, Option.some.injEq,
      Array.getElem_toList] at h
    simp only [Prod.mk.injEq] at h
    exact h
  refine ⟨hcs i hi, ?_, (hfield i hi).2.2, ?_⟩
  · rw [Array.findIdx?_eq_some_iff_getElem]
    refine ⟨hcs i hi, by simp [(hfield i hi).1], fun t hti => ?_⟩
    have ht := Nat.lt_trans hti hi
    simp only [(hfield t ht).1, beq_iff_eq]
    intro he
    have hne : (pt.S.ports.map (·.1))[t]'(by simpa using ht) = (pt.S.ports.map (·.1))[i]'(by simpa using hi) := by
      simpa using he
    have := (hnd.getElem_inj).1 hne
    omega
  · have he : (c.ports.extract 0 i).foldl (· + ·.words) 0 = ((c.ports.toList.take i).map (·.words)).sum :=
      extract_words (·.words) c.ports i
    rw [he]
    congr 1
    apply List.ext_getElem
    · simp only [List.length_map, List.length_take, Array.length_toList]
      have := hcs i hi
      omega
    · intro t h1 h2
      have ht : t < pt.S.ports.length := by simp at h2; omega
      simp only [List.getElem_map, List.getElem_take, Array.getElem_toList]
      exact (hfield t ht).2.1

/-- The row column where row port `i` starts, from column `k`, where it has an element: `16` bits a word of the row
ports before it. -/
theorem rowCol_start : ∀ (ns : List ℕ) (k i : ℕ) (hi : i < ns.length), 0 < ns[i] →
    FlockVBridge.OfVStar.rowCol (ns.map ("", ·)) k (128 * (ns.take i).sum) =
      k + 16 * ((ns.take i).map VStar.portWordsOf).sum
  | [], _, _, hi, _ => absurd hi (Nat.not_lt_zero _)
  | n :: ns, k, 0, _, hn => by
    have hn' : 0 < n := hn
    show FlockVBridge.OfVStar.rowCol (("", n) :: ns.map ("", ·)) k (128 * 0) = k + 16 * 0
    simp only [FlockVBridge.OfVStar.rowCol]
    try dsimp only
    split_ifs with h
    · omega
    · exact absurd (show 128 * 0 < 128 * n by omega) h
  | n :: ns, k, i + 1, hi, hn => by
    have hi' : i < ns.length := by simp at hi; omega
    have ih := rowCol_start ns (k + 16 * VStar.portWordsOf n) i hi' hn
    show FlockVBridge.OfVStar.rowCol (("", n) :: ns.map ("", ·)) k (128 * (n + (ns.take i).sum)) =
      k + 16 * (VStar.portWordsOf n + ((ns.take i).map VStar.portWordsOf).sum)
    simp only [FlockVBridge.OfVStar.rowCol]
    try dsimp only
    split_ifs with h
    · exact absurd h (by omega)
    · rw [show 128 * (n + (ns.take i).sum) - 128 * n = 128 * (ns.take i).sum by omega, ih]
      omega

theorem rowPortCol_eq {ns : List ℕ} {i : ℕ} (hi : i < ns.length) (hn : 0 < ns[i]) :
    rowPortCol ns i = 16 * ((ns.take i).map VStar.portWordsOf).sum := by
  unfold rowPortCol
  rw [rowCol_start ns 0 i hi hn, Nat.zero_add]

/-- **A copy's row port elements are its committed row's**: at a staged algebra part whose unit the verifier accepted
as part `jj` of its shape (`(S, ns, nv)`), element `o` of row port `i` among the rows the copy of instance `n` reads is
element `o` of the bytes of port `i`'s row at instance `n` (`rowHidden`'s forms) where the port isn't named `out`. The
port is the circuit's port `i`, which its name finds (`port_at`), its row's columns are the copy's from the port's first word (`rowHidden_eq`,
the unit's first input group at column 0, `inCol_unitLaid`), and the verifier's row column of port `i` is that word
(`rowPortCol_eq`). -/
theorem copy_rowElems {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm' : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit}
    (hpc : r.w.parts[s]?.bind partCircuit = some c) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (ht : 0 < p.sts.size)
    {m k cl jj : ℕ} {sh : Flock.Shape} {holo : Bool} {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hres : RecResiduals.params c.unit.name = some (m, k, cl, jj)) (hshp : RecResiduals.shape? m k cl = some sh)
    (hholo : RecResiduals.holoOf c = .ok holo) (hpart : RecResiduals.part { sh with holo } jj = .ok (S, ns, nv))
    {i o : ℕ} (hi : i < ns.length) (ho : o < ns[i]) (n : ℕ) :
    ∃ hi' : i < pt.S.ports.length, c.ports.findIdx? (·.name == pt.S.ports[i].1) = some i ∧
      (∀ P, RecVStar.partPorts { sh with holo } jj = .ok P →
      ∃ hP : i < P.size, pt.S.ports[i].1 = (RecVStar.portNames { sh with holo })[P[i]]!) ∧
      (pt.S.ports[i].1 ≠ "out" →
      _root_.Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p 0 ht w (n / c.g) (n % c.g) 0) ((ns.take i).sum + o) =
        elemsOf (readForms w (((rowHidden c pt.S.ports[i].1 n).getD []).map HmNets.Lin.var)) ns[i] o) := by
  obtain ⟨p', r', sv, hx', ht', hpi, hr', ⟨ss', hss', hsv'⟩, h0, htab⟩ := stagedTables hS
  rw [hx] at hx'
  cases hx'
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hS' := hS
  obtain ⟨-, i0, spec, sts, -, hsu, -⟩ := hS'
  have hp : p = { i0.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  have hsz : p.sts.size = 1 := by rw [hp]; exact (oneFull_staged hS hsu).1
  have hc0 := partCircuit_staged hS hx ht' hpi hr hpc h0
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hx ht' hpi h0
  obtain ⟨-, hpk, hpa⟩ := htab 0 h0
  obtain ⟨u, hu, hupv, -, hl, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok hpa
  subst hc0
  obtain ⟨S', ns', nv', hpart', hn⟩ := FlockVBridge.OfVStar.check_part_at P.recResiduals hres hshp hholo
  rw [hpart] at hpart'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hpart'
  obtain ⟨rfl, rfl, rfl⟩ := hpart'
  obtain ⟨hns, -, hnames⟩ := FlockSoundness.Discharge.Recursion.checkPart_ok hpk rfl hres hshp hholo hpart
  have hi' : i < pt.S.ports.length := by
    have := congrArg List.length hns
    simp only [List.length_map] at this
    omega
  obtain ⟨hcs, hfind, hbits, hwords⟩ := port_at hu hrows hi'
  refine ⟨hi', hfind, fun P hP => ?_, fun hnout => ?_⟩
  · have heq := hnames P hP
    have hlen := congrArg List.length heq
    simp only [List.length_map, Array.length_toList] at hlen
    refine ⟨by omega, ?_⟩
    have h2 := List.getElem_of_eq heq (by simpa using hi')
    simp only [List.getElem_map, Array.getElem_toList] at h2
    exact h2
  have hnsi : ns[i] = pt.S.ports[i].2 := by subst hns; simp
  have hpos : 0 < ns[i] := by omega
  have hout : pt.S.ports[i].1 ∉ (VStar.Stmt.inner j nm' pt).sub.outputs := by
    show pt.S.ports[i].1 ∉ ["out"]
    simpa using hnout
  have hN := in_block hrows hout hfind
  rw [getElem!_pos p.sts[0].c.ports i hcs] at hN
  have hrh := rowHidden_eq n hl hfind hcs hN
  have hcol := inCol_unitLaid hn
  have hpw := FlockVBridge.OfVStar.le_portWords ns[i]
  rw [hnsi] at hpw
  have hcp := leaves_le_constPos P hl (N := u.inWords) (by omega)
  rw [hcol, Nat.zero_mul, Nat.zero_add] at hcp hrh
  have hrpc : rowPortCol ns i = 16 * ((pt.S.ports.take i).map fun p => VStar.portWordsOf p.2).sum := by
    rw [rowPortCol_eq hi hpos, ← hns, ← List.map_take, List.map_map]
    rfl
  rw [hrh, Option.getD_some, rowsRead_port hi ho]
  unfold _root_.Flock.Readable.readElems elemsOf
  rw [ite_eq_left ho, ite_eq_left ho]
  refine FlockVBridge.OfVStar.ofBits_congr fun t htt => ?_
  show Flock.Readable.copyOf p 0 ht w (n / p.sts[0].c.g) (n % p.sts[0].c.g) 0 (rowPortCol ns i + (128 * o + t)) = _
  rw [hnsi] at ho
  rw [bitL_readForms (by simp only [List.length_map, List.length_range]; rw [hbits]; omega)]
  simp only [List.getElem_map, List.getElem_range, formBit_var]
  rw [hwords] at hN
  rw [hrpc, Claims.Recursion.VStar.copyOf_eq (by omega), hwords]
  simp [Claims.Recursion.VStar.copyIdx, hsz, hupv]

end RecDeploy.ResLower
