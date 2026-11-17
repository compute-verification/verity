module

public import Flock.PortRoots
public import Proofs.Flock.Bridge.TransportsRefs
public import Proofs.Flock.Claims.Bridge.Transports
public import Proofs.Flock.Soundness.Discharge.Hidden.Public
public import Proofs.Flock.Soundness.Discharge.Hidden.Spec
public import Proofs.Flock.Soundness.Discharge.Layout.Drawn
public import Proofs.Flock.Soundness.Discharge.Layout.QSort
public import Proofs.Flock.Soundness.Discharge.Layout.SharedRows

/-! A session row opens its population's published tree (`pubPort_opens`, `setupHidden_unit`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **The population unit a session's instance `n` is**: its sample's `n`-th proved unit (`Draw.UnitSample.proved`),
`n` itself without a sample. -/
def unitOf (draw : Option Lean.Json) (n : ℕ) : ℕ :=
  match draw with
  | some d =>
    match _root_.Flock.Draw.UnitSample.ofJson d with
    | .ok D => D.proved.getD n 0
    | .error _ => n
  | none => n

/-- **Unit `u`'s row of port `q`'s published tree**: its ref at a shared-row file, else `u`
(`Bridge.LeafMap.ofPublic`'s). -/
def rowAt (c : _root_.Flock.Circuit) (pop : _root_.Flock.Public) (q u : ℕ) : ℕ :=
  (_root_.Flock.Bridge.LeafMap.ofPublic c pop pop.n).ref q u

/-- **Port `q`'s published tree's leaves** over a population `pop`, as `PortRoots.pubRoot` hashes them: leaf `r` is the
frame-v3 leaf, at the port's schema and published domain, of the tree leaf of row `r`'s `b ‖ c`. -/
def pubLeaves (c : _root_.Flock.Circuit) (pop : _root_.Flock.Public) (q : ℕ) : Array ByteArray :=
  (Array.range (_root_.Flock.PortRoots.countOf pop q)).map fun r =>
    _root_.Flock.FV3.Sha512.leafAs (_root_.Flock.PortRoots.schemaOf c q) (_root_.Flock.PortRoots.idOf c pop q) r
      (_root_.Flock.Hm96.Default512.treeLeaf (_root_.Flock.PortRoots.digOf c pop q r))

/-- **Port `q`'s published tree as a registered port**: its root (`PortRoots.pubRoot`), leaf count, published domain
and schema, the fields a registered read (`Registered.opens`) reads. -/
def pubPort (c : _root_.Flock.Circuit) (pop : _root_.Flock.Public) (q : ℕ) : _root_.Flock.Registered.Port :=
  { name := "", value := "", root := _root_.Flock.PortRoots.pubRoot c pop q,
    leaves := _root_.Flock.PortRoots.countOf pop q, words := none, wordBits := 0,
    domain := _root_.Flock.PortRoots.idOf c pop q, readsRel := #[], schema := _root_.Flock.PortRoots.schemaOf c q }

/-- **The path from row `r` of port `q`'s published tree to its root**: its siblings, lowest first, 64 bytes each. -/
def pubPath (c : _root_.Flock.Circuit) (pop : _root_.Flock.Public) (q r : ℕ) : ByteArray :=
  bytes (pathMsg (FlockC3.RegLevels.sibsV (_root_.Flock.PortRoots.idOf c pop q) (pubLeaves c pop q) r))

theorem ofPublic_cnt (c : _root_.Flock.Circuit) (pop : _root_.Flock.Public) (q : ℕ) :
    (_root_.Flock.Bridge.LeafMap.ofPublic c pop pop.n).cnt q = _root_.Flock.PortRoots.countOf pop q := by
  unfold _root_.Flock.Bridge.LeafMap.ofPublic _root_.Flock.PortRoots.countOf
  cases pop.tables <;> rfl

theorem ofPublic_dig (c : _root_.Flock.Circuit) (pop : _root_.Flock.Public) (q r : ℕ) :
    (_root_.Flock.Bridge.LeafMap.ofPublic c pop pop.n).dig q r = _root_.Flock.PortRoots.digOf c pop q r := by
  unfold _root_.Flock.Bridge.LeafMap.ofPublic _root_.Flock.PortRoots.digOf
  cases pop.tables <;> rfl

theorem pubPort_opens {tags : _root_.Flock.Tags} {c : _root_.Flock.Circuit} {file : ByteArray}
    {pop : _root_.Flock.Public} (h : _root_.Flock.HmOut.loadPublic tags c file = .ok pop) {u q : ℕ} (hu : u < pop.n)
    (hq : q < c.ports.size) :
    _root_.Flock.Registered.opens (pubPort c pop q) (pop.digest u q c.ports.size) (rowAt c pop q u)
      (pubPath c pop q (rowAt c pop q u)) = true := by
  have hlt : rowAt c pop q u < _root_.Flock.PortRoots.countOf pop q := by
    rw [← ofPublic_cnt c pop q]
    exact _root_.Flock.Bridge.ref_lt_cnt h hu hq
  have hdig : _root_.Flock.PortRoots.digOf c pop q (rowAt c pop q u) = pop.digest u q c.ports.size := by
    rw [← ofPublic_dig c pop q]
    exact _root_.Flock.Bridge.LeafMap.dig_ref c pop pop.n q u
  have hsz : (pubLeaves c pop q).size = _root_.Flock.PortRoots.countOf pop q := by simp [pubLeaves]
  unfold _root_.Flock.Registered.opens
  rw [← hdig]
  apply SamplingDeploy.opensLeaf_tree (pubPort c pop q) (pubLeaves c pop q) hsz
  · intro x hx
    simp only [pubLeaves, Array.mem_map] at hx
    obtain ⟨_, -, rfl⟩ := hx
    exact _root_.Flock.Sha512.hash_size _
  · rfl
  · exact hlt
  · rw [getElem!_pos _ _ (by rw [hsz]; exact hlt)]
    simp [pubLeaves, pubPort]

theorem setupHidden_unit {tags : _root_.Flock.Tags} {cf pf : ByteArray} {coins : _root_.Flock.Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray}
    {st : _root_.Flock.Stmt}
    (h : _root_.Flock.Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st)
    {pop : _root_.Flock.Public} (hpop : _root_.Flock.HmOut.loadPublic tags st.c pf = .ok pop) {n q : ℕ}
    (hn : n < st.pub.n) (hq : q < st.c.ports.size) :
    unitOf draw n < pop.n ∧ st.pub.digest n q st.c.ports.size = pop.digest (unitOf draw n) q st.c.ports.size := by
  obtain ⟨pop', hpop', hnone, hsome⟩ := (FlockSoundness.Discharge.Hidden.setupHidden_spec' h).pub
  have e : pop' = pop := Except.ok.inj (hpop'.symm.trans hpop)
  rw [e] at hnone hsome
  have hload := FlockSoundness.Discharge.Hidden.loadPublic_nil hpop
  cases draw with
  | none =>
    have hp : st.pub = pop := hnone rfl
    simp only [unitOf]
    rw [hp] at hn ⊢
    exact ⟨hn, rfl⟩
  | some d =>
    have hdrawn : _root_.Flock.HmRow.drawn { st.c with outPorts := #[] } pop d = .ok st.pub := hsome d rfl
    obtain ⟨D, hD, hpn, hnD, -⟩ := FlockSoundness.Discharge.Layout.sampled_spec hdrawn
    have hlt := FlockSoundness.Discharge.Layout.proved_lt_population hD
    have hr : n < D.proved.size := hnD ▸ hn
    have hu : unitOf (some d) n = D.proved[n] := by
      simp only [unitOf, hD, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hr, Option.getD_some]
    rw [hu]
    exact ⟨hpn ▸ hlt _ (Array.getElem_mem hr),
      FlockSoundness.Discharge.Hidden.sampled_digestW hload hdrawn hD hlt hr hq⟩

end SpotFlock
