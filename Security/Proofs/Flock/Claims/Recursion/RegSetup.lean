import Proofs.Flock.Claims.Recursion.PartAcceptsSetup
import Proofs.Flock.Claims.Recursion.DeployInputs
import Proofs.Flock.Verifier.Registered

/-!
# The registered reads' evaluations in the V\* setup's (`RegSetup`)

The compiled verifier opens each registered port's table rows itself (`Flock.Registered.checkPort`, which
`checkRegistered` runs on every statement of the recursive setup, `RecIns.setup`): each row's `b ‖ c` is hashed into its
tree leaf, framed at its position, and climbed to the registered root (`Flock.Registered.opens`). Those are SHA-512
evaluations outside the oracle, and `DecompBV`'s escape (`setupCalls`) searches them once the brought setup lists them.

* `opensInputs reg bc pos path`: what `opens` hashes, the tree leaf's input, the frame-v3 leaf's and each climb node's
  (`climbInputs`, `leafInputs`). `opens_coll`: two different commit strings opened at one position under one root give
  two different inputs among them with one SHA-512 digest (`opens_binding`, keeping where the pair sits).
* `regOpenings xO`: each opening the recursive setup at the V\* statement `xO` makes of a row an instance reads,
  `(reg, b ‖ c, position, path)` of the table row each statement's instance's ref names in each of the verifier's
  registered ports, as `checkPort` reads them (`portOpenings`). Rows no instance references are left out: no conclusion
  reads them, and their number is the prover's to stage. `regIns xO`: their inputs, as messages.
* `RegSetup b`: the brings' setup lists `regIns` at every V\* statement; `withReg b` is `b` with them appended
  (`regSetup_withReg`), and the records `recBound` compiles list them (`regListed_recBound`).
* `bc_eq_of_regOpenings`: at a record listing them, two listed openings at one position under one root (`SameRoot`),
  both of which open, have one commit string, or the recorded calls and the setup's evaluations collide: so a lowering
  under `JointV`'s no-collision conjunct reads two parts' commit strings as one.
-/

namespace Flock.Registered

/-- What `climbL` hashes climbing from `cur` at index `idx` of level `d` through `sibs`: each node's frame, lowest
first. -/
def climbInputs (dom : ByteArray) : ByteArray → Nat → Nat → List ByteArray → List ByteArray
  | _, _, _, [] => []
  | cur, idx, d, sib :: rest =>
    (if idx % 2 = 0 then nodeIn dom d (idx / 2) cur sib else nodeIn dom d (idx / 2) sib cur) ::
      climbInputs dom
        (Sha512.hash (if idx % 2 = 0 then nodeIn dom d (idx / 2) cur sib else nodeIn dom d (idx / 2) sib cur))
        (idx / 2) (d + 1) rest

/-- What `opensLeaf reg payload pos path` hashes: the frame-v3 leaf's frame, then the climb's. -/
def leafInputs (reg : Port) (payload : ByteArray) (pos : Nat) (path : ByteArray) : List ByteArray :=
  leafInAs reg.schema reg.domain pos payload ::
    climbInputs reg.domain (Sha512.hash (leafInAs reg.schema reg.domain pos payload)) pos 0
      (sibs path (log2ceil reg.leaves))

/-- What `opens reg bc pos path` hashes: the tree leaf's input `leafPrefix ‖ b ‖ c`, then `leafInputs`. -/
def opensInputs (reg : Port) (bc : ByteArray) (pos : Nat) (path : ByteArray) : List ByteArray :=
  (Hm96.Default512.leafPrefix ++ bc) :: leafInputs reg (Hm96.Default512.treeLeaf bc) pos path

/-- **Two climbs to one root from different digests collide among their inputs** (`climbPair_collides`, keeping
where the pair sits). -/
theorem climb_coll (dom : ByteArray) :
    ∀ (s1 s2 : List ByteArray) (a b : ByteArray) (idx d : Nat), s1.length = s2.length →
      (∀ x ∈ s1, x.size = 64) → (∀ x ∈ s2, x.size = 64) → a.size = 64 → b.size = 64 → a ≠ b →
      climbL dom a idx d s1 = climbL dom b idx d s2 →
      ∃ u ∈ climbInputs dom a idx d s1, ∃ v ∈ climbInputs dom b idx d s2, u ≠ v ∧ Sha512.hash u = Sha512.hash v := by
  intro s1
  induction s1 with
  | nil =>
    intro s2 a b idx d hl _ _ _ _ hab h
    cases s2 with
    | nil => exact absurd h hab
    | cons _ _ => simp at hl
  | cons s rest ih =>
    intro s2 a b idx d hl hw1 hw2 ha hb hab h
    cases s2 with
    | nil => simp at hl
    | cons t rest2 =>
      have hl' : rest.length = rest2.length := by simpa using hl
      have hsz := hw1 s (by simp)
      have htz := hw2 t (by simp)
      have hr1 : ∀ x ∈ rest, x.size = 64 := fun x hx => hw1 x (by simp [hx])
      have hr2 : ∀ x ∈ rest2, x.size = 64 := fun x hx => hw2 x (by simp [hx])
      simp only [climbL, node_eq] at h
      by_cases hp : idx % 2 = 0
      · simp only [hp, ↓reduceIte] at h
        simp only [climbInputs, hp, ↓reduceIte]
        by_cases he : Sha512.hash (nodeIn dom d (idx / 2) a s) = Sha512.hash (nodeIn dom d (idx / 2) b t)
        · exact ⟨_, List.mem_cons.2 (Or.inl rfl), _, List.mem_cons.2 (Or.inl rfl),
            fun hh => hab (nodeIn_inj (by rw [ha, hb]) (by rw [hsz, htz]) hh).1, he⟩
        · obtain ⟨u, hu, v, hv, huv⟩ :=
            ih rest2 _ _ (idx / 2) (d + 1) hl' hr1 hr2 (Sha512.hash_size _) (Sha512.hash_size _) he h
          exact ⟨u, List.mem_cons.2 (Or.inr hu), v, List.mem_cons.2 (Or.inr hv), huv⟩
      · simp only [hp, ↓reduceIte] at h
        simp only [climbInputs, hp, ↓reduceIte]
        by_cases he : Sha512.hash (nodeIn dom d (idx / 2) s a) = Sha512.hash (nodeIn dom d (idx / 2) t b)
        · exact ⟨_, List.mem_cons.2 (Or.inl rfl), _, List.mem_cons.2 (Or.inl rfl),
            fun hh => hab (nodeIn_inj (by rw [hsz, htz]) (by rw [ha, hb]) hh).2, he⟩
        · obtain ⟨u, hu, v, hv, huv⟩ :=
            ih rest2 _ _ (idx / 2) (d + 1) hl' hr1 hr2 (Sha512.hash_size _) (Sha512.hash_size _) he h
          exact ⟨u, List.mem_cons.2 (Or.inr hu), v, List.mem_cons.2 (Or.inr hv), huv⟩

/-- **Two accepted openings at one position of different leaves collide among their inputs** (`opensLeaf_binding`,
keeping where the pair sits). -/
theorem opensLeaf_coll (reg : Port) (pos : Nat) (x path y path' : ByteArray) (hx : x.size = 64) (hy : y.size = 64)
    (hne : x ≠ y) (h1 : opensLeaf reg x pos path = true) (h2 : opensLeaf reg y pos path' = true) :
    ∃ u ∈ leafInputs reg x pos path, ∃ v ∈ leafInputs reg y pos path', u ≠ v ∧ Sha512.hash u = Sha512.hash v := by
  simp only [opensLeaf, Bool.and_eq_true, decide_eq_true_eq] at h1 h2
  have heq : climbL reg.domain (leafAs reg.schema reg.domain pos x) pos 0 (sibs path (log2ceil reg.leaves)) =
      climbL reg.domain (leafAs reg.schema reg.domain pos y) pos 0 (sibs path' (log2ceil reg.leaves)) :=
    h1.2.trans h2.2.symm
  rw [leafAs_eq, leafAs_eq] at heq
  by_cases hl : Sha512.hash (leafInAs reg.schema reg.domain pos x) = Sha512.hash (leafInAs reg.schema reg.domain pos y)
  · exact ⟨_, List.mem_cons.2 (Or.inl rfl), _, List.mem_cons.2 (Or.inl rfl),
      fun hh => hne (leafInAs_inj (by rw [hx, hy]) hh), hl⟩
  · obtain ⟨u, hu, v, hv, huv⟩ := climb_coll reg.domain _ _ _ _ pos 0 (by rw [sibs_length, sibs_length])
      (sibs_size h1.1.2) (sibs_size h2.1.2) (Sha512.hash_size _) (Sha512.hash_size _) hl heq
    exact ⟨u, List.mem_cons.2 (Or.inr hu), v, List.mem_cons.2 (Or.inr hv), huv⟩

/-- **Two different commit strings opened at one position under one root collide among their inputs**
(`opens_binding`, keeping where the pair sits). -/
theorem opens_coll (reg : Port) (pos : Nat) (bc path bc' path' : ByteArray) (hne : bc ≠ bc')
    (h1 : opens reg bc pos path = true) (h2 : opens reg bc' pos path' = true) :
    ∃ u ∈ opensInputs reg bc pos path, ∃ v ∈ opensInputs reg bc' pos path', u ≠ v ∧
      Sha512.hash u = Sha512.hash v := by
  by_cases hl : Hm96.Default512.treeLeaf bc = Hm96.Default512.treeLeaf bc'
  · exact ⟨_, List.mem_cons.2 (Or.inl rfl), _, List.mem_cons.2 (Or.inl rfl),
      fun e => hne ((ByteArray.append_right_inj _).mp e), hl⟩
  · obtain ⟨u, hu, v, hv, huv⟩ :=
      opensLeaf_coll reg pos _ path _ path' (Sha512.hash_size _) (Sha512.hash_size _) hl h1 h2
    exact ⟨u, List.mem_cons.2 (Or.inr hu), v, List.mem_cons.2 (Or.inr hv), huv⟩

/-- Two registered ports with one tree: what `opens` reads of a port (its leaves, domain, schema and root) is one. -/
def SameRoot (reg reg' : Port) : Prop :=
  reg'.leaves = reg.leaves ∧ reg'.domain = reg.domain ∧ reg'.schema = reg.schema ∧ reg'.root = reg.root

theorem opens_same {reg reg' : Port} (h : SameRoot reg reg') (bc : ByteArray) (pos : Nat) (path : ByteArray) :
    opens reg' bc pos path = opens reg bc pos path := by
  cases reg
  cases reg'
  obtain ⟨hl, hd, hs, hr⟩ := h
  dsimp only at hl hd hs hr
  subst hl hd hs hr
  rfl

theorem opensInputs_same {reg reg' : Port} (h : SameRoot reg reg') (bc : ByteArray) (pos : Nat) (path : ByteArray) :
    opensInputs reg' bc pos path = opensInputs reg bc pos path := by
  obtain ⟨hl, hd, hs, -⟩ := h
  simp only [opensInputs, leafInputs, hl, hd, hs]

end Flock.Registered

namespace RecDeploy

/-- **`checkPort`'s openings of `reg` at the statement `st` that its instances read**: for each of the statement's `n`
instances, the row of the port's table its ref names, that row's `b ‖ c`, the position the header's entry gives it and
its path, as `checkPort` reads them (`badInstance`, `badRow`). A row no instance references is not listed: `checkPort`
opens it too, but nothing concluded from an accepted statement reads it (`checkPort_at`, `Registered.Checked` and
`bc_eq_of_regOpenings` are about the rows instances read), so its climb needs no escape. So a port lists at most `n`
openings, which the setup's `m ≤ 35` refusal caps, and never the table's row count, which the prover stages. -/
def portOpenings (st : _root_.Flock.Stmt) (reg : _root_.Flock.Registered.Port) : List (ByteArray × ℕ × ByteArray) :=
  match st.pub.tables, (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == reg.name),
      _root_.Flock.Registered.entry st.pub reg with
  | some t, some p, .ok (positions, paths) =>
    (List.range st.pub.n).map fun i =>
      (_root_.Flock.Registered.tableRow st.pub t p (_root_.Flock.Registered.ref t st.c.ports.size p i),
        positions.getD (_root_.Flock.Registered.ref t st.c.ports.size p i) 0,
        paths.getD (_root_.Flock.Registered.ref t st.c.ports.size p i) ByteArray.empty)
  | _, _, _ => []

/-- **The openings the recursive setup makes at the V\* statement `xO`**: `portOpenings` of each of the verifier's
registered ports at each of the setup's statements, with its port. -/
def regOpenings (xO : Claims.Msg) : List (_root_.Flock.Registered.Port × ByteArray × ℕ × ByteArray) :=
  match setupO xO with
  | some (s, _, sts) =>
    match s.base.registered with
    | some own => sts.toList.flatMap fun st => own.toList.flatMap fun reg => (portOpenings st reg).map fun o => (reg, o)
    | none => []
  | none => []

/-- **The SHA-512 inputs of the recursive setup's registered reads at `xO`**: each opening's (`opensInputs`). -/
def regIns (xO : Claims.Msg) : List Claims.Msg :=
  (regOpenings xO).flatMap fun o => (_root_.Flock.Registered.opensInputs o.1 o.2.1 o.2.2.1 o.2.2.2).map Claims.toMsg

/-- **The brings' setup lists every registered read's evaluations** at every V\* statement. -/
def RegSetup (b : Brings) : Prop := ∀ xO, ∀ u ∈ regIns xO, u ∈ b.setup xO

/-- A record whose setup lists every registered read's evaluations at every V\* statement. -/
def RegListed {Coin : Type} (D : Claims.Recursion.Deploy Coin) : Prop := ∀ xO, ∀ u ∈ regIns xO, u ∈ D.setup xO

/-- `b` with the registered reads' evaluations appended to its setup. -/
def withReg (b : Brings) : Brings := { b with setup := fun xO => b.setup xO ++ regIns xO }

theorem regSetup_withReg (b : Brings) : RegSetup (withReg b) := fun _ _ hu => List.mem_append_right _ hu

theorem regSetup_with_tr {b : Brings} (h : RegSetup b) (t : Claims.Msg → List (Claims.Msg × Claims.Out) → ByteArray) :
    RegSetup { b with tr := t } := h

/-- **The compiled record lists them where the brings do**: `recBound`'s setup is the brings'. -/
theorem regListed_recBound (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) {b : Brings}
    (h : RegSetup b) (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier FlockDeploy.Coin) :
    RegListed (recBound rel b L IV) := h

theorem mem_portOpenings {st : _root_.Flock.Stmt} {reg : _root_.Flock.Registered.Port} {t : _root_.Flock.Tables}
    {p : ℕ} {positions : Array ℕ} {paths : Array ByteArray} (ht : st.pub.tables = some t)
    (hp : (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == reg.name) = some p)
    (he : _root_.Flock.Registered.entry st.pub reg = .ok (positions, paths)) {i : ℕ} (hi : i < st.pub.n) :
    (_root_.Flock.Registered.tableRow st.pub t p (_root_.Flock.Registered.ref t st.c.ports.size p i),
      positions.getD (_root_.Flock.Registered.ref t st.c.ports.size p i) 0,
      paths.getD (_root_.Flock.Registered.ref t st.c.ports.size p i) ByteArray.empty) ∈ portOpenings st reg := by
  simp only [portOpenings, ht, hp, he, List.mem_map, List.mem_range]
  exact ⟨i, hi, rfl⟩

theorem mem_regOpenings {xO : Claims.Msg} {s : RecIns} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    {own : Array _root_.Flock.Registered.Port} {st : _root_.Flock.Stmt} {reg : _root_.Flock.Registered.Port}
    {o : ByteArray × ℕ × ByteArray} (h : setupO xO = some (s, spec, sts)) (hr : s.base.registered = some own)
    (hst : st ∈ sts.toList) (hreg : reg ∈ own.toList) (ho : o ∈ portOpenings st reg) : (reg, o) ∈ regOpenings xO := by
  simp only [regOpenings, h, hr, List.mem_flatMap, List.mem_map]
  exact ⟨st, hst, reg, hreg, o, ho, rfl⟩

theorem toMsg_mem_regIns {xO : Claims.Msg} {reg : _root_.Flock.Registered.Port} {bc path : ByteArray} {pos : ℕ}
    (ho : (reg, bc, pos, path) ∈ regOpenings xO) {u : ByteArray}
    (hu : u ∈ _root_.Flock.Registered.opensInputs reg bc pos path) : Claims.toMsg u ∈ regIns xO := by
  unfold regIns
  exact List.mem_flatMap.2 ⟨_, ho, List.mem_map.2 ⟨u, hu, rfl⟩⟩

/-- A listed evaluation at a V\* statement of `xs` is among the setup's evaluations at `xs`. -/
theorem mem_setupCalls_of_listed {Coin : Type} {H : Claims.Msg → Claims.Msg} {D : Claims.Recursion.Deploy Coin}
    {xs : List Claims.Msg} {xO u : Claims.Msg} (hx : xO ∈ xs) (hu : u ∈ D.setup xO) :
    (u, H u) ∈ Claims.Recursion.setupCalls H D xs := by
  unfold Claims.Recursion.setupCalls
  exact List.mem_flatMap.2 ⟨xO, hx, List.mem_map.2 ⟨u, hu, rfl⟩⟩

/-- **A registered root binds the commit strings the setup opens, up to a collision the escape sees.** At a record
whose setup lists the registered reads' evaluations (`RegListed`), two openings the setup makes at V\* statements of
`xs`, at one position under one tree (`SameRoot`), both of which open, have one commit string, wherever the true calls
`l` and the setup's evaluations at `xs` have no collision (`JointV`'s conjunct, at `l := L.allCalls x ws`). -/
theorem bc_eq_of_regOpenings {Coin : Type} {D : Claims.Recursion.Deploy Coin} (hD : RegListed D)
    {l : List (Claims.Msg × Claims.Msg)} (hl : ∀ e ∈ l, FlockDeploy.sha512 e.1 = e.2) {xs : List Claims.Msg}
    (hc : (Claims.findColl (l ++ Claims.Recursion.setupCalls FlockDeploy.sha512 D xs)).isSome = false)
    {xO xO' : Claims.Msg} (hx : xO ∈ xs) (hx' : xO' ∈ xs) {reg reg' : _root_.Flock.Registered.Port}
    {bc bc' path path' : ByteArray} {pos : ℕ} (ho : (reg, bc, pos, path) ∈ regOpenings xO)
    (ho' : (reg', bc', pos, path') ∈ regOpenings xO') (hR : _root_.Flock.Registered.SameRoot reg reg')
    (h1 : _root_.Flock.Registered.opens reg bc pos path = true)
    (h2 : _root_.Flock.Registered.opens reg' bc' pos path' = true) : bc = bc' := by
  by_contra hne
  have h2' : _root_.Flock.Registered.opens reg bc' pos path' = true :=
    (_root_.Flock.Registered.opens_same hR bc' pos path').symm.trans h2
  obtain ⟨u, hu, v, hv, huv, he⟩ := _root_.Flock.Registered.opens_coll reg pos bc path bc' path' hne h1 h2'
  rw [← _root_.Flock.Registered.opensInputs_same hR] at hv
  have htrue : ∀ e ∈ l ++ Claims.Recursion.setupCalls FlockDeploy.sha512 D xs, FlockDeploy.sha512 e.1 = e.2 :=
    fun e he => (List.mem_append.1 he).elim (hl e) (Claims.Recursion.setupCalls_true _ D xs e)
  have hmem : ∀ {y : Claims.Msg} {w : ByteArray}, y ∈ xs → Claims.toMsg w ∈ D.setup y →
      Claims.toMsg w ∈ (l ++ Claims.Recursion.setupCalls FlockDeploy.sha512 D xs).map Prod.fst :=
    fun hy hw => List.mem_map.2 ⟨_, List.mem_append_right _ (mem_setupCalls_of_listed hy hw), rfl⟩
  have hcol : Claims.Collision FlockDeploy.sha512 (Claims.toMsg u) (Claims.toMsg v) :=
    ⟨fun e => huv (by rw [← Claims.ofMsg_toMsg u, ← Claims.ofMsg_toMsg v, e]), by
      simp only [FlockDeploy.sha512, Claims.ofMsg_toMsg, he]⟩
  have hfind := Claims.findColl_isSome_mem FlockDeploy.sha512 _ htrue (Claims.toMsg u) (Claims.toMsg v)
    (hmem hx (hD xO _ (toMsg_mem_regIns ho hu))) (hmem hx' (hD xO' _ (toMsg_mem_regIns ho' hv))) hcol
  exact Bool.false_ne_true (hc.symm.trans hfind)

end RecDeploy
