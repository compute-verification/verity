import RemoteAudit

/-!
# The remote bandwidth audit's verifier: what its Lean verdicts mean (`RemoteAudit`)

The properties of `RemoteAudit` (`verity/core/protocols/remote_bandwidth/lean/`, the verifier's definition, which
`audit.py` is held to by the difftest `remote-audit-difftest`), stated against its functions as they run, for every input of their types. They
are the code's half of the bridge from the verifier to the stacked audit's game (`Rate.DeployedSession`, and the
composed statement `Accounting.Guarantees.DeclaredCircuitRequirement` that reads it): each says what an acceptance
establishes about the bytes the verifier holds. They take no premise. What the code cannot see (the coins' uniformity,
the receipts' and the store's meaning as the game's oracle and words, the window's wall time) is not here: it is a
premise or a proofs unit of the bridge, named in each docstring.

The memory line of record (`RemoteAudit.Line.line`, its W from `RemoteAudit.Pooled`) is one statement over what this
verifier accepts: `Properties.RemoteAudit.Line.LineHeld`, the bits its credited passes read
(`Properties/RemoteAudit/Line.lean`, proved in `Proofs/Accounting/RemoteHeld/Line.lean`, where `heldAt_of_read` derives
memory held from it under the device model), with no code-to-model premise (the trace of record is over the
registration's blocks by definition, `Properties/RemoteAudit/Line/Premises.lean`, and the verifier refuses a record not
of the registration given, `LineRegistered`) and the W's cover of the session's pooled bound in `Properties/RemoteAudit/Pooled.lean`.

The window (`verifyWindow`):

* `AcceptChecksOpenings`: an accepted window drew its receipts from its openings coin, has one opening a draw, and
  each drawn opening passes `checkOpening` under the receipter of its pass's seed;
* `AcceptOpensDraws`: the same, with `checkOpening`'s five checks spelled out (`OpeningPasses`): the opening is the
  drawn receipt's, its leaf is under its pass's root in the domain of the pass's seed and plan, its block decodes to a
  payload under the payload tree or its append's tree and coin, and the receipt recomputed from the block is the leaf's.
* `ChunkedIsWhole`: the verifier's process, which checks a window's openings in chunks on threads of their own
  (`checkWindowChunked`), gives `checkWindow`'s verdict on the whole window, the one `Session.run` sets.

The draw (`draws`):

* `DrawIsSubset`: the window's draws are the sampler's `min(s, U)`-subset of `[0, U)` (`Randomness.subset` on the key
  derived from the openings coin, the registration and the window), each value `u` mapped to the receipt `(i, p)` whose
  index among the window's `U = Σ L_i` receipts in pass order is `u` (`receiptIndex`). So the draws are distinct
  receipts of the window, as many as the sampler's values, and the map is a bijection: what `Key.subset`'s law gives
  the values (a uniform `k`-subset over independent uniform streams: `KeyedDraw.subsetOn_escape_le`, a proofs unit,
  under a uniform live coin, a premise) it gives the receipts.

A pass's plan (`Plan.make?`, `Plan.changed`):

* `PlanReadsDistinct`, `ChangedReadsDistinct`: a plan the code accepts (whole, as runs, or as a change of the previous
  pass's) reads `L` distinct blocks of the store at its positions `0 … L − 1`.

The session (`Session.run` from `State.init`, every event's result `.ok`; the refusals and rejections included):

* `AcceptedMinPasses`: an accepted session has its `W` windows, each with at least `min_passes` passes;
* `PassesStepped`: each pass recorded has one plan, one step and one chain head, and the steps strictly increase in
  arrival order (window, then pass) below `2^64`;
* `PassesReadDistinct`: each pass recorded reads distinct blocks of the store as it stands;
* `ReceiptsBelowLimit`: each window recorded has fewer than `2^64` receipts, its passes' lengths summed;
* `AppendsInOrder`: the appends are the store's next indices in order, append `k` answered with append coin `k`;
* `PassesChained`: the session's head is the chain over its appends, and each pass's head the chain over a prefix of
  them, past which its plan reads nothing, so its seed binds every append its plan reads;
* `ReleasedStayOut`: a block released by an accepted release is read by no pass received after it;
* `LifetimeCapped`: each append's blocks are read within `lifetime` consecutive windows.

When the session reads a window's openings coin, and what its draw fixes (the code's half of the opening draw's
`Refused` and `Live`), is `Properties.RemoteAudit.Coins`, and at one openings coin over `g` windows
(`groups_per_coin`, `Shared.run g`) `Properties.RemoteAudit.Shared`. That the verifier's process, which takes the events
one request at a time on the coins the challenger sent and settles each window's check later, accepts what
`Shared.run g` (`Session.run` at g = 1) accepts on the events its transcript records, at two windows in flight, is
`Properties.RemoteAudit.Live` (`LiveIsRun`).
-/

open RemoteAudit

namespace Properties.RemoteAudit

/-! ## The window -/

/-- The receipter `checkWindow` checks pass `i`'s openings with: the registration's receipt kind under pass `i`'s
seed. -/
def receipterOf (reg : Registration) (w : WindowRecord) (i : Nat) : Receipter :=
  receipter reg.receipt (w.seedOf i) reg.rows

/-- An accepted window drew `drawn` from its openings coin over its passes' plans, has exactly one opening a draw, and
each drawn receipt's opening passes `checkOpening` under the receipter of its pass's seed. -/
def AcceptChecksOpenings : Prop :=
  ∀ (reg : Registration) (appends : Array Append) (w : WindowRecord) (coin : ByteArray) (samples : Nat)
    (openings : Array Opening),
    verifyWindow reg appends w coin samples openings = some .accept →
      ∃ drawn, draws reg.digest w.index coin samples (w.passes.map (·.plan.length)) = some drawn ∧
        openings.size = drawn.size ∧
        ∀ k (hk : k < drawn.size),
          checkOpening reg reg.digest appends w (receipterOf reg w drawn[k].1) drawn[k].1 drawn[k].2 openings[k]! = none

/-- The verifier's process (`Verifier.spawnCheck`) checks a window's openings in chunks of `size` draws, each on a
thread of its own (`checkWindowChunked`): its verdict is `checkWindow`'s on the whole window, which `Session.verify`
sets in `Session.run`, for every chunk size and every input, a rejection naming the same failure and draw. -/
def ChunkedIsWhole : Prop :=
  ∀ (size : Nat) (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord)
    (drawn : Array (Nat × Nat)) (openings : Array Opening),
    checkWindowChunked size reg digest appends w drawn openings = checkWindow reg digest appends w drawn openings

/-- Draw `(i, p)`'s opening `o` passes checks 1–5 (`checkOpening`), with `j` the block at position `p` of pass `i`'s
plan:

1. it is the drawn receipt's (`o.pass = i`, `o.position = p`); its block is 4,096 bytes, it carries no tile, its salts
   are 192 bytes, its paths' siblings 64 bytes, and it has a payload digest exactly when `j` is an appended block;
2. its leaf holds a receipt `y` at `p` (the leaf's size is its positions'), and the leaf, under its salt and path, is
   leaf `p / 16` of pass `i`'s receipt tree under the pass's root, in the domain of the pass's seed and plan;
3. and 4. its block decodes to a payload: a registered block (`j < N`) under the registration's key, the payload's leaf
   under its salt and path at `j` under the payload root; an appended block under the bound key of its digest and its
   append's coin, the payload's leaf at `j` under its append's root (the append holding `j`, `appendOf`);
5. the receipt recomputed from `j` and the block, under the pass's receipter, is `y`. -/
def OpeningPasses (reg : Registration) (appends : Array Append) (w : WindowRecord) (i p : Nat) (o : Opening) :
    Prop :=
  let pr := w.passes[i]!
  let j := pr.plan.block p
  o.pass = i ∧ o.position = p ∧ o.block.size = Codec.BLOCK_BYTES ∧ o.tile.size = 0 ∧
    o.leafSalt.size = SALT_BYTES ∧ o.payloadSalt.size = SALT_BYTES ∧
    (∀ x ∈ o.receiptPath, x.size = DIGEST_BYTES) ∧ (∀ x ∈ o.payloadPath, x.size = DIGEST_BYTES) ∧
    o.digest.size = (if j < reg.blocks then 0 else DIGEST_BYTES) ∧
    ∃ y, leafReceipt o.leaf p pr.plan.length = some y ∧
      verifyLeaf (w.seedOf i) pr.plan pr.root p o.leaf o.leafSalt o.receiptPath = true ∧
      (∃ payload,
        (j < reg.blocks ∧ Codec.decode o.block j reg.key none = .ok (some payload) ∧
          FrameV3.verifyOpening reg.payloadDomain reg.payloadRoot j (hidden payload o.payloadSalt) SCHEMA
            o.payloadPath = true) ∨
        (reg.blocks ≤ j ∧ ∃ a, appendOf appends j = some a ∧
          Codec.decode o.block j reg.key (some (o.digest, some a.coin)) = .ok (some payload) ∧
          FrameV3.verifyOpening (appendDomain reg.digest a.first a.count) a.root j (hidden payload o.payloadSalt)
            SCHEMA o.payloadPath = true)) ∧
      (receipterOf reg w i).receipt j o.block = y

/-- An accepted window drew `drawn` from its openings coin, has exactly one opening a draw, and each drawn receipt's
opening passes checks 1–5 (`OpeningPasses`). -/
def AcceptOpensDraws : Prop :=
  ∀ (reg : Registration) (appends : Array Append) (w : WindowRecord) (coin : ByteArray) (samples : Nat)
    (openings : Array Opening),
    verifyWindow reg appends w coin samples openings = some .accept →
      ∃ drawn, draws reg.digest w.index coin samples (w.passes.map (·.plan.length)) = some drawn ∧
        openings.size = drawn.size ∧
        ∀ k (hk : k < drawn.size), OpeningPasses reg appends w drawn[k].1 drawn[k].2 openings[k]!

/-! ## The draw -/

/-- Receipt `(i, p)`'s index among a window's receipts in pass order: the positions of the passes before pass `i`,
then `p`. -/
def receiptIndex (lengths : Array Nat) (i p : Nat) : Nat := (lengths.toList.take i).sum + p

/-- The key a window's openings are drawn on: `derive(c, OPENINGS_DOMAIN, {registration, window: t})`. -/
def openingsKey (registration : ByteArray) (t : Nat) (coin : ByteArray) : ByteArray :=
  FlockSoundness.Randomness.derive coin OPENINGS_DOMAIN [("registration", .bytes registration), ("window", .int t)]

/-- The draws of window `t` (passes of `lengths` positions, `U` receipts in all) are `Key.subset`'s
`min(s, U)`-subset of `[0, U)` on the openings key, its stream read to some length `L`, each value `u` mapped to the
receipt `(i, p)` with `receiptIndex = u`: so they are `min(s, U)` distinct receipts of the window (`i` a pass,
`p < L_i`), one for each of the sampler's values. -/
def DrawIsSubset : Prop :=
  ∀ (registration : ByteArray) (t : Nat) (coin : ByteArray) (samples : Nat) (lengths : Array Nat)
    (drawn : Array (Nat × Nat)),
    draws registration t coin samples lengths = some drawn →
      ∃ L us, FlockSoundness.Randomness.subset (openingsKey registration t coin) (min samples lengths.toList.sum)
          lengths.toList.sum [.str "openings"] L = some us ∧
        drawn.size = us.size ∧ us.size = min samples lengths.toList.sum ∧
        (∀ k (hk : k < drawn.size), drawn[k].1 < lengths.size ∧ drawn[k].2 < lengths[drawn[k].1]! ∧
          receiptIndex lengths drawn[k].1 drawn[k].2 = us[k]!) ∧
        (∀ k₁ k₂ (h₁ : k₁ < drawn.size) (h₂ : k₂ < drawn.size), drawn[k₁] = drawn[k₂] → k₁ = k₂)

/-! ## A pass's plan -/

/-- A plan reads `L` distinct blocks of a store of `blocks` blocks: position `p < L` reads block `pl.block p`, below
`blocks`, and no two positions read one block. -/
def ReadsDistinct (pl : Plan) (blocks : Nat) : Prop :=
  (∀ p, p < pl.length → pl.block p < blocks) ∧
    ∀ p q, p < pl.length → q < pl.length → pl.block p = pl.block q → p = q

/-- A plan given as runs that the code accepts reads distinct blocks of the store. -/
def PlanReadsDistinct : Prop :=
  ∀ (runs : Array (Int × Int)) (blocks : Nat) (pl : Plan), Plan.make? runs blocks = some pl → ReadsDistinct pl blocks

/-- A plan given as a change of the previous pass's that the code accepts reads distinct blocks of the store. -/
def ChangedReadsDistinct : Prop :=
  ∀ (pl : Plan) (added removed : Array (Int × Int)) (blocks : Nat) (pl' : Plan),
    pl.changed added removed blocks = some pl' → ReadsDistinct pl' blocks

/-! ## The session -/

/-- The chain over the appends `as` in order, `Session.append`'s `head`: `appendHead` folded from `none`. -/
def chainOf (as : Array Append) : Option ByteArray :=
  as.foldl (fun h a => some (appendHead h a.first a.count a.root)) none

/-- An accepted session has every window, each with at least `min_passes` passes (`close` refuses fewer with `few`). -/
def AcceptedMinPasses : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
    s.outcome = some "accepted" →
      s.windows.size = params.windows ∧ ∀ t (ht : t < s.windows.size), params.minPasses ≤ s.windows[t].roots.size

/-- Every pass the verifier records is bound to one plan, one step and one append chain head, and the steps, read in
the order the passes arrive (window, then pass), strictly increase and stay below `2^64` (`receive`'s `step`). -/
def PassesStepped : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
      (∀ t (ht : t < s.windows.size), s.windows[t].plans.size = s.windows[t].roots.size ∧
        s.windows[t].steps.size = s.windows[t].roots.size ∧ s.windows[t].heads.size = s.windows[t].roots.size ∧
        ∀ i (hi : i < s.windows[t].steps.size), s.windows[t].steps[i] < Session.STEP_LIMIT) ∧
      ∀ t₁ t₂ (h₁ : t₁ < s.windows.size) (h₂ : t₂ < s.windows.size) i₁ i₂ (hi₁ : i₁ < s.windows[t₁].steps.size)
        (hi₂ : i₂ < s.windows[t₂].steps.size), (t₁ < t₂ ∨ (t₁ = t₂ ∧ i₁ < i₂)) →
          s.windows[t₁].steps[i₁] < s.windows[t₂].steps[i₂]

/-- Every pass the verifier records reads distinct blocks of the store as it stands (registered and appended). -/
def PassesReadDistinct : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
      ∀ t (ht : t < s.windows.size) i (hi : i < s.windows[t].plans.size),
        ReadsDistinct s.windows[t].plans[i] s.extent

/-- Every window the verifier records has fewer than `2^64` receipts, its passes' lengths summed (`receive` rejects with
`plan` a pass that would take them to `RECEIPT_LIMIT`). -/
def ReceiptsBelowLimit : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
      ∀ t (ht : t < s.windows.size), s.windows[t].receipts < Session.RECEIPT_LIMIT

/-- The appends are the store's next indices in order (append `k` starts where the one before it ends, the first at
`N`, and the store ends at the last), and append `k` answered with the `k`-th append coin, each drawn once. -/
def AppendsInOrder : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
      s.extent = reg.blocks + (s.appends.toList.map (·.count)).sum ∧
      (∀ k (hk : k < s.appends.size), s.appends[k].first = reg.blocks + ((s.appends.toList.take k).map (·.count)).sum) ∧
      s.appends.size ≤ appendCoins.size ∧ ∀ k (hk : k < s.appends.size), s.appends[k].coin = appendCoins[k]!

/-- The session's head is the chain over its appends, and every pass is bound to the chain over the appends received
before it, a prefix of them, and reads only registered blocks and theirs: so its seed,
`seed(challenge, i, chained(previous root, head))`, binds every append its plan reads (no block appended after a
pass's seed is in its plan). -/
def PassesChained : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
      s.head = chainOf s.appends ∧
      ∀ t (ht : t < s.windows.size) i (hi : i < s.windows[t].heads.size),
        ∃ m, m ≤ s.appends.size ∧ s.windows[t].heads[i] = chainOf (s.appends.extract 0 m) ∧
          ∀ p, p < s.windows[t].plans[i]!.length →
            s.windows[t].plans[i]!.block p < reg.blocks + ((s.appends.toList.take m).map (·.count)).sum

/-- A block released by an accepted release is read by no pass received after it: a run to `s₁`, a release of `runs`
(naming any `withdrawn` among them) answered `.ok`, and a run on to `s`; every pass `s` records past `s₁`'s (in a later
window, or later in its window) reads no block of `runs` (`receive` refuses a plan holding a released block with
`plan`). -/
def ReleasedStayOut : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events₁ events₂ : Array Session.Event) (runs withdrawn : Array (Int × Int)) (rs₁ rs : Array Session.Result)
    (s₁ s₂ s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events₁ = .ok (rs₁, s₁) →
    Session.step s₁ (.release runs withdrawn) = .ok (.ok, s₂) →
    Session.run s₂ events₂ = .ok (rs, s) →
      ∀ r ∈ runs, ∀ t (ht : t < s.windows.size) i (hi : i < s.windows[t].plans.size),
        (s₁.windows.size ≤ t ∨ s₁.windows[t]!.plans.size ≤ i) →
        ∀ p, p < s.windows[t].plans[i].length →
          ¬ (r.1 ≤ (s.windows[t].plans[i].block p : Int) ∧ (s.windows[t].plans[i].block p : Int) < r.1 + r.2)

/-- Each append's blocks are read within `lifetime` consecutive windows: for append `k` there is a window `j` (the one
it joined, being filled at the first pass received after it) such that every pass reading one of its blocks is in a
window `j ≤ t < j + L` (`receive` refuses with `lifetime` a plan holding a block of an append that joined a window `L`
or more before). The registered blocks, below `N`, carry no such bound. -/
def LifetimeCapped : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
      ∀ k (hk : k < s.appends.size), ∃ j, ∀ t (ht : t < s.windows.size) i (hi : i < s.windows[t].plans.size) p,
        p < s.windows[t].plans[i].length →
        s.appends[k].first ≤ s.windows[t].plans[i].block p →
        s.windows[t].plans[i].block p < s.appends[k].first + s.appends[k].count →
          j ≤ t ∧ t < j + params.lifetime

end Properties.RemoteAudit
