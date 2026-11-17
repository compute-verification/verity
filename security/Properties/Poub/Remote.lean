module

public import Poub
public import Poub.Session

/-! The properties of PoUB's Lean verifier (`Poub`): what its acceptance of a window, a sample, a pass's plan and a
session establishes about the bytes it holds. -/

@[expose] public section

open Poub

namespace Properties.Poub

/-- The receipter `checkWindow` checks pass `i`'s openings with: the registration's receipt kind under pass `i`'s
seed. -/
def receipterOf (reg : Registration) (w : WindowRecord) (i : Nat) : Receipter :=
  receipter reg.receipt (w.seedOf i) reg.rows

/-- An accepted window drew `sampled` from its openings coin over its passes' plans, has exactly one opening a sample, and
each sampled receipt's opening passes `checkOpening` under the receipter of its pass's seed. -/
def AcceptChecksOpenings : Prop :=
  ∀ (reg : Registration) (appends : Array Append) (w : WindowRecord) (coin : ByteArray) (samples : Nat)
    (openings : Array Opening),
    verifyWindow reg appends w coin samples openings = some .accept →
      ∃ sampled, receiptSample reg.digest w.index coin samples (w.passes.map (·.plan.length)) = some sampled ∧
        openings.size = sampled.size ∧
        ∀ k (hk : k < sampled.size),
          checkOpening reg reg.digest appends w (receipterOf reg w sampled[k].1) sampled[k].1 sampled[k].2 openings[k]! = none

/-- The verifier's process (`Verifier.spawnCheck`) checks a window's openings in chunks of `size` samples, each on a
thread of its own (`checkWindowChunked`): its verdict is `checkWindow`'s on the whole window, which `Session.verify`
sets in `Session.run`, for every chunk size and every input, a rejection naming the same failure and sample. -/
def ChunkedIsWhole : Prop :=
  ∀ (size : Nat) (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord)
    (sampled : Array (Nat × Nat)) (openings : Array Opening),
    checkWindowChunked size reg digest appends w sampled openings = checkWindow reg digest appends w sampled openings

/-- Sample `(i, p)`'s opening `o` passes checks 1–5 (`checkOpening`), with `j` the block at position `p` of pass `i`'s
plan:

1. it is the sampled receipt's (`o.pass = i`, `o.position = p`); its block is 4,096 bytes, it carries no tile, its salts
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
          reg.payloadDomain.verifyOpening reg.payloadRoot reg.payloadDomain.count j (hidden payload o.payloadSalt)
            SCHEMA o.payloadPath = true) ∨
        (reg.blocks ≤ j ∧ ∃ a, appendOf appends j = some a ∧
          Codec.decode o.block j reg.key (some (o.digest, some a.coin)) = .ok (some payload) ∧
          (appendDomain reg.digest a.first a.count).verifyOpening a.root (appendDomain reg.digest a.first a.count).count
            j (hidden payload o.payloadSalt) SCHEMA o.payloadPath = true)) ∧
      (receipterOf reg w i).receipt j o.block = y

/-- An accepted window drew `sampled` from its openings coin, has exactly one opening a sample, and each sampled receipt's
opening passes checks 1–5 (`OpeningPasses`). -/
def AcceptOpensSamples : Prop :=
  ∀ (reg : Registration) (appends : Array Append) (w : WindowRecord) (coin : ByteArray) (samples : Nat)
    (openings : Array Opening),
    verifyWindow reg appends w coin samples openings = some .accept →
      ∃ sampled, receiptSample reg.digest w.index coin samples (w.passes.map (·.plan.length)) = some sampled ∧
        openings.size = sampled.size ∧
        ∀ k (hk : k < sampled.size), OpeningPasses reg appends w sampled[k].1 sampled[k].2 openings[k]!

/-- Receipt `(i, p)`'s index among a window's receipts in pass order: the positions of the passes before pass `i`,
then `p`. -/
def receiptIndex (lengths : Array Nat) (i p : Nat) : Nat := (lengths.toList.take i).sum + p

/-- The key a window's openings are sampled on: `derive(c, OPENINGS_DOMAIN, {registration, window: t})`. -/
def openingsKey (registration : ByteArray) (t : Nat) (coin : ByteArray) : ByteArray :=
  FlockSoundness.Randomness.derive coin OPENINGS_DOMAIN [("registration", .bytes registration), ("window", .int t)]

/-- The samples of window `t` (passes of `lengths` positions, `U` receipts in all) are `Key.subset`'s
`min(s, U)`-subset of `[0, U)` on the openings key, its stream read to some length `L`, each value `u` mapped to the
receipt `(i, p)` with `receiptIndex = u`: so they are `min(s, U)` distinct receipts of the window (`i` a pass,
`p < L_i`), one for each of the sampler's values. -/
def SampleIsSubset : Prop :=
  ∀ (registration : ByteArray) (t : Nat) (coin : ByteArray) (samples : Nat) (lengths : Array Nat)
    (sampled : Array (Nat × Nat)),
    receiptSample registration t coin samples lengths = some sampled →
      ∃ L us, FlockSoundness.Randomness.subset (openingsKey registration t coin) (min samples lengths.toList.sum)
          lengths.toList.sum [.str "openings"] L = some us ∧
        sampled.size = us.size ∧ us.size = min samples lengths.toList.sum ∧
        (∀ k (hk : k < sampled.size), sampled[k].1 < lengths.size ∧ sampled[k].2 < lengths[sampled[k].1]! ∧
          receiptIndex lengths sampled[k].1 sampled[k].2 = us[k]!) ∧
        (∀ k₁ k₂ (h₁ : k₁ < sampled.size) (h₂ : k₂ < sampled.size), sampled[k₁] = sampled[k₂] → k₁ = k₂)

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
`N`, and the store ends at the last), and append `k` answered with the `k`-th append coin, each sampled once. -/
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

end Properties.Poub
