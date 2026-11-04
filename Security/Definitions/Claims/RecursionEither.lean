import Definitions.Claims.Recursion

/-!
# Claims: the recursion's zero knowledge if the firewall or the prover is honest (a draft, before it freezes)

The recursion's reference program (`Claims.Recursion`), which holds the witness and makes the release, split between
two parties, beside the verifier:

* **The prover** holds the values and proves with Π_in, C-Flock in its masked interaction (`--zk`, the masked record
  `Flock.ZkDeploy`), on masks, pads and salts of its own. Its first message is its registration, the registered roots
  of Π_in's statement. It hears only the verifier, and sends only through a data diode to the firewall.
* **The firewall** is the only exit. It draws the coin tree's nonce and sends `Hello`, with a commit string of its own
  to each of the prover's registered roots `r_P`, whose leaf is the registration leaf `ϱ = Leaf(r_P; s)`. It commits
  each of the prover's later messages under fresh salts of its own, as many commit strings a round as the schedule
  fixes. A message that misses its deadline is a stand-in, which the model reads as a message of the prover's. It runs
  `V_in` on Π_in's transcript and checks every V* statement before the first Π_out session. Then it runs the Π_out
  sessions in `--zk`, or sends the refusal from one fixed position on. It sends nothing to the prover.
* **The verifier** sends each live coin to the prover and the firewall, after the firewall's commit strings of its
  round, and runs `V_out`, the compiled verifier, on the firewall's messages.

Two claims, one for each honest party:

1. **`FirewallZK`: an honest firewall, any prover.**
   - The prover is any program: its next message is any function of the release's history so far, the firewall's
     messages and the verifier's answers, which a colluding verifier may relay to it. It never reads the firewall's
     tape, and its SHA-512 calls aren't the release's (`msgsAt`).
   - At every statement of the covered forms, against every malicious verifier of `V_out`'s rounds, every event of the
     transcript is at most twice as likely at `SimSplit`, which commits to the zero digest everywhere the firewall
     commits and reads neither the witness nor the prover.
   - The error is the Π_out sessions' `--zk` simulation errors and `2δ₁` per commit string in each of the release's
     two forms (`εzk`), plus `γzk` times the probability that `Fork` finds a collision among the firewall's SHA-512
     calls in two real runs.
   - The factor `2` is the release's two forms, the Π_out sessions or the refusal: a release leaks at most one bit.
2. **`ProverZK`: an honest prover, any firewall.** The firewall and the verifier together (`Firewall.against`) are one
   malicious verifier of the prover's Π_in session. Their joint view, the release's transcript, is within `εI`, plus
   `γI` times `Fork`'s probability at the prover's program, of the transcript they make from the messages of Π_in's
   simulator `SI`, run against them as that one verifier. `SI` reads only Π_in's statement.

**The four conditions**, `EitherZK`'s hypotheses. The mechanism is De Santis and Yung's oblivious warden ("Cryptographic
Applications of the Non-Interactive Metaproof and Many-prover Systems", CRYPTO '90), which checks a proof and proves
that it checked from fresh coins of its own, over Ben-Or et al.'s commitment to a proof's messages with a proof that the
verifier accepts them ("Everything Provable is Provable in Zero-Knowledge", CRYPTO '88). The trust model is a machine
and its firewall (Mironov and Stephens-Davidowitz, "Cryptographic Reverse Firewalls", EUROCRYPT 2015), for proofs as
Ganesh, Magri and Venturi bound it ("Cryptographic Reverse Firewalls for Interactive Proof Systems", ICALP 2020): no
firewall hides whether a proof failed.

1. **Only in the witness** (`OnlyInWitness`): nothing the prover chooses reaches the verifier except inside Π_out's
   witness. Before the Π_out sessions the firewall's messages read the prover's messages only through the digests it
   commits, in `Hello` and in each round's public layout (`innerMsg`, `stage`). The Π_out sessions read them only
   through their witnesses (`wit`), as Ben-Sasson, Chiesa, Tromer and Virza's zero-knowledge proof-carrying data
   clears a message of all but what it must carry ("Scalable Zero Knowledge via Cycles of Elliptic Curves",
   CRYPTO 2014).
2. **Π_in is zero knowledge against the firewall** (`InnerZK`): `ZKmalR` at the prover's program against every
   malicious verifier of its session, whose coins come live from the verifier and which the firewall and a colluding
   verifier may play together.
3. **Split shares** (`SplitShares`): every commitment the firewall sends is under its own salts, uniform and
   independent of the rest of its tape, and each digest it commits reads no salt of its own round or a later one. The
   prover fixes its share before it learns the firewall's: coin tossing into the well, as in Chakraborty, Dziembowski
   and Nielsen's reverse firewalls for MPC ("Reverse Firewalls for Actively Secure MPCs", CRYPTO 2020).
4. **Check first** (`CheckFirst`): where its check fails, the firewall sends the refusal from one fixed position on,
   the first round after the staging rounds. Where the check passes at a history it sent, every V* session's witness is
   in Π_out's relation (`Honest`), so the Π_out sessions' simulators apply.

`FirewallZK` reads conditions 1, 3 and 4, and `ProverZK` reads 2. As for every record, the Π_out sessions (`outer`),
their simulators (`SO`) and the hm96 key are the deployment's, bound on the code side; the four conditions are what the
split adds.

**Where a condition isn't met** at today's deployment, at which `EitherZK` is therefore vacuous:
- 2: the deployed prover runs Π_in without `--zk`, so the firewall reads the prover's values.
- 1 and 3: the registered roots are in the statement in the clear, drawn by the prover alone, a channel the model
  can't see, since a statement is public; the split record carries them only as the firewall's commit strings in
  `Hello`. The deployed firewall also answers the prover, its salts among the answers, where the diode sends nothing
  back. `Flock.Guarantees.RecursiveZKLoopAnySum` covers a prover that reads the salts drawn so far.
- 4: the deployed firewall checks nothing first, so its proved bound pays every form of the release, the factor
  `|T| + 1 + |St|` of `RecursiveZKLoopAnySum`, where the check leaves two. The V* statement that checks Π_in's proof is
  built by hand, not generated from the Lean `V_in` or proved equal to it, so `Honest` isn't discharged.

Timing is outside both claims: that the firewall's release times are fixed, so the prover's latency doesn't reach the
view, is its deployment's.

**Scope: the witness, at the covered forms.** Both claims hide what the statement doesn't hold: the values (weights,
data, intermediates), their openings and their salts. The covered forms are those the compiled verifier accepts
(`Claims.Recursion.InScope`, which `FirewallZK` takes): no circuit commitment, and the dense form only, since
`RecSetup.denseOnly` refuses the holographic and sparse ones. At them Π_in's circuit is in the statement, public, and
neither claim hides it: `SimSplit` reads the statement, and `SI` reads Π_in's. A form that hides the circuit, the
universal unit's (`Claims.Recursion.RPriv`, which hides it up to its size class), is outside the covered forms until
the compiled verifier accepts it and a proof covers it. A plain holographic index binds the circuit without hiding it.

**Premises.**
- `Hm96Hiding` at the recursion's pinned key, which `FirewallZK` reads as `Claims.Recursion.ZK` does; `ProverZK`'s
  errors are its argument, as `Flock.ZK`'s are.
- The coin tree's binding is the `Fork` term in both. In `FirewallZK` it is at the firewall's own nonce. In `ProverZK`
  the nonce is one the firewall and the verifier choose, so there it is SHA-512's collision resistance at a chosen key.

**The relation at a wrapped statement**, `RWrap`, is `Claims.Recursion.R` where the statement carries the registration
leaf `ϱ` in place of `r_P` and the witness carries `r_P` and the salt `s`. Two openings of one `ϱ` are a collision of
SHA-512's, so knowledge soundness at it costs one more collision term than at `R`. The locked claims read `R`.

**Relations.**
- `Claims.Recursion.ZK`, `ZKPriv` and `ZKStop` are the recursion's zero knowledge at its reference program
  (`ZkDeploy.prover`), one honest party that holds the witness and makes the release. `FirewallZK` is `ZKStop`'s form
  with that program split into the firewall and any prover. Its one stop the simulator can't read, the firewall's
  check, is paid by the factor `2`, as a one-sided bound.
- The proved cores, which neither claim is proved from yet, are the following.
  - `Flock.Guarantees.MaskedZK` is `Flock.ZK` at the deployed masked record. `InnerZK` is that statement at a nonce
    from the verifier's answer and a registration in place of `Hello`.
  - `Flock.Guarantees.RecursiveZK` takes any Π_in messages under the firewall's fresh salts, the Π_out sessions'
    simulator, and a stop of a public family, which leaks `log₂` of the family's size. `FirewallZK` asks for it at the
    release's two forms, over `V_out`'s transcript.

**Status.** A draft: nothing here is frozen, no guarantee reads it, and it is in no lock.
-/

namespace Claims.Recursion.Either

open Claims
open scoped NNReal

/-! ## The links -/

/-- **A firewall program** on the tape `Ωf`: its next message from the release's history so far (its messages and the
verifier's answers) and the prover's messages, one a round, the current round's last. It reads nothing else of the
prover's. -/
structure Firewall (Ωf : Type) where
  next : Msg → List (Msg × Msg) → List Msg → Ωf → OC Msg

/-- **The prover's messages at the release's history** `hist`, for a prover `P` on the tape `ω` that reads all of it,
as a colluding verifier could relay it: one for each round so far and the current round's, each at the history before
it. Its SHA-512 calls are evaluated, not the release's. -/
def msgsAt (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (hist : List (Msg × Msg)) : List Msg :=
  (List.range (hist.length + 1)).map fun i => ((P.next x (hist.take i) ω).eval H).1

/-- The prover `P`'s messages at the verifier's answers `bs`, each round's from its messages so far paired with the
answers (`acc`), the current round's last. Its SHA-512 calls are evaluated, not the firewall's. -/
def proverMsgs (H : Msg → Msg) {Ωp : Type} (P : Prover Ωp) (x : Msg) (ω : Ωp) :
    List (Msg × Msg) → List Msg → List Msg
  | acc, [] => acc.map Prod.fst ++ [((P.next x acc ω).eval H).1]
  | acc, b :: bs => proverMsgs H P x ω (acc ++ [(((P.next x acc ω).eval H).1, b)]) bs

/-- **The firewall `F` and the prover `P` as one program**, on both tapes: at each history, `F`'s next message at the
prover's messages, which hear only the verifier's answers, its coins. -/
def Firewall.join (H : Msg → Msg) {Ωf Ωp : Type} (F : Firewall Ωf) (P : Prover Ωp) : Prover (Ωf × Ωp) :=
  ⟨fun x hist ω => F.next x hist (proverMsgs H P x ω.2 [] (hist.map Prod.snd)) ω.1⟩

/-- The verifier's answer to the prover's last message in `ws`, where the firewall `F` at its tape `ωf` and the
verifier `Vs` at its tape `θ` answer together from the release's history `hist` and the prover's earlier messages `pre`:
each round, `F`'s message at the prover's messages so far, then the verifier's answer to it, until the verifier
decides. -/
def Firewall.answer {Ωf Θ : Type} (F : Firewall Ωf) (ωf : Ωf) (Vs : VStar Θ) (x : Msg) (θ : Θ) :
    List (Msg × Msg) → List Msg → List Msg → OC Out
  | _, _, [] => .ret .rej
  | hist, pre, w :: ws =>
    (F.next x hist (pre ++ [w]) ωf).bind fun f =>
      (Vs.next x (hist.map Prod.fst ++ [f]) θ).bind fun o =>
        if ws.isEmpty then .ret o else
          match o with
          | .msg m => Firewall.answer F ωf Vs x θ (hist ++ [(f, m)]) (pre ++ [w]) ws
          | o => .ret o

/-- **The firewall `F` at its tape `ωf` and the verifier `Vs` as one verifier** of the prover's Π_in session, of `k`
rounds: its answer to the prover's messages is the verifier's (`Firewall.answer`). -/
def Firewall.against {Ωf Θ : Type} (F : Firewall Ωf) (ωf : Ωf) (k : Msg → ℕ) (Vs : VStar Θ) : VStar Θ where
  k := k
  next x ws θ := Firewall.answer F ωf Vs x θ [] [] ws

/-- The release's history as the verifier saw it. -/
def msgsOf (hist : List (Msg × Msg)) : List (Msg × Out) := hist.map fun p => (p.1, Out.msg p.2)

/-- **The commit string to the digest `d`** under the salt `y` at the record's key: hm96's `b ‖ c`, with `b = d ⊕ Φy`
on `d`'s first 64 bytes and `c = H(p_salt ‖ y)`. At the zero digest it is `Deploy.zeroCommit`'s. -/
def commitTo {Coin : Type} (D : Deploy Coin) (H : Msg → Msg) (d : Msg) (y : Fin D.saltLen → UInt8) : Msg :=
  List.ofFn (fun i : Fin 64 => d.getD i 0 ^^^ D.mask y i) ++ D.saltHash H y

/-! ## The split record -/

/-- **The split deployment**: the recursion's record with its Π_in commit strings (`StopDeploy`; the firewall's one stop
is its check, so `stops` isn't read), and the following.

* `inner`: the prover's masked record, Π_in's session's. `regOf x`: the prover's registration at Π_in's statement `x`,
  its registered roots, its first message. `helloAns`: the verifier's first answer, read as the coin tree's nonce and
  its commitment to every coin. `SI`: Π_in's simulator.
* `firewall x`: the firewall's program at the statement `x`, on its own nonce and rest. Its parts, which the conditions
  read: `helloMsg x ν cs`, its `Hello` at the nonce `ν` with the commit strings `cs` to the prover's registration,
  `nReg x` of them; `salt x r j l` and `digest x r ms j l`, the salt of its commit string `l` in round `j` at its rest
  `r` and the digest that string commits at the prover's messages `ms`; `own x r`, the rest of its rest, which the
  Π_out sessions run on; `check`, its check before the Π_out sessions (`V_in` on Π_in's transcript, and every V*
  statement) at the transcript and the prover's messages; `wit x r ms s`, V* session `s`'s witness; `outer`, the Π_out
  sessions in `--zk` at the transcript before them, their witnesses and `own`; `refusal`, its message in every round
  from there where the check fails.
* `leafPrefix`, `wrapped` and `unwrap`, which `RWrap` reads: the registration leaf's prefix at the record's key, the
  registration leaf Π_in's statement carries, and Π_in's statement with a registered root in the leaf's place.

With an honest prover, `firewall` and `refProver` are the reference program `P` that `Complete` reads, split in two;
binding the two to it, and `firewall` to the deployed firewall, is the code side's, as for every record. -/
structure SplitDeploy (Coin Col T : Type) extends StopDeploy Coin where
  inner : Flock.ZkDeploy Coin Col T
  regOf : Msg → Msg
  helloAns : Msg → inner.Nonce × Msg
  SI : Simulator
  firewall : (x : Msg) → Firewall (Nonce x × Rest x)
  nReg : Msg → ℕ
  helloMsg : (x : Msg) → Nonce x → List Msg → Msg
  salt : (x : Msg) → Rest x → ℕ → ℕ → Fin saltLen → UInt8
  digest : (x : Msg) → Rest x → List Msg → ℕ → ℕ → Msg
  Own : Msg → Type
  own : (x : Msg) → Rest x → Own x
  check : Msg → List (Msg × Msg) → List Msg → Bool
  wit : (x : Msg) → Rest x → List Msg → ℕ → Flock.Wit
  outer : (x : Msg) → List (Msg × Out) → (ℕ → Flock.Wit) → Own x → List (Msg × Msg) → OC Msg
  refusal : Msg
  leafPrefix : Msg
  wrapped : Msg → Option Msg
  unwrap : Msg → Msg → Msg

section Split

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- Π_in's rounds after the registration, at the statement's Π_in statement. -/
def SplitDeploy.nI (D : SplitDeploy Coin Col T) (x : Msg) : ℕ := D.inner.V.k (xI D.toDeploy x)

/-- The release's rounds before the Π_out sessions: `Hello`, Π_in's rounds and the staging rounds. -/
def SplitDeploy.pre (D : SplitDeploy Coin Col T) (x : Msg) : ℕ := D.nI x + 1 + D.kS x

/-- The firewall's commit strings in round `j` before the Π_out sessions: the registration's `nReg` in `Hello`, Π_in
round `j − 1`'s `nAt`, and `n_C` in each staging round. -/
def SplitDeploy.nRound (D : SplitDeploy Coin Col T) (x : Msg) (j : ℕ) : ℕ :=
  if j = 0 then D.nReg x else if j ≤ D.nI x then D.nAt x (j - 1) else D.nC x

/-- The firewall's commit strings in round `j` at its rest `r` and the prover's messages `ms`. -/
def SplitDeploy.commits (D : SplitDeploy Coin Col T) (H : Msg → Msg) (x : Msg) (r : D.Rest x) (ms : List Msg)
    (j : ℕ) : List Msg :=
  (List.range (D.nRound x j)).map fun l => commitTo D.toDeploy H (D.digest x r ms j l) (D.salt x r j l)

/-- **The prover's reference program** in its Π_in session at the witness `w` and the rest `r`. Its first message is
its registration (`regOf`), since the firewall sends `Hello`. The verifier's answer to it is read as the coin tree's
nonce `ν` and its commitment `ρ` (`helloAns`). Then, if every opening so far passes the masked record's check against
`ρ` under `ν`, it is the masked record's reference prover at the history with each opening read as its coin, at the
tape `(ν, r)`; else `abort`. After the session's `k + 1` rounds it sends the empty message. Its nonce is `PUnit`: the
prover draws none. -/
def SplitDeploy.refProver (D : SplitDeploy Coin Col T) (w : Flock.Wit) : Prover (PUnit × D.inner.Rest) :=
  ⟨fun x hist ω => if D.inner.V.k x < hist.length then .ret [] else
    match hist with
    | [] => .ret (D.regOf x)
    | (_, a) :: ops =>
      (Flock.allPass (D.inner.chk x (D.helloAns a).1 (D.helloAns a).2) 0 (ops.map Prod.snd)).bind fun ok =>
        if ok then
          (D.inner.P w).next x (ops.map fun p => (p.1, D.inner.coinOf p.2)) (D.inner.tapeOf ((D.helloAns a).1, ω.2))
        else .ret D.inner.abort⟩

/-- **The release** at the statement `x` with the prover `P`: the firewall's program at `x`, given the prover's
messages at each history (`msgsAt`). Its tape is the firewall's nonce and rest and the prover's tape. -/
def SplitDeploy.release {Ω : Type} (D : SplitDeploy Coin Col T) (H : Msg → Msg) (x : Msg) (P : Prover Ω) :
    Prover (D.Nonce x × (D.Rest x × Ω)) :=
  ⟨fun x' hist ω => (D.firewall x).next x' hist (msgsAt H P x' ω.2.2 hist) (ω.1, ω.2.1)⟩

/-- The history `hist` is the firewall's at its nonce `ν`, its rest `r` and the prover's messages `ms`: each of its
messages is the one the firewall sends at the history before it. -/
def SplitDeploy.Sent (D : SplitDeploy Coin Col T) (H : Msg → Msg) (x : Msg) (ν : D.Nonce x) (r : D.Rest x)
    (hist : List (Msg × Msg)) (ms : List Msg) : Prop :=
  ∀ i (h : i < hist.length),
    (((D.firewall x).next x (hist.take i) (ms.take (i + 1)) (ν, r)).eval H).1 = (hist[i]'h).1

/-! ## The simulator and the error -/

/-- **`FirewallZK`'s simulator** at the Π_out simulators `SO`. Its coins are a nonce, a salt for each of the firewall's
commit strings before the Π_out sessions, one bit, and each V* session's simulator's coins. It sends the firewall's
messages before the Π_out sessions, each commit string to the zero digest: `Hello` at the nonce, then each Π_in round's
public layout (`innerMsg`) and each staging message (`stage`). Then, at the bit, each V* session by `SO` at its
statement, or the refusal in every round left. -/
def SimSplit (D : SplitDeploy Coin Col T) (SO : Msg → ℕ → OSim) : Simulator where
  C x := D.Nonce x × ((j : Fin (D.pre x)) → Fin (D.nRound x j) → Fin D.saltLen → UInt8) × Bool ×
    ((s : Fin (D.nS x)) → (SO x s).C)
  prog x c :=
    spBind (simRounds (fun ms => if h : ms.length < D.pre x then
        spBind (D.toDeploy.zeroCommits
            ((List.finRange (D.nRound x (⟨ms.length, h⟩ : Fin (D.pre x)))).map (c.2.1 ⟨ms.length, h⟩))) fun cs =>
          if ms.length = 0 then .ret (D.helloMsg x c.1 cs)
          else if ms.length ≤ D.nI x then spOf (D.innerMsg x ms cs) else spOf (D.stage x ms cs)
      else .ret []) (D.pre x) []) fun ms =>
      if c.2.2.1 then simOuter SO x (D.vstar x ms) c.2.2.2 (List.finRange (D.nS x)) ms
      else simRounds (fun _ => .ret D.refusal) (D.V.k x - D.pre x) ms

/-- **`FirewallZK`'s error** at the attempt budget `t`, where hm96 leaves hide up to `δ₁`: each V* session's `--zk`
simulation error (`Nums.εzk`), and `2δ₁` for each commit string the firewall sends before the Π_out sessions, in each
of the release's two forms. -/
noncomputable def εzk (D : SplitDeploy Coin Col T) (t : ℕ) (δ₁ : ℝ) (x : Msg) : ℝ :=
  ∑ s : Fin (D.nS x), (D.nums x s).εzk t δ₁ + 2 * (2 * ∑ j : Fin (D.pre x), D.nRound x j) * δ₁

/-! ## The four conditions -/

/-- **Condition 1, only in the witness.** Before the Π_out sessions, the firewall's messages read the prover's messages
only through the digests it commits: at `Hello` its nonce and its commit strings to the registration, in each Π_in
round the round's public layout (`innerMsg`) at its commit strings, in each staging round the staging message (`stage`)
at its commit strings. From there, where its check passes, its messages are the Π_out sessions', which read the prover's
messages only through their witnesses. -/
def OnlyInWitness (H : Msg → Msg) (D : SplitDeploy Coin Col T) : Prop :=
  ∀ x hist ms (ν : D.Nonce x) (r : D.Rest x),
    (hist.length < D.pre x → (D.firewall x).next x hist ms (ν, r) =
      if hist.length = 0 then .ret (D.helloMsg x ν (D.commits H x r ms 0))
      else if hist.length ≤ D.nI x then D.innerMsg x (msgsOf hist) (D.commits H x r ms hist.length)
      else D.stage x (msgsOf hist) (D.commits H x r ms hist.length)) ∧
    (D.pre x ≤ hist.length → D.check x (hist.take (D.pre x)) (ms.take (D.pre x)) = true →
      (D.firewall x).next x hist ms (ν, r) =
        D.outer x (msgsOf (hist.take (D.pre x))) (D.wit x r (ms.take (D.pre x))) (D.own x r) (hist.drop (D.pre x)))

/-- **Condition 2, Π_in is zero knowledge against the firewall.** The prover's session is masked, and `ZKmalR` holds at
its reference program against every malicious verifier of its `k + 1` rounds (the registration round, then `V_in`'s),
at Π_in's simulator `SI`, within `εI` plus `γI` times the probability that `Fork` finds a collision among the prover's
SHA-512 calls in two real runs. The coins are the verifier's, live: the firewall draws none of them. -/
def InnerZK (H : Msg → Msg) (D : SplitDeploy Coin Col T) (εI γI : Msg → ℝ) : Prop :=
  D.inner.zk = true ∧ ZKmalR H (fun x => D.inner.V.k x + 1) D.refProver (Flock.R D.inner.parse) D.SI εI γI

/-- **Condition 3, split shares.** The firewall's rest is its salts, one for each commit string before the Π_out
sessions, and the rest of its own (`own`), uniform and independent; and each digest it commits reads no salt of its own
round or a later one, so the prover fixes its share before it learns the firewall's. -/
def SplitShares (D : SplitDeploy Coin Col T) : Prop :=
  ∀ x, Function.Bijective (fun r : D.Rest x =>
      ((fun (j : Fin (D.pre x)) (l : Fin (D.nRound x j)) => D.salt x r j l), D.own x r)) ∧
    ∀ (r r' : D.Rest x) ms j l, D.own x r = D.own x r' →
      (∀ i < j, ∀ l' < D.nRound x i, D.salt x r i l' = D.salt x r' i l') →
        D.digest x r ms j l = D.digest x r' ms j l

/-- **Condition 4, check first.** Where its check fails, the firewall sends the refusal in every round from the first
after the staging rounds on. Where the check passes at a history it sent, every V* session's witness is in Π_out's
relation at that session's statement (`Honest`). -/
def CheckFirst (H : Msg → Msg) (D : SplitDeploy Coin Col T) (DO : Flock.Deploy Coin Col' T') : Prop :=
  ∀ x hist ms (ν : D.Nonce x) (r : D.Rest x),
    (D.pre x ≤ hist.length → D.check x (hist.take (D.pre x)) (ms.take (D.pre x)) = false →
      (D.firewall x).next x hist ms (ν, r) = .ret D.refusal) ∧
    (hist.length = D.pre x → D.Sent H x ν r hist ms → D.check x hist (ms.take (D.pre x)) = true →
      ∀ s : Fin (D.nS x),
        Flock.R DO.parse ((D.vstar x (msgsOf hist)).getD s []) (D.wit x r (ms.take (D.pre x)) s))

/-! ## The claims -/

/-- **An honest firewall, any prover.** At every statement of the covered forms (it parses into `InScope`), against
every prover program and every malicious verifier of `V_out`'s rounds, with the Π_out sessions in `--zk`, at each
attempt budget `t ≥ 1` and wherever hm96 leaves hide up to `δ₁` at the deployed key:
$$\Pr[\mathrm{Ev}(\text{release})] \le 2\cdot\Pr[\mathrm{Ev}(\mathrm{SimSplit})] + \varepsilon_{zk}(t,\delta_1,x) +
\gamma_{zk}(t,x)\cdot\Pr[\mathrm{Fork}].$$
It holds at every such statement: the prover needn't hold a witness. -/
def FirewallZK (H : Msg → Msg) (D : SplitDeploy Coin Col T) (DO : Flock.Deploy Coin Col' T') : Prop :=
  DO.zk = true ∧ ∀ t, 0 < t → ∀ δ₁ : ℝ≥0, Hm96Hiding D.mask (D.toDeploy.saltHash H) δ₁ →
    ∀ x, (∃ s, D.parse x = some s ∧ InScope s) →
      ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (Θ : Type) (Vs : VStar Θ) (θ : Θ),
        Vs.k x = D.V.k x → ∀ Ev : List (Msg × Out) → Prop,
          Pr (fun ω : D.Nonce x × (D.Rest x × Ω) => Ev (runStar H (D.release H x P) Vs x ω θ).msgs) ≤
            2 * Pr (fun c : (SimSplit D (D.SO t)).C x => Ev ((SimSplit D (D.SO t)).out H Vs θ x c)) +
              εzk D t δ₁ x + γzk D.toDeploy t x * (Fork H (D.release H x P) Vs θ x).Pr H

/-- **An honest prover, any firewall.** At every statement and witness of Π_in's relation, every firewall program `F`
at every tape `ωf` and every malicious verifier, on every event of the release's transcript (the joint view of `F` and
the verifier): the release with the honest prover is within `εI` of the release with the messages of `SI`, run against
`F` and the verifier as one verifier of the prover's `k + 1` rounds (`Firewall.against`), plus `γI` times the
probability that `Fork` finds a collision among the prover's SHA-512 calls in two real runs against that verifier. -/
def ProverZK (H : Msg → Msg) (D : SplitDeploy Coin Col T) (εI γI : Msg → ℝ) : Prop :=
  ∀ x w, Flock.R D.inner.parse x w →
    ∀ (Ωf : Type) (F : Firewall Ωf) (ωf : Ωf) (Θ : Type) (Vs : VStar Θ) (θ : Θ) (Ev : List (Msg × Out) → Prop),
      |Pr (fun ω : PUnit × D.inner.Rest => Ev (runStar H (Firewall.join H F (D.refProver w)) Vs x (ωf, ω) θ).msgs) -
        Pr (fun c : D.SI.C x => Ev (runStar H (Firewall.join H F (replay
          ((D.SI.out H (Firewall.against F ωf (fun x => D.inner.V.k x + 1) Vs) θ x c).map Prod.fst))) Vs x (ωf, ())
            θ).msgs)| ≤
        εI x + γI x * (Fork H (D.refProver w) (Firewall.against F ωf (fun x => D.inner.V.k x + 1) Vs) θ x).Pr H

/-- **Zero knowledge if the firewall or the prover is honest**: at a split record that meets the four conditions,
`FirewallZK` and `ProverZK`. -/
def EitherZK (H : Msg → Msg) (D : SplitDeploy Coin Col T) (DO : Flock.Deploy Coin Col' T') (εI γI : Msg → ℝ) :
    Prop :=
  OnlyInWitness H D → InnerZK H D εI γI → SplitShares D → CheckFirst H D DO →
    FirewallZK H D DO ∧ ProverZK H D εI γI

/-! ## The relation at a wrapped statement -/

/-- **The registration leaf** `ϱ = Leaf(r_P; s)`: the hm96 leaf of the prover's registered root `r_P` under the
firewall's salt `s`, `Hash.Hm96.leaf`'s form at the record's key, mask and leaf prefix. -/
def regLeaf (H : Msg → Msg) (D : SplitDeploy Coin Col T) (rP : Msg) (s : Fin D.saltLen → UInt8) : Msg :=
  H (D.leafPrefix ++ commitTo D.toDeploy H rP s)

/-- **The recursion's relation at a wrapped statement**: `Claims.Recursion.R` where the statement carries the
registration leaf `ϱ` in place of the prover's root `r_P`, and the witness carries `r_P` and the salt `s` beside Π_in's
witness. The statement parses with no circuit commitment, its Π_in statement carries `Leaf(r_P; s)` (`wrapped`), and
Π_in's relation holds at that statement with `r_P` in the leaf's place (`unwrap`). In the release the leaf's commit
string is the firewall's, in its `Hello`; at knowledge soundness the leaf is in the statement, registered before any
coin. -/
def RWrap {Wi : Type} (H : Msg → Msg) (D : SplitDeploy Coin Col T) (I : Inner Coin Wi) :
    Rel (Msg × (Fin D.saltLen → UInt8) × Wi) := fun x p =>
  ∃ s, D.parse x = some s ∧ s.circ = none ∧ D.wrapped s.inner = some (regLeaf H D p.1 p.2.1) ∧
    I.R (D.unwrap s.inner p.1) p.2.2

end Split

end Claims.Recursion.Either
