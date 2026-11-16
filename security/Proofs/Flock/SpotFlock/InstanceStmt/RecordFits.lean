module

public import Proofs.Flock.SpotFlock.InstanceStmt.ReadyZk
public import Proofs.Flock.Claims.Complete.ZkTables.RecordText
import Proofs.Flock.Soundness.Discharge.Hidden.Theorems
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Claims.Complete.ZkTables.Stmt
import Proofs.Flock.Claims.Recursion.HonestVerdictFinal
import Proofs.Flock.Claims.Complete.ZkTables.HelloSize
import Proofs.Flock.Claims.Complete.Json.CanonSize

/-!
# The two-instance statement's honest record fits its length word, from its spec's text

At `SpotFlock.InstanceStmt.x r` the setup's one table is `setupHidden`'s (`setupTables_okAny`), so the structural facts
`recordFitsZk_of_text` reads (the leaf scheme, the range, `m`, the `Hello` and the keyed coin tree) come from its `Spec`
and from `tabX_coinTree`, and the record fits where its spec's text (`specText`) is under `2^55`.
-/

@[expose] public section
open Flock RecDeploy.VStarNets
open Properties.Flock.Nets (FixedNetsBuild)
open FlockComplete (RecordFitsZkAt jsz)
open Lean (Json)

set_option maxHeartbeats 200000000
set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt

theorem setup_tabAt (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) :
    Stmt.setupTables tz (bytesOf (HmNets.shaLaid ()) (HmNets.hmLaid ()))
      (pubOf r (hexOf circuitHash) (hexOf classHash)) coinsX 1 [] (some sampleJ) (some partObj) (some progBytes)
      none none none none = .ok ((tabAt r).1, (tabAt r).2) := by
  have hp := FlockDeploy.fixedNets_parse hn
  have hCe : classHash.isEmpty = false := classOfB_isEmpty (HmNets.shaLaid ()) (HmNets.hmLaid ())
  exact setupTables_X r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X r (hexOf circuitHash) (hexOf classHash))
    (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX hr rfl (classOf_eq _ _)
    (FlockDeploy.sha_lines hn) (FlockDeploy.hm_lines hn) hp.1.2 hp.2.2 hc1_of hc2_of
    (fun c h1 h2 => FlockDeploy.Concrete.check_of hp.1.1 hp.2.1 c h1 h2) hCe (hu1_of _)
    (hu2_of _ _ _ _ _ _ _ _ _ hCe) (hd_of _ _ _) (hd2_of _ _ _ _ _ _ _ _ _ _ _ _) hid_of hhello_of

/-- **The statement's honest masked record fits its length word** where its spec's text is under `2^55`: the table is
`setupHidden`'s, of `25 ≤ m ≤ 35` (`zkIn_x`), under `hm96-sha512/v1`, with `k_log ≤ m`, at most `2^10` regions and
`m_pts ≤ 64` (its `Spec`), at two link points, a keyed coin tree and a `Hello` that reads back. -/
theorem recordFits_x_of_text (hn : FixedNetsBuild) {r : Run} (hr : r.Plain)
    (hfit : FlockComplete.ZkTables.specText (tabAt r).1 < 2 ^ 55) :
    RecordFitsZkAt FlockDeploy.sha512 (x r) (tabAt r).1 (tabAt r).2 := by
  have hset := setup_tabAt hn hr
  have hct0 : (tabAt r).1.coinTree.isSome = true := tabX_coinTree r _ _ _ _ _ _ _ _ _ _ _ _ _ _
  have hz := zkIn_x hn hr
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny hset with
    ⟨-, st, hd, hspec, hsts⟩ | ⟨hc, -⟩
  · rw [FlockSoundness.Discharge.HiddenExec.setupOne_hidden hidden_tzX] at hd
    have hsp := FlockSoundness.Discharge.Hidden.setupHidden_spec hd
    have hms : st.merkle = MerkleScheme.hm96Sha512 :=
      FlockSoundness.Discharge.HiddenExec.setupHidden_hm96 (by decide +kernel) hd
    have hrange := (checkInRange_ok _ _ _).1 hsp.inRange
    have hmem : st ∈ (tabAt r).2.toList := by rw [hsts]; simp
    obtain ⟨h25, h35, -⟩ := hz.tab st hmem
    obtain ⟨ct, hct⟩ : ∃ ct, st.spec.coinTree = some ct := by
      rw [hspec] at hct0
      exact Option.isSome_iff_exists.1 hct0
    obtain ⟨m, kvs, hown, -, -, -, hg⟩ := FlockComplete.ZkTables.hello_ok hsp.hello
    obtain ⟨hz', hc, -⟩ := FlockComplete.JsonRT.canon_ok _ hg
    rw [hspec] at hfit
    rw [hspec, hsts]
    exact FlockComplete.ZkTables.recordFitsZk_of_text FlockDeploy.out64_sha512 (t := st.tags.table) rfl hct hown hc st
      h25 h35 hms (by have := hsp.m; omega) hrange.2.1 rfl (by have := hsp.mPts; omega) (Nat.le_refl 2) hfit
  · exact absurd hc (Nat.lt_irrefl 1)

theorem repStr_size (n : ℕ) (hn : n < 2) : (toString n).utf8ByteSize ≤ 4 := by
  have h1 : (toString n).length ≤ 1 :=
    (Nat.length_repr_le_iff (n := n) (k := 1) (by decide)).2 (by interval_cases n <;> norm_num)
  have h2 := FlockComplete.size_le_length (toString n)
  omega

theorem domain_size (rep : ℕ) (hrep : rep < 2) : (tz.domain rep).utf8ByteSize ≤ 1004 := by
  have h1 : tz.domainPrefix.utf8ByteSize ≤ 1000 := by decide +kernel
  have h2 := repStr_size rep hrep
  unfold Tags.domain
  rw [String.utf8ByteSize_append]
  omega

/-- **The statement's spec text is small**: its `Hello` (`helloLen_le`), its sample, its coin tree, `Σ`, its table's
name and its two domains weigh far under `2^55`. -/
theorem specText_x_lt (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) :
    FlockComplete.ZkTables.specText (tabAt r).1 < 2 ^ 55 := by
  have hset := setup_tabAt hn hr
  have hctree : (tabAt r).1.coinTree = some (Zk.coinSpec (t0.tableNames 1)) := by kernel_rfl'
  have hunit : jsz ((tabAt r).1.unitSample.getD Json.null) ≤ 100000 := by decide +kernel
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny hset with ⟨-, st, hd, hspec, hsts⟩ | ⟨hc, -⟩
  · rw [FlockSoundness.Discharge.HiddenExec.setupOne_hidden hidden_tzX] at hd
    have hsp := FlockSoundness.Discharge.Hidden.setupHidden_spec hd
    have hrange := (checkInRange_ok _ _ _).1 hsp.inRange
    have hσ : st.sigma.size = 64 := by rw [hsp.sigma]; exact Sha512.hash_size _
    have hl := FlockComplete.ZkTables.helloLen_le (tags := tz) (sigma := st.sigma) (mPts := PT_LOCAL + st.nbl)
      (ct := Zk.coinSpec (t0.tableNames 1)) (names := #[tz.table]) hsp.hello hσ hrange.2.2 (spec := st.spec) rfl
      (by rw [← hspec]; exact hctree)
    have hT : st.spec.tables = #[tz.table] := by
      show #[st.tags.table] = _
      rw [hsp.tagsEq]
    have hD : ∀ rep, st.spec.domain tz.table rep = tz.domain rep := fun rep => by
      show st.tags.domain rep = _
      rw [hsp.tagsEq]
    have hG : st.spec.tables.getD 0 "" = tz.table := by rw [hT]; rfl
    have hS : st.spec.sigma.size = 64 := hσ
    have c1 : jsz (Json.arr ((#[tz.table]).map Json.str)) ≤ 1000 := by decide +kernel
    have c2 : jsz ((Zk.coinSpec (t0.tableNames 1)).coins "") ≤ 1000000 := by decide +kernel
    have c3 : jsz (Zk.coinSpec (t0.tableNames 1)).toJson ≤ 1000000 := by decide +kernel
    have c5 : tz.table.utf8ByteSize ≤ 1000 := by decide +kernel
    have c6 := domain_size 0 (by decide)
    have c7 := domain_size 1 (by decide)
    rw [hspec] at hctree hunit ⊢
    unfold FlockComplete.ZkTables.specText FlockComplete.ZkTables.coinTreeText
    rw [hctree, hG, hD, hD]
    simp only [Option.getD_some]
    omega
  · exact absurd hc (Nat.lt_irrefl 1)

/-- **The statement's honest masked record fits its length word.** -/
theorem recordFits_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) :
    RecordFitsZkAt FlockDeploy.sha512 (x r) (tabAt r).1 (tabAt r).2 :=
  recordFits_x_of_text hn hr (specText_x_lt hn hr)

end SpotFlock.InstanceStmt
