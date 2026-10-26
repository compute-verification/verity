import Flock.With.SimSetup
import Proofs.Flock.Claims.Complete.CostWalk

/-!
# The setup's hash calls, counted function by function, in any hash

The counting oracle (`Hashes.counted h`) runs a `With` function and counts its SHA-512 calls. `countC` is the cost
bound on it (`FlockComplete.Cost`): from any count, a run adds at most `n` calls. Each function the setup calls a
bounded number of times is bounded here, for every `h`: the digests (`partitionDigestWith`, `publicDigestWith`), the
decoders (`Program.decodeWith`, `Desc.parseWith`, `HmRow.parseWith`), the loads and draws of a public file, the
partition's checks and laws, and the links check. Each proof walks the definition with `countC`'s rules: the walk
assigns each bound, and `omega` closes what it leaves. The walk is spelled out in each proof rather than kept as a
macro, which only a `compile_time` module may hold.
-/

namespace RecDeploy.SetupCount

open Flock FlockComplete
open Lean (Json)

theorem go_bind {α β : Type} (x : CountV α) (f : α → CountV β) (k : Nat) :
    (x >>= f).go k = match x.go k with
      | (.ok a, k') => (f a).go k'
      | (.error e, k') => (.error e, k') := by
  simp only [StV.go, Id.run, ExceptT.run_bind, StateT.run_bind]
  rcases hr : x.run.run k with ⟨r, k'⟩
  cases r <;> rfl

/-- **The counting oracle's cost bound**: from any count, a run adds at most `n` calls. -/
def countC : Cost CountV where
  P x n := ∀ k, (x.go k).2 ≤ k + n
  mono h hnm k := by have := h k; omega
  pure _ k := Nat.le_add_right _ _
  bind {α β x f n m} hx hf k := by
    rw [go_bind]
    have h1 := hx k
    rcases hr : x.go k with ⟨r, k'⟩
    rw [hr] at h1
    cases r with
    | error e => dsimp only at h1 ⊢; omega
    | ok a => have := hf a k'; dsimp only at h1 ⊢; omega
  throw _ _ k := Nat.le_add_right _ _
  lift v k := by cases v <;> exact Nat.le_add_right _ _

theorem sha_cost (h : ByteArray → ByteArray) (b : ByteArray) : countC.P ((Hashes.counted h).sha512 b) 1 :=
  fun _ => Nat.le_refl _

/-- A `throw` at no calls, so that the walk assigns its bound. -/
theorem throw0 {α : Type} (e : String) : countC.P (throw e : CountV α) 0 := countC.throw e 0

/-! The walk's rules at a bound an earlier branch already fixed: a `bind` within it, a branch in it on both sides. -/

theorem throwBind {α β : Type} (e : String) (f : α → CountV β) :
    countC.P ((throw e : CountV α) >>= f) 0 := fun k => by
  rw [go_bind]; exact Nat.le_add_right _ _

theorem throwBindAny {α β : Type} (e : String) (f : α → CountV β) (N : Nat) :
    countC.P ((throw e : CountV α) >>= f) N := fun k => by
  rw [go_bind]; exact Nat.le_add_right _ _

theorem bindAny {α β : Type} {x : CountV α} {f : α → CountV β} {n m N : Nat} (hx : countC.P x n)
    (hf : ∀ a, countC.P (f a) m) (hN : n + m ≤ N) : countC.P (x >>= f) N :=
  countC.mono (countC.bind hx hf) hN

theorem pureAny {α : Type} (a : α) (N : Nat) : countC.P (pure a : CountV α) N :=
  countC.mono (countC.pure a) (Nat.zero_le _)

theorem liftAny {α : Type} (v : V α) (N : Nat) : countC.P (monadLift v : CountV α) N :=
  countC.mono (countC.lift v) (Nat.zero_le _)

theorem iteAny {α : Type} {c : Prop} [Decidable c] {x y : CountV α} {N : Nat} (hx : c → countC.P x N)
    (hy : ¬c → countC.P y N) : countC.P (if c then x else y) N := by
  by_cases hc : c
  · rw [ite_eq_left hc]; exact hx hc
  · rw [ite_eq_right hc]; exact hy hc

theorem diteMax {α : Type} {c : Prop} [Decidable c] {x : c → CountV α} {y : ¬c → CountV α} {n m : Nat}
    (hx : ∀ h, countC.P (x h) n) (hy : ∀ h, countC.P (y h) m) : countC.P (if h : c then x h else y h) (max n m) := by
  by_cases hc : c
  · rw [dite_eq_left hc]; exact countC.mono (hx hc) (Nat.le_max_left _ _)
  · rw [dite_eq_right hc]; exact countC.mono (hy hc) (Nat.le_max_right _ _)

theorem diteAny {α : Type} {c : Prop} [Decidable c] {x : c → CountV α} {y : ¬c → CountV α} {N : Nat}
    (hx : ∀ h, countC.P (x h) N) (hy : ∀ h, countC.P (y h) N) : countC.P (if h : c then x h else y h) N := by
  by_cases hc : c
  · rw [dite_eq_left hc]; exact hx hc
  · rw [dite_eq_right hc]; exact hy hc

/-- A loop over a range whose steps make no call. -/
theorem forIn_range0 {β : Type} (r : Std.Legacy.Range) (b : β) {f : Nat → β → CountV (ForInStep β)}
    (hf : ∀ a b, countC.P (f a b) 0) : countC.P (forIn r b f) 0 := by
  have := countC.forIn_range r b hf
  rwa [Nat.mul_zero] at this

theorem forIn_range0Any {β : Type} (r : Std.Legacy.Range) (b : β) {f : Nat → β → CountV (ForInStep β)} {N : Nat}
    (hf : ∀ a b, countC.P (f a b) 0) : countC.P (forIn r b f) N :=
  countC.mono (forIn_range0 r b hf) (Nat.zero_le _)

theorem forIn_array0Any {α β : Type} (xs : Array α) (b : β) {f : α → β → CountV (ForInStep β)} {N : Nat}
    (hf : ∀ a ∈ xs, ∀ b, countC.P (f a b) 0) : countC.P (forIn xs b f) N :=
  countC.mono (countC.forIn_array0 xs b hf) (Nat.zero_le _)

theorem forIn_list0Any {α β : Type} (l : List α) (b : β) {f : α → β → CountV (ForInStep β)} {N : Nat}
    (hf : ∀ a ∈ l, ∀ b, countC.P (f a b) 0) : countC.P (forIn l b f) N :=
  countC.mono (countC.forIn_list0 l b hf) (Nat.zero_le _)

theorem mapM_array0Any {α β : Type} (xs : Array α) {f : α → CountV β} {N : Nat} (hf : ∀ a, countC.P (f a) 0) :
    countC.P (xs.mapM f) N :=
  countC.mono (countC.mapM_array xs hf) (Nat.zero_le _)

variable (h : ByteArray → ByteArray)

theorem partitionDigest_cost (obj : Json) : countC.P (HmRow.partitionDigestWith (Hashes.counted h) obj) 1 := by
  apply countC.mono
  · unfold HmRow.partitionDigestWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply sha_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply sha_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem publicDigest_cost (header : Json) (pubs : ByteArray) :
    countC.P (HmRow.publicDigestWith (Hashes.counted h) header pubs) 1 := by
  apply countC.mono
  · unfold HmRow.publicDigestWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply sha_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply sha_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem descParse_cost (bytes : ByteArray) : countC.P (TemplateQuery.Desc.parseWith (Hashes.counted h) bytes) 1 := by
  apply countC.mono
  · unfold TemplateQuery.Desc.parseWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply sha_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply sha_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

set_option maxHeartbeats 2000000 in
theorem decode_cost (bytes : ByteArray) : countC.P (Program.decodeWith (Hashes.counted h).sha512 bytes) 1 := by
  apply countC.mono
  · unfold Program.decodeWith
    repeat (first
      | extract_lets
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply sha_cost h)
      | (with_reducible apply throw0)
      | (with_reducible apply forIn_range0)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply sha_cost h)
        | (with_reducible apply throw0)
        | (with_reducible apply forIn_range0)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero]); omega

theorem classOf_cost (bytes : ByteArray) (head : String) (mj : Json) :
    countC.P (HmRow.classOfWith (Hashes.counted h) bytes head mj) 1 := by
  apply countC.mono
  · unfold HmRow.classOfWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply sha_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply sha_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem checkPartition_cost (c : Circuit) (obj : Option Json) :
    countC.P (HmRow.checkPartitionWith (Hashes.counted h) c obj) 1 := by
  unfold HmRow.checkPartitionWith
  extract_lets part rule digest
  clear_value part rule digest
  by_cases hr : rule = "verity/partition/v1"
  · subst hr
    cases obj with
    | none => simp only; exact countC.throw _ _
    | some o =>
      simp only
      apply countC.mono
      · repeat (first
          | extract_lets
          | (with_reducible apply throwBind)
          | (with_reducible apply Cost.bind)
          | (with_reducible apply bindAny)
          | (with_reducible apply partitionDigest_cost h)
          | (with_reducible intro _)
          | (with_reducible apply Cost.pure)
          | (with_reducible apply pureAny)
          | (with_reducible apply throw0)
          | (with_reducible apply Cost.throw)
          | (with_reducible apply Cost.lift)
          | (with_reducible apply liftAny)
          | (with_reducible apply Cost.forIn_array)
          | (with_reducible apply Cost.forIn_array0)
          | (with_reducible apply forIn_array0Any)
          | (with_reducible apply Cost.forIn_list)
          | (with_reducible apply Cost.forIn_list0)
          | (with_reducible apply forIn_list0Any)
          | (with_reducible apply forIn_range0)
          | (with_reducible apply forIn_range0Any)
          | (with_reducible apply Cost.mapM_array)
          | (with_reducible apply mapM_array0Any)
          | (with_reducible apply Cost.ite)
          | (with_reducible apply iteAny)
          | (with_reducible apply diteMax)
          | (with_reducible apply diteAny)
          | split)
        iterate 40
          all_goals try (dsimp +zetaDelta -zeta only)
          all_goals repeat (first
            | extract_lets
            | (with_reducible apply throwBind)
            | (with_reducible apply Cost.bind)
            | (with_reducible apply bindAny)
            | (with_reducible apply partitionDigest_cost h)
            | (with_reducible intro _)
            | (with_reducible apply Cost.pure)
            | (with_reducible apply pureAny)
            | (with_reducible apply throw0)
            | (with_reducible apply Cost.throw)
            | (with_reducible apply Cost.lift)
            | (with_reducible apply liftAny)
            | (with_reducible apply Cost.forIn_array)
            | (with_reducible apply Cost.forIn_array0)
            | (with_reducible apply forIn_array0Any)
            | (with_reducible apply Cost.forIn_list)
            | (with_reducible apply Cost.forIn_list0)
            | (with_reducible apply forIn_list0Any)
            | (with_reducible apply forIn_range0)
            | (with_reducible apply forIn_range0Any)
            | (with_reducible apply Cost.mapM_array)
            | (with_reducible apply mapM_array0Any)
            | (with_reducible apply Cost.ite)
            | (with_reducible apply iteAny)
            | (with_reducible apply diteMax)
            | (with_reducible apply diteAny)
            | split)
        all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
      · omega
  · cases obj with
    | none => simp only; exact countC.mono (countC.pure _) (Nat.zero_le _)
    | some o => simp only; exact countC.throw _ _

theorem templateStrata_cost (law : String) (obj : Json) (program : ByteArray) :
    countC.P (HmRow.templateStrataWith (Hashes.counted h) law obj program) 1 := by
  apply countC.mono
  · unfold HmRow.templateStrataWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply descParse_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply descParse_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem stratifiedLaw_cost (obj : Json) (program : ByteArray) (k : Nat) :
    countC.P (HmRow.stratifiedLawWith (Hashes.counted h) obj program k) 1 := by
  apply countC.mono
  · unfold HmRow.stratifiedLawWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply templateStrata_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply templateStrata_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem workLaw_cost (obj : Json) (program : ByteArray) (k : Nat) (table : Array (String × Nat × Nat)) :
    countC.P (HmRow.workLawWith (Hashes.counted h) obj program k table) 1 := by
  apply countC.mono
  · unfold HmRow.workLawWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply templateStrata_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply templateStrata_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem deriveTemplate_cost (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    countC.P (HmRow.deriveTemplateUnitsWith (Hashes.counted h) c obj program indices) 1 := by
  apply countC.mono
  · unfold HmRow.deriveTemplateUnitsWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply descParse_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply descParse_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero]); omega

theorem deriveQword_cost (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    countC.P (HmRow.deriveQwordUnitsWith (Hashes.counted h) c obj program indices) 1 := by
  apply countC.mono
  · unfold HmRow.deriveQwordUnitsWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply decode_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply decode_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem deriveQcall_cost (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    countC.P (HmRow.deriveQcallUnitsWith (Hashes.counted h) c obj program indices) 1 := by
  apply countC.mono
  · unfold HmRow.deriveQcallUnitsWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply decode_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply decode_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero]); omega

theorem qwordFinding_cost (c : Circuit) (pub : Public) (obj : Json) (program : ByteArray) (f : HmRow.PartitionFinding) :
    countC.P (HmRow.qwordFindingWith (Hashes.counted h) c pub obj program f) 1 := by
  apply countC.mono
  · unfold HmRow.qwordFindingWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply deriveQword_cost h)
      | (with_reducible apply deriveQcall_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply deriveQword_cost h)
        | (with_reducible apply deriveQcall_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

set_option maxHeartbeats 4000000 in
theorem linksCheck_cost (c : Circuit) (pop : Public) (program : ByteArray) :
    countC.P (HmLinks.checkWith (Hashes.counted h) c pop program) 1 := by
  apply countC.mono
  · unfold HmLinks.checkWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply decode_cost h)
      | (with_reducible apply throw0)
      | (with_reducible apply forIn_range0)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply decode_cost h)
        | (with_reducible apply throw0)
        | (with_reducible apply forIn_range0)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero]); omega

set_option maxHeartbeats 4000000 in
theorem loadPublic_cost (tags : Tags) (c : Circuit) (file : ByteArray) :
    countC.P (HmRow.loadPublicWith (Hashes.counted h) tags c file) 1 := by
  apply countC.mono
  · unfold HmRow.loadPublicWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply publicDigest_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply publicDigest_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem inLoadPublic_cost (tags : Tags) (c : Circuit) (pi : HmIn.Ports) (file : ByteArray) :
    countC.P (HmIn.loadPublicWith (Hashes.counted h) tags c pi file) 1 := by
  apply countC.mono
  · unfold HmIn.loadPublicWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply loadPublic_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply loadPublic_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

set_option maxHeartbeats 4000000 in
theorem drawn_cost (c : Circuit) (pub : Public) (draw : Json) :
    countC.P (HmRow.drawnWith (Hashes.counted h) c pub draw) 1 := by
  obtain ⟨header, n, digests, outputs, sha, population, tables⟩ := pub
  unfold HmRow.drawnWith
  cases tables
  all_goals
    dsimp only
    apply countC.mono
    · repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply publicDigest_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
      iterate 40
        all_goals try (dsimp +zetaDelta -zeta only)
        all_goals repeat (first
          | extract_lets
          | (with_reducible apply throwBind)
          | (with_reducible apply Cost.bind)
          | (with_reducible apply bindAny)
          | (with_reducible apply publicDigest_cost h)
          | (with_reducible intro _)
          | (with_reducible apply Cost.pure)
          | (with_reducible apply pureAny)
          | (with_reducible apply throw0)
          | (with_reducible apply Cost.throw)
          | (with_reducible apply Cost.lift)
          | (with_reducible apply liftAny)
          | (with_reducible apply Cost.forIn_array)
          | (with_reducible apply Cost.forIn_array0)
          | (with_reducible apply forIn_array0Any)
          | (with_reducible apply Cost.forIn_list)
          | (with_reducible apply Cost.forIn_list0)
          | (with_reducible apply forIn_list0Any)
          | (with_reducible apply forIn_range0)
          | (with_reducible apply forIn_range0Any)
          | (with_reducible apply Cost.mapM_array)
          | (with_reducible apply mapM_array0Any)
          | (with_reducible apply Cost.ite)
          | (with_reducible apply iteAny)
          | (with_reducible apply diteMax)
          | (with_reducible apply diteAny)
          | split)
      all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
    · (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero]); omega

theorem rowParse_cost (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) :
    countC.P (HmRow.parseWith (Hashes.counted h) tags bytes tables) 2 := by
  apply countC.mono
  · unfold HmRow.parseWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply sha_cost h)
      | (with_reducible apply classOf_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply sha_cost h)
        | (with_reducible apply classOf_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega


/-! ## The public file's frame-v3 roots: a cost that depends on the rows' counts -/

/-- On success, the value satisfies `Q`. -/
def okC {α : Type} (x : CountV α) (Q : α → Prop) : Prop := ∀ k a k', x.go k = (.ok a, k') → Q a

theorem okC_pure {α : Type} {Q : α → Prop} {a : α} (h : Q a) : okC (pure a : CountV α) Q := by
  intro k b k' hb
  cases hb
  exact h

theorem okC_bind {α β : Type} {x : CountV α} {f : α → CountV β} {R : α → Prop} {Q : β → Prop} (hx : okC x R)
    (hf : ∀ a, R a → okC (f a) Q) : okC (x >>= f) Q := by
  intro k b k'' hb
  rw [go_bind] at hb
  rcases hr : x.go k with ⟨r, k'⟩
  rw [hr] at hb
  cases r with
  | error e => cases hb
  | ok a => exact hf a (hx k a k' hr) k' b k'' hb

theorem bindQ {α β : Type} {x : CountV α} {f : α → CountV β} {Q : α → Prop} {n m N : Nat} (hx : countC.P x n)
    (hq : okC x Q) (hf : ∀ a, Q a → countC.P (f a) m) (hN : n + m ≤ N) : countC.P (x >>= f) N := by
  intro k
  rw [go_bind]
  have h1 := hx k
  rcases hr : x.go k with ⟨r, k'⟩
  rw [hr] at h1
  cases r with
  | error e => dsimp only at h1 ⊢; omega
  | ok a => have := hf a (hq k a k' hr) k'; dsimp only at h1 ⊢; omega

theorem mapM_list_cost {α β : Type} (l : List α) {f : α → CountV β} {c : Nat} (hf : ∀ a, countC.P (f a) c) :
    countC.P (l.mapM f) (l.length * c) := by
  induction l with
  | nil => exact countC.mono (countC.pure _) (Nat.zero_le _)
  | cons a l ih =>
    rw [List.mapM_cons]
    refine countC.mono (countC.bind (hf a) fun b => countC.bind ih fun bs => countC.pure _) ?_
    rw [List.length_cons, Nat.succ_mul]
    omega

theorem mapM_array_cost {α β : Type} (xs : Array α) {f : α → CountV β} {c : Nat} (hf : ∀ a, countC.P (f a) c) :
    countC.P (xs.mapM f) (xs.size * c) := by
  rw [Array.mapM_eq_mapM_toList, map_eq_pure_bind]
  refine countC.mono (countC.bind (mapM_list_cost xs.toList hf) fun _ => countC.pure _) ?_
  rw [Array.length_toList]
  omega

theorem mapM_list_ok {α β : Type} (l : List α) (f : α → CountV β) : okC (l.mapM f) (·.length = l.length) := by
  induction l with
  | nil => exact okC_pure rfl
  | cons a l ih =>
    rw [List.mapM_cons]
    exact okC_bind (R := fun _ => True) (fun _ _ _ _ => trivial) fun b _ =>
      okC_bind ih fun bs hbs => okC_pure (by rw [List.length_cons, List.length_cons, hbs])

theorem mapM_array_ok {α β : Type} (xs : Array α) (f : α → CountV β) : okC (xs.mapM f) (·.size = xs.size) := by
  rw [Array.mapM_eq_mapM_toList, map_eq_pure_bind]
  exact okC_bind (mapM_list_ok xs.toList f) fun bs hbs =>
    okC_pure (by rw [List.size_toArray, hbs, Array.length_toList])

/-- A loop whose steps each cost at most `c` while an invariant `I` of its state holds, and keep it. -/
theorem forIn_list_inv {α β : Type} (l : List α) (b : β) {f : α → β → CountV (ForInStep β)} (I : β → Prop) {c : Nat}
    (hb : I b) (hc : ∀ a s, I s → countC.P (f a s) c)
    (hI : ∀ a s, I s → okC (f a s) (fun r => match r with | .done s' => I s' | .yield s' => I s')) :
    countC.P (forIn l b f) (l.length * c) := by
  induction l generalizing b with
  | nil => rw [List.forIn_nil]; exact countC.mono (countC.pure _) (Nat.zero_le _)
  | cons a l ih =>
    rw [List.forIn_cons]
    refine bindQ (hc a b hb) (hI a b hb) (fun r hr => ?_) (m := l.length * c) ?_
    · cases r with
      | done s' => exact countC.mono (countC.pure _) (Nat.zero_le _)
      | yield s' => exact ih s' hr
    · rw [List.length_cons, Nat.succ_mul]
      omega

theorem forIn_upto_inv {β : Type} (L : Nat) (b : β) {f : Nat → β → CountV (ForInStep β)} (I : β → Prop) {c : Nat}
    (hb : I b) (hc : ∀ a s, I s → countC.P (f a s) c)
    (hI : ∀ a s, I s → okC (f a s) (fun r => match r with | .done s' => I s' | .yield s' => I s')) :
    countC.P (forIn [0:L] b f) (L * c) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have := forIn_list_inv (List.range' [0:L].start [0:L].size [0:L].step) b I hb (fun a s => hc a s) fun a s => hI a s
  rw [List.length_range'] at this
  simpa [Std.Legacy.Range.size] using this

theorem log2ceil_le {n K : Nat} (h : n ≤ 2 ^ K) : log2ceil n ≤ K := by
  unfold log2ceil
  split
  · exact Nat.zero_le _
  · have hlt : (n - 1).log2 < K := (Nat.log2_lt (by omega)).2 (by omega)
    omega

theorem nextPow2_le {n K : Nat} (h : n ≤ 2 ^ K) : nextPow2 n ≤ 2 ^ K :=
  Nat.pow_le_pow_right (by decide) (log2ceil_le h)

theorem hash_cost (tag : String) (parts : List ByteArray) :
    countC.P (FV3.hashWith (Hashes.counted h) tag parts) 1 := sha_cost h _

/-- `FV3.rootWith`'s calls: a pad leaf to the next power of two, and every node on the levels above. -/
theorem root_cost (dom : ByteArray) (leaves : Array ByteArray) :
    countC.P (FV3.rootWith (Hashes.counted h) dom leaves)
      ((nextPow2 leaves.size - leaves.size)
        + log2ceil (nextPow2 leaves.size) * (leaves.size + (nextPow2 leaves.size - leaves.size))) := by
  unfold FV3.rootWith
  dsimp only
  refine bindQ (mapM_array_cost _ (c := 1) fun _ => hash_cost h _ _) (mapM_array_ok _ _) (fun pads hp => ?_)
    (m := log2ceil (nextPow2 leaves.size) * (leaves.size + (nextPow2 leaves.size - leaves.size)) + 0) ?_
  · refine countC.bind (forIn_upto_inv _ _
      (fun lv : Array ByteArray => lv.size ≤ leaves.size + (nextPow2 leaves.size - leaves.size)) ?_ ?_ ?_)
      fun _ => countC.pure _
    · show (leaves ++ pads).size ≤ _
      rw [Array.size_append, hp, Array.size_range]
      exact Nat.le_refl _
    · intro i s hs
      refine countC.mono (countC.bind (mapM_array_cost _ (c := 1) fun _ => hash_cost h _ _)
        fun _ => countC.pure _) ?_
      rw [Array.size_range]
      omega
    · intro i s hs
      exact okC_bind (mapM_array_ok _ _) fun lv hlv => okC_pure (by show lv.size ≤ _; rw [hlv, Array.size_range]; omega)
  · rw [Array.size_range]
    omega

/-- The bound on one row's leaves and root, at a row count of at most `2 ^ K`. -/
def perRow (K : Nat) : Nat := (K + 3) * 2 ^ K

/-- One row's leaves (a tree leaf and a frame leaf each) and the root over them, then anything at no calls. -/
theorem leavesRoot_cost {β : Type} {K count : Nat} (hc : count ≤ 2 ^ K) (dom : ByteArray)
    {g : Nat → CountV ByteArray} (hg : ∀ r, countC.P (g r) 2) {k : ByteArray → CountV β} (hk : ∀ x, countC.P (k x) 0) :
    countC.P ((Array.range count).mapM g >>= fun leaves => FV3.rootWith (Hashes.counted h) dom leaves >>= k)
      (perRow K) := by
  refine bindQ (mapM_array_cost _ hg) (mapM_array_ok _ _) (fun leaves hl => ?_)
    (m := (nextPow2 count - count) + log2ceil (nextPow2 count) * (count + (nextPow2 count - count)) + 0) ?_
  · have := countC.bind (root_cost h dom leaves) hk
    rwa [hl, Array.size_range] at this
  · have hw := nextPow2_le hc
    have hl : log2ceil (nextPow2 count) ≤ K := log2ceil_le hw
    have hm : log2ceil (nextPow2 count) * (count + (nextPow2 count - count)) ≤ K * 2 ^ K :=
      Nat.mul_le_mul hl (by omega)
    rw [Array.size_range, perRow, Nat.add_mul]
    omega

theorem leafPair_cost (bc dom : ByteArray) (r : Nat) (schema : String) :
    countC.P (Hm96.Default512.treeLeafWith (Hashes.counted h) bc >>= fun x => FV3.leafWith (Hashes.counted h) dom r schema x)
      2 :=
  countC.bind (sha_cost h _) fun _ => sha_cost h _

theorem getElem!_le {xs : Array Nat} {B : Nat} (h : ∀ r ∈ xs, r ≤ B) (p : Nat) : xs[p]! ≤ B := by
  by_cases hp : p < xs.size
  · rw [getElem!_pos xs p hp]; exact h _ (Array.getElem_mem hp)
  · rw [getElem!_neg xs p hp]; exact Nat.zero_le _

/-- **`HmOut.checkPublicWith`'s calls**: each port's rows hashed into leaves and a frame-v3 root, at most `2 ^ K` rows a
port. -/
theorem checkPublic_cost (c : Circuit) (pub : Public) (K : Nat)
    (hT : ∀ t, pub.tables = some t → ∀ r ∈ t.counts, r ≤ 2 ^ K)
    (hN : (pub.population.getD (pub.n, pub.digests, pub.outputs)).1 ≤ 2 ^ K) :
    countC.P (HmOut.checkPublicWith (Hashes.counted h) c pub) (c.ports.size * perRow K) := by
  obtain ⟨header, n0, digests, outputs, sha, population, tables⟩ := pub
  have hpop : ∃ n bcs o, population.getD (n0, digests, outputs) = (n, bcs, o) ∧ n ≤ 2 ^ K :=
    ⟨_, _, _, rfl, hN⟩
  obtain ⟨n, bcs, o, hp, hn⟩ := hpop
  unfold HmOut.checkPublicWith
  dsimp only
  rw [hp]
  cases tables with
  | none =>
    cases hd : header.getObjVal? "unit_draw"
    all_goals
      simp only [hd]
      apply countC.mono
      · repeat (first
          | extract_lets
          | (with_reducible apply leavesRoot_cost h)
          | (with_reducible apply leafPair_cost h)
          | (with_reducible apply throwBind)
          | (with_reducible apply Cost.bind)
          | (with_reducible apply bindAny)
          | (with_reducible apply Cost.forIn_upto)
          | (with_reducible intro _)
          | (with_reducible apply Cost.pure)
          | (with_reducible apply pureAny)
          | (with_reducible apply throw0)
          | (with_reducible apply Cost.throw)
          | (with_reducible apply Cost.lift)
          | (with_reducible apply liftAny)
          | (with_reducible apply Cost.forIn_array)
          | (with_reducible apply Cost.forIn_array0)
          | (with_reducible apply forIn_array0Any)
          | (with_reducible apply Cost.forIn_list)
          | (with_reducible apply Cost.forIn_list0)
          | (with_reducible apply forIn_list0Any)
          | (with_reducible apply forIn_range0)
          | (with_reducible apply forIn_range0Any)
          | (with_reducible apply Cost.mapM_array)
          | (with_reducible apply mapM_array0Any)
          | (with_reducible apply Cost.ite)
          | (with_reducible apply iteAny)
          | (with_reducible apply diteMax)
          | (with_reducible apply diteAny)
          | split)
        iterate 40
          all_goals try (dsimp +zetaDelta -zeta only)
          all_goals repeat (first
            | extract_lets
            | (with_reducible apply leavesRoot_cost h)
            | (with_reducible apply leafPair_cost h)
            | (with_reducible apply throwBind)
            | (with_reducible apply Cost.bind)
            | (with_reducible apply bindAny)
            | (with_reducible apply Cost.forIn_upto)
            | (with_reducible intro _)
            | (with_reducible apply Cost.pure)
            | (with_reducible apply pureAny)
            | (with_reducible apply throw0)
            | (with_reducible apply Cost.throw)
            | (with_reducible apply Cost.lift)
            | (with_reducible apply liftAny)
            | (with_reducible apply Cost.forIn_array)
            | (with_reducible apply Cost.forIn_array0)
            | (with_reducible apply forIn_array0Any)
            | (with_reducible apply Cost.forIn_list)
            | (with_reducible apply Cost.forIn_list0)
            | (with_reducible apply forIn_list0Any)
            | (with_reducible apply forIn_range0)
            | (with_reducible apply forIn_range0Any)
            | (with_reducible apply Cost.mapM_array)
            | (with_reducible apply mapM_array0Any)
            | (with_reducible apply Cost.ite)
            | (with_reducible apply iteAny)
            | (with_reducible apply diteMax)
            | (with_reducible apply diteAny)
            | split)
        all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *)
        all_goals first | omega | exact Nat.le_refl _
      · (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero]); first | omega | exact Nat.le_refl _
  | some t =>
    have hT' := hT t rfl
    cases hd : header.getObjVal? "unit_draw"
    all_goals
      simp only [hd]
      apply countC.mono
      · repeat (first
          | extract_lets
          | (with_reducible apply leavesRoot_cost h)
          | (with_reducible apply leafPair_cost h)
          | (with_reducible apply throwBind)
          | (with_reducible apply Cost.bind)
          | (with_reducible apply bindAny)
          | (with_reducible apply Cost.forIn_upto)
          | (with_reducible intro _)
          | (with_reducible apply Cost.pure)
          | (with_reducible apply pureAny)
          | (with_reducible apply throw0)
          | (with_reducible apply Cost.throw)
          | (with_reducible apply Cost.lift)
          | (with_reducible apply liftAny)
          | (with_reducible apply Cost.forIn_array)
          | (with_reducible apply Cost.forIn_array0)
          | (with_reducible apply forIn_array0Any)
          | (with_reducible apply Cost.forIn_list)
          | (with_reducible apply Cost.forIn_list0)
          | (with_reducible apply forIn_list0Any)
          | (with_reducible apply forIn_range0)
          | (with_reducible apply forIn_range0Any)
          | (with_reducible apply Cost.mapM_array)
          | (with_reducible apply mapM_array0Any)
          | (with_reducible apply Cost.ite)
          | (with_reducible apply iteAny)
          | (with_reducible apply diteMax)
          | (with_reducible apply diteAny)
          | split)
        iterate 40
          all_goals try (dsimp +zetaDelta -zeta only)
          all_goals repeat (first
            | extract_lets
            | (with_reducible apply leavesRoot_cost h)
            | (with_reducible apply leafPair_cost h)
            | (with_reducible apply throwBind)
            | (with_reducible apply Cost.bind)
            | (with_reducible apply bindAny)
            | (with_reducible apply Cost.forIn_upto)
            | (with_reducible intro _)
            | (with_reducible apply Cost.pure)
            | (with_reducible apply pureAny)
            | (with_reducible apply throw0)
            | (with_reducible apply Cost.throw)
            | (with_reducible apply Cost.lift)
            | (with_reducible apply liftAny)
            | (with_reducible apply Cost.forIn_array)
            | (with_reducible apply Cost.forIn_array0)
            | (with_reducible apply forIn_array0Any)
            | (with_reducible apply Cost.forIn_list)
            | (with_reducible apply Cost.forIn_list0)
            | (with_reducible apply forIn_list0Any)
            | (with_reducible apply forIn_range0)
            | (with_reducible apply forIn_range0Any)
            | (with_reducible apply Cost.mapM_array)
            | (with_reducible apply mapM_array0Any)
            | (with_reducible apply Cost.ite)
            | (with_reducible apply iteAny)
            | (with_reducible apply diteMax)
            | (with_reducible apply diteAny)
            | split)
        all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *)
        all_goals first | omega | exact Nat.le_refl _ | exact getElem!_le hT' _
      · (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero]); first | omega | exact Nat.le_refl _

end RecDeploy.SetupCount
