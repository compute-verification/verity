import Proofs.Flock.Claims.DeployEfficient
import Definitions.Claims.Sampling
import Definitions.Claims.Recursion

/-!
# What the claims ask of a record holds at the deployed ones

The lock's `claims` section names, for each claim, theorems `meets`, one per deployed record, that state the claim's
checkable antecedents there, so that the claim is not true at the deployment only because nothing meets them. The
deployed records are Flock's, `deployPlain` and `deployZk` (`DeployCode`). The sampling and recursion compilers' claims
read a Flock record, the sampling record's `flock` and the recursion's outer record `DO` (over Flock, its inner record
too), and their antecedents are of that record:

* `Flock.decoder_meets_plain`, `_zk`: the record's own decoder meets the list-decoding spec at the record's code,
  length and radius (`decode_plain`, `listWithin_plain`). That is the antecedent of Flock's `KS` and `Bits` (every
  decoder that meets the spec) and a conjunct of its `Efficient`.
* `Sampling.decoder_meets_plain`, `_zk`: the same at a sampling record whose `flock` is the deployed record, the
  antecedent of the sampling `KS`, `KSCount` and `Bits` and a conjunct of its `Efficient`.
* `Recursion.decoder_meets_plain`, `_zk`: the deployed record's decoder is one the recursion's claims quantify over
  (`decOK`).
* `Flock.masked_meets_zk`: `deployZk` is the masked verifier, the first conjunct of Flock's `ZK` and of the recursion's
  (`DO.zk`); `Sampling.masked_meets_zk`: so is the `flock` of a sampling record over it, the first conjunct of the
  sampling `ZK`.
-/

open FlockDeploy

local notation "𝔽" => FlockLevel3.GF128

namespace Claims

/-- **The plain record's decoder meets the list-decoding spec** at the record's code, length and radius. -/
theorem Flock.decoder_meets_plain (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    Claims.Flock.ListWithin (deployPlain rel).len (deployPlain rel).enc (deployPlain rel).θ
      (deployPlain rel).decode := by
  rw [decode_plain]
  exact listWithin_plain rel

/-- **The masked record's decoder meets the list-decoding spec** at the record's code, length and radius. -/
theorem Flock.decoder_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    Claims.Flock.ListWithin (deployZk rel).len (deployZk rel).enc (deployZk rel).θ (deployZk rel).decode := by
  rw [decode_zk]
  exact listWithin_zk rel

/-- **The masked record is the masked verifier.** -/
theorem Flock.masked_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation) : (deployZk rel).zk = true :=
  rfl

/-- **At a sampling record over the plain record**, its Flock record's decoder meets the list-decoding spec. -/
theorem Sampling.decoder_meets_plain (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (D : Claims.Sampling.Deploy Coin (List 𝔽) Claims.Flock.PlainTrace) (hD : D.flock = deployPlain rel) :
    Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ D.flock.decode := by
  rw [hD]
  exact Claims.Flock.decoder_meets_plain rel

/-- **At a sampling record over the masked record**, its Flock record's decoder meets the list-decoding spec. -/
theorem Sampling.decoder_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (D : Claims.Sampling.Deploy Coin (List 𝔽) Claims.Flock.ZkTrace) (hD : D.flock = deployZk rel) :
    Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ D.flock.decode := by
  rw [hD]
  exact Claims.Flock.decoder_meets_zk rel

/-- **At a sampling record over the masked record**, its Flock record is the masked verifier. -/
theorem Sampling.masked_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (D : Claims.Sampling.Deploy Coin (List 𝔽) Claims.Flock.ZkTrace) (hD : D.flock = deployZk rel) :
    D.flock.zk = true := by
  rw [hD]
  rfl

/-- **The plain record's decoder is one the recursion's claims quantify over.** -/
theorem Recursion.decoder_meets_plain (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    Claims.Recursion.decOK (deployPlain rel) (deployPlain rel).decode :=
  Claims.Flock.decoder_meets_plain rel

/-- **The masked record's decoder is one the recursion's claims quantify over.** -/
theorem Recursion.decoder_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    Claims.Recursion.decOK (deployZk rel) (deployZk rel).decode :=
  Claims.Flock.decoder_meets_zk rel

end Claims
