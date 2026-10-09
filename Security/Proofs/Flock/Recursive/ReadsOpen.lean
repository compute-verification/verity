import Proofs.Flock.Recursive.Property
import Proofs.Flock.Recursive.Reads

/-!
# Recursion: the read commit strings open the registration's root

`zk_reads_open` is the recursive audit's hypothesis `_hop` (`Flock.Guarantees.RecursiveAudit`) at the rows, commit
strings and openings read off the outcome (`readsZ`, `rowZ`, `opZ`, `Proofs.Flock.Recursive.Reads`): at an outcome
whose sessions all verify, each registered read's commit string opens the registration's root `rt` at its row, as V*
checks a registered read (`Flock.Guarantees.regBcCommit`). The root is an argument, an instance's registered
commitment. The definitions are in their own module so that the property can read them; this one reads the property's
`VerifiesV` and `regBcCommit`.
-/

namespace FlockSoundness.Discharge.Recursion

/-- **At an outcome whose sessions all verify, each read commit string opens `rt` at its row**: `_hop` at `readsZ`,
`rowZ` and `opZ`. -/
theorem zk_reads_open {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {dg : ByteArray → ByteArray} {N : ℕ}
    {Reg Ω H : Type} (zs : Reg → Ω → H → Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (rt : Flock.Registered.Port) :
    ∀ o : Reg × Ω × H × (Reg₂ × VRec Vs Reg' dg N), Flock.Guarantees.VerifiesV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 →
      ∀ i, readsZ zs rt o i → (Flock.Guarantees.regBcCommit i).Opens rt (rowZ zs rt o i) (opZ zs rt o i) :=
  fun o h i hi => opens_of_accepts zs rt o (And.left h) i hi

end FlockSoundness.Discharge.Recursion
