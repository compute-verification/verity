import Proofs.Accounting.Memory.Counting
import Proofs.Accounting.Memory.Adds

/-!
# Memory adds, by counting: the guarantees

Each statement of `Security/Models/Accounting/Guarantees/Memory.lean`, proved:

* `Memory/Counting.lean`: the counting core, Markov's inequality over the disjoint fibres;
* `Memory/Adds.lean`: the composition with PoUS's guarantee: each fibre coded into `S` bits, the coded adversary
  answering as the location does, and the union over the challenges; the isolation forms at `H = 0`.
-/

namespace Accounting.SecurityProofs.Memory

theorem Counting : Accounting.Guarantees.Memory.Counting := Accounting.MemoryProofs.counting
theorem Adds : Accounting.Guarantees.Memory.Adds := Accounting.MemoryProofs.adds
theorem AddsOn : Accounting.Guarantees.Memory.AddsOn := Accounting.MemoryProofs.addsOn
theorem AddsP2Ring : Accounting.Guarantees.Memory.AddsP2Ring := Accounting.MemoryProofs.addsP2Ring
theorem AddsIsolated : Accounting.Guarantees.Memory.AddsIsolated := Accounting.MemoryProofs.addsIsolated
theorem AddsOnIsolated : Accounting.Guarantees.Memory.AddsOnIsolated := Accounting.MemoryProofs.addsOnIsolated
theorem AddsP2RingIsolated : Accounting.Guarantees.Memory.AddsP2RingIsolated :=
  Accounting.MemoryProofs.addsP2RingIsolated
theorem CountingAdvice : Accounting.Guarantees.Memory.CountingAdvice := Accounting.MemoryProofs.countingAdvice
theorem AddsAdvice : Accounting.Guarantees.Memory.AddsAdvice := Accounting.MemoryProofs.addsAdvice
theorem AddsAdviceOn : Accounting.Guarantees.Memory.AddsAdviceOn := Accounting.MemoryProofs.addsAdviceOn
theorem AddsAdviceP2Ring : Accounting.Guarantees.Memory.AddsAdviceP2Ring := Accounting.MemoryProofs.addsAdviceP2Ring

end Accounting.SecurityProofs.Memory
