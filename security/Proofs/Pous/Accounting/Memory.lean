module

public import Proofs.Pous.Accounting.Memory.Counting
public import Proofs.Pous.Accounting.Memory.Adds
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Combinatorics.SimpleGraph.Init
import Mathlib.Data.Sym.Sym2.Init

/-! The proofs that memory adds, by counting. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Memory

theorem Counting : ResourceAccounting.Guarantees.Memory.Counting := ResourceAccounting.MemoryProofs.counting
theorem Adds : ResourceAccounting.Guarantees.Memory.Adds := ResourceAccounting.MemoryProofs.adds
theorem AddsOn : ResourceAccounting.Guarantees.Memory.AddsOn := ResourceAccounting.MemoryProofs.addsOn
theorem AddsP2Ring : ResourceAccounting.Guarantees.Memory.AddsP2Ring := ResourceAccounting.MemoryProofs.addsP2Ring
theorem AddsIsolated : ResourceAccounting.Guarantees.Memory.AddsIsolated := ResourceAccounting.MemoryProofs.addsIsolated
theorem AddsOnIsolated : ResourceAccounting.Guarantees.Memory.AddsOnIsolated := ResourceAccounting.MemoryProofs.addsOnIsolated
theorem AddsP2RingIsolated : ResourceAccounting.Guarantees.Memory.AddsP2RingIsolated :=
  ResourceAccounting.MemoryProofs.addsP2RingIsolated
theorem CountingAdvice : ResourceAccounting.Guarantees.Memory.CountingAdvice := ResourceAccounting.MemoryProofs.countingAdvice
theorem AddsAdvice : ResourceAccounting.Guarantees.Memory.AddsAdvice := ResourceAccounting.MemoryProofs.addsAdvice
theorem AddsAdviceOn : ResourceAccounting.Guarantees.Memory.AddsAdviceOn := ResourceAccounting.MemoryProofs.addsAdviceOn
theorem AddsAdviceP2Ring : ResourceAccounting.Guarantees.Memory.AddsAdviceP2Ring := ResourceAccounting.MemoryProofs.addsAdviceP2Ring

end ResourceAccounting.SecurityProofs.Memory
