import Definitions.Pouw.FpG1.Encoding
import Definitions.Pouw.FpG1.Gemm
import Definitions.Pouw.FpG1.H1

/-! FP8-g1 and W4A8-g1: the operand encodings on E4M3's and
E2M1's integer grids with a live coin (`FpG1/Encoding.lean`), the int8 GEMM, the coin's rank-one terms and the
epilogue (`FpG1/Gemm.lean`), and g1 on sm_120's record H1: its grids and a round's operand leaves (`FpG1/H1.lean`).
`Proofs.Pouw.FpG1` proves their arithmetic. `Definitions.Accounting.Guarantees.FpG1`'s headlines read H1's leaves and,
through them, the grids; they are the record's bound at g1's windows, which the record's headlines already cover by
their own quantifier over committed values (`Proofs.Pouw.FpG1`'s module docstring). -/
