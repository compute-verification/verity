import Definitions.Pouw.FpG1.Encoding
import Definitions.Pouw.FpG1.Gemm

/-! FPX-DESIGN's FP8-g1 and W4A8-g1 (note:20261019T0817Z-report-fpx-design): the operand encodings on E4M3's and
E2M1's integer grids with a live coin (`FpG1/Encoding.lean`), and the int8 GEMM, the coin's rank-one terms and the
epilogue (`FpG1/Gemm.lean`). `Models.Accounting.Guarantees.FpG1` states the record's sm_120 headlines at g1's windows;
`Proofs.Pouw.FpG1` proves the arithmetic. -/
