import InferenceOnlyVerifier.RecordDifftest
import InferenceOnlyVerifier.DescDifftest

-- Scratch, never merged: Lean's record difftest and descriptor difftest on the cases of TOP-441's ask.
#eval show IO Unit from do
  let cwd ← IO.currentDir
  IO.println s!"cwd {cwd}"
  let mut base : System.FilePath := cwd / "scratch_ioread"
  for c in [cwd / "scratch_ioread", cwd / "verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scratch_ioread"] do
    if ← c.pathExists then base := c
  IO.println s!"data {base}"
  let r ← InferenceOnlyVerifier.RecordDifftest.main
    [(base / "record_cases.jsonl").toString, (base / "record_python.jsonl").toString, "5"]
  IO.println s!"record difftest exit {r}"
  let d ← InferenceOnlyVerifier.DescDifftest.main
    [(base / "desc_instances.jsonl").toString, (base / "desc_expanded.jsonl").toString, (base / "desc_python.jsonl").toString, "2"]
  IO.println s!"desc difftest exit {d}"
