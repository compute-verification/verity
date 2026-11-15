module

public import CoreProtocol.Cli
public import CoreRemote

/-! # `remote-v1`: remote's composite of record on the core protocol command line

`Core.Remote.V1.protocol` (`verity/remote/v1`: the inference-only policy, Pearl+'s run verdict and SPOT)
behind `Core.BlobProtocol.main`. `id` prints the protocol id; `check PARAMS COMMITMENT EVIDENCE [BLOB ...]` reads the
files and prints the answer's canonical JSON line; `serve` reads one case a line on standard input and prints one answer
line for each. `benchmarks/one_stage/composite.py` (`remote-v1`) and the composition's
`tests/test_composition_difftest.py` run it. -/

@[expose] public section

def main (args : List String) : IO UInt32 := Core.Remote.V1.protocol.main args
