module

public import CircuitRequirement.Difftest

/-! `requirement-difftest`: `CircuitRequirement.Difftest.table`, the one function `requirement`, behind
`Difftest.serve`. It reads one request a line on standard input, `{"fn": "requirement", "input": INSTANCE}`, until the
input ends, and prints one line for each: `{"output": JSON}`, what the package's functions compute on the instance;
`{"reject": REASON}`, where `requirement` refuses it; or `{"error": MESSAGE}`, a fault of the line protocol (a line that
is no request, another `fn`, an output with no canonical form). `tests/test_requirement_difftest.py` drives it against
`verity.definitions.circuits.requirement`, and `tests/test_census_and_memory_map.py` against that module's
`from_census` and memory-map/v1's builder. -/

@[expose] public section

def main : IO Unit := Difftest.serve CircuitRequirement.Difftest.table
