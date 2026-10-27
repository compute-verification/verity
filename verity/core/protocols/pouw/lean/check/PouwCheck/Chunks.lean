/-!
# An H1 lane's chunks along K (`weights.chunks`)

The port of `weights.chunks` with its `CHUNK_MAX` and `CHUNK_ALIGN`: the chunks an H1 lane at contraction K is proved in
(`h1zy.lane_statements`), and so the chunk rows a table at K commits each of its rows as (`h1zy.chunk_leaves`, row r's
chunk c its leaf n·r + c), which credit's layout reads. `pouw-check-difftest`'s `chunks` (`PouwCheck.Difftest`) holds it
to the Python, refusal for refusal, at every multiple of 32 below 2^17 (`tests/test_difftest.py`).
-/

namespace PouwCheck.Chunks

/-- `weights.CHUNK_MAX`: the largest K a lane is proved at whole, and a chunk's largest. -/
def CHUNK_MAX : Nat := 4096

/-- `weights.CHUNK_ALIGN`: a chunk is whole 512-bit blocks of its rows' and columns' noise streams. -/
def CHUNK_ALIGN : Nat := 512

/-- `weights.chunks`: 1 at 0 < K ≤ `CHUNK_MAX`; past it the least n from ⌈K / CHUNK_MAX⌉ to K / CHUNK_ALIGN with
n · CHUNK_ALIGN dividing K; else why K is refused, the Python's ValueError word for word. -/
def chunks (k : Nat) : Except String Nat :=
  if k = 0 then .error s!"K = {k}: a lane is proved only at a positive K"
  else if k ≤ CHUNK_MAX then .ok 1
  else
    let lo := (k + CHUNK_MAX - 1) / CHUNK_MAX
    match (List.range' lo (k / CHUNK_ALIGN + 1 - lo)).find? (fun n => k % (n * CHUNK_ALIGN) == 0) with
    | some n => .ok n
    | none => .error (s!"K = {k}: no n splits it into chunks of at most {CHUNK_MAX} entries that are whole " ++
        s!"{CHUNK_ALIGN}-bit noise blocks")

end PouwCheck.Chunks
