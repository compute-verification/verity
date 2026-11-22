# commitments

The commitment fixes, before any coin, every value a circuit's blocks read or write, and hides it: each row is an
hm96-sha512 hiding leaf under a fresh salt, in a frame-v3-sha512 tree whose root the docket registers. Beside the
scheme it holds what an integration hands Verity each period, the Boolean gates a proof block runs to read a committed
row, and the same commitment on a GPU.

The guarantee: `Hash.Guarantees.RowBinding` (`security/Definitions/Hash/Guarantees.lean`).

Not yet: a supplied input's domain source kind in `Supplied`, with a table's rows in the docket's `supplied`; the verifier's registered file built from the auditor's pins; the serving committers (`native_rows.cu`, `step_rows.py`) under `cuda/`; one trace commitment at serve time; the adapter's commitment code deleted; supplied rows' public salts from the one expander (`registered_value.supplied_salts` derives its own).

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| hiding leaves | `hm96-sha512/v1`: a row digest under a 192-byte salt | `hm96/` (`PROTOCOL.md`) |
| trees | `frame-v3-sha512` indexed trees over verifier-derived domains | `frame_v3/` (`PROTOCOL.md`), `merkle.py`, `indexed.py` |
| rows | the row formats, row tables and registered values | `rowleaf.py`, `row_table.py`, `registered_value.py` |
| the window's record | what an integration hands Verity a period: the circuit, a compiler and its advice, the values, a sample's openings | `periods/` (`interface.py`) |
| the gates | the commitment's reads as Boolean Definitions a proof block runs | `gates/` |
| the device code | the row commitment and SHA-512 a CUDA kernel includes; the salt stream | `cuda/` |
| the salt expander | the one challenge derivation, and the keyed SHA-512 stream a derived hiding salt reads | `randomness/` |
| the domains | every versioned string hashed under, in Lean, which Python reads | `identity/`, `domains.py` |
| the Lean | frame-v3, hm96, the sample, the session coins | `lean/` |
| signatures and channels | LMS over SHA-512, a device's signed period roots, the SHA-512 AEAD | `crypto/`, `lms/`, `attest/` |

## What varies

| implementation | CPU (Python) | sm_90 | sm_120 |
| --- | --- | --- | --- |
| `hm96-sha512/row/v1`: Flock's ports, registered values | served | never written | never written |
| `hm96-sha512/row/v2`: serving's step rows, supplied rows | served | served | exists |
| `hm96-sha512/row-seg/v1`: row tables, PoUW's operand rows | served | served | exists |

The hash is SHA-512 with its whole output; SHA-256 frames and BLAKE3 stay only where a recorded format names them, and
TurboSHAKE128 only as Pearl-C's ticket hash (`crypto/turboshake.py`).

A new implementation supplies: a committer for its device that writes the CPU reference's leaves and roots bit for bit,
from `cuda/row_commit.cuh` (the one copy a kernel includes), held to `hm96/vectors_sha512.json` and
`frame_v3/vectors_sha512.json`.
Fixed for every implementation: SHA-512, the hm96-sha512 leaf, the frame-v3-sha512 framing, the row formats, the domain
table, and salts fresh from the operating system or from `randomness`'s stream under a key fresh for one step.

## Interfaces

- From the workload: each unit's values as computed, and the circuit's layout that says which rows they fill
  (`periods/interface.py`, `Window.record`).
- From `run`: the window's domain (`window.py`), a fresh salt key for each step, and registration before any coin.
- It gives: the roots the docket registers, the openings a sampled member's proof reads (`periods/flock_inputs.py`), and
  the reads of a committed row at a hidden position (`gates/frame_v3.py`).

## Try it

~~~sh
uv run pytest -q tests/verity/commitments/test_hm96.py tests/verity/commitments/test_frame_v3.py tests/verity/commitments/test_row_table.py
~~~

It needs no GPU: `119 passed`, the hm96 leaves and frame-v3 trees held to their vectors, and the row tables to the reference.
