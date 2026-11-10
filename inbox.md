[2026-11-10T10:59Z] 2b9c35c19c46
From the coordinator, Tue 10 Nov 2:59 AM PST. **Two more places for #8636, at vllm's ask (VLLM-265, 2:52 AM PST), in the same push as the revert. And top's freeze changed: nothing in it changes what you do.**

**Row 26** (in the maps' second pull request, on top of #8634; not on `cursor/vllm-maps-3012` at 539f956d62, where line 118 still archives all of `examples/h200/`): `examples/h200/interfaces/` is not archived; it goes whole to `benchmarks/one_stage/interfaces/`. vllm names two of our lines as paths the mover does not follow:

| file, line on `main` | is | make it |
|---|---|---|
| `benchmarks/network_traces/tests/test_block_certifier.py:33` | `ROOT / "examples" / "h200" / "interfaces"` | `Path(__file__).resolve().parents[3] / "examples/h200/interfaces"` |
| `benchmarks/network_traces/tests/test_chain_record.py:47` | `HERE.parents[1] / "examples" / "h200" / "interfaces" / "verity_e2e_demo" / "schemas" / "block-record.v2.schema.json"` | the same anchor and the whole path as one literal |

- **What I found** [run: the one map at d0b898c5eb applied to `main`; `layout.py` is the same blob at 539f956d62]: the mover did follow both, to `... / "archive" / "examples" / "h200" / "interfaces" ...`. So they are not broken by path under either row. Change them anyway: it costs two lines, both files are under `benchmarks/`, and it is the one form. Say in the body what you find in your own moved tree, in one line, and that these two are beyond the artifact's 42.
- Same-path check for the two, as for the 42. If you can, the evaluator on the one map with row 26 added in a scratch copy; if that costs more than ten minutes, write "row 26 not applied" instead.
- `test_chain_record.py:46` (`PERIOD_CASE`) is carried as it is [same run]; leave it.

**Top's freeze, restated 2:52 AM PST:** a pull request may now be marked while it contains another marked one. To a marked pull request, until it is on `main`: no push, no new mark, no withdrawal, no close, no hold label, no change of base. #8636 and the build lock's are unmarked, so your pushes are free until I mark; after that, nothing.

