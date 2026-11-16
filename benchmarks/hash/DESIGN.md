# benchmarks/hash: design

## Interface

What code outside `benchmarks/hash` loads. Nothing else here is an interface: any other file may change or go without a
word to another directory. The vLLM integration loads nothing from here.

| File | What is loaded | Loaded by |
|---|---|---|
| `dither/pin.sh` | run as a script: the int8 pin in a node's model cache | `benchmarks/harness/setup/int8_pin.sh` |

Tests of `benchmarks/pouw` import `dither/dither_one.py`, `dither_a.py`, `j10r_units.py` and `forms.py` by name, and its
suite declares them and what they import as inputs. A test's import fixes a file's path, nothing more.
