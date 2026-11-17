# Circuit properties

The slowdown calculator prices training run inside inference blocks from a few numbers about a circuit: how many
products it counts, how many of them one value feeds, how many feed one output, and how many gates the smallest
product takes. It derives those numbers from a model's matrix shapes. This benchmark measures them exactly, on the
Boolean circuit itself, and sets the two side by side.

Each property is defined for any Boolean circuit, once you say which of its operations are the counted products (a
designation). The circuits measured here are forward layers: no Program here declares a training step's backward or
optimizer circuit.

## Try it

    uv sync --all-packages --extra torch-cpu
    uv run verity bench circuit_properties [--models NAME,...] [--T T] [--out DIR]

It needs only a CPU. For each circuit it prints one JSON line: its gates, its counted products, s_×, a_out and a_in,
and whether each agrees with the slowdown calculator's. A served layer of 10^9 to 10^12 gates measures in seconds. With
`--out DIR` it writes `DIR/properties.json`.

In Python:

```python
from circuit_properties.measure import measure, whole, Blocks
from circuit_properties.references import GEMM_PRODUCTS, layer, configs

m = measure(layer(configs()["qwen25-1b5"][0]), GEMM_PRODUCTS)
m.products, m.s_x, m.a_out, m.a_in        # 46792704, 681, 17920, 1536
```

## The properties

| Property | What it is |
| --- | --- |
| \|𝒥\| | the counted products |
| s_× | the fewest gates in any one product's subcircuit |
| a_out | the most counted products any one gate's value feeds |
| a_in | the fewest counted products that feed one output, over the outputs some product feeds |
| whole in a block | which products a partition computes wholly inside one block, and which it splits |

s_× and whole-in-a-block are defined only on a Boolean lowering: a word circuit's tensor-core step computes 16
products in one gate. The package also reads off a word Program what the min-cut bound takes of a circuit at a
policy's k, which the training benchmark holds beside its own reading of a trace.
