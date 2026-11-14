# CIR-274: the request window's external ports, read at the join's head

Read at `0b9785f7ba0` (`cursor/circuits-cir274-eight-6f04`). Paths are under
`adapters/vllm/verity_vllm/program/predict/` unless they start with `verity/`.

## The prompt

- **Confirmed: one `Ingress` a rank.** Its port is `tok` of `EmbeddingRead_v1{V,H=192}`, and its units are
  `[at, at + A*M)`.
  - A = H / 192 is the column blocks: 6144 / 192 = 32 at GLM-5.3. M is the prompt's ids.
  - The `Ingress` is recorded in `node_window.py:449-454` (`_ingress`), only where every source of the port is the
    `prompt` parameter.
  - The embedding lowering feeds `tok` at shape (A, M), entry i being `tok[i % M]` (`node_window_embed.py:143-145`).
- **Confirmed: unit `at + b*M + t` reads prompt id t.** Its two u16 words are low first, because a row is its leaves'
  little-endian bytes (`verity/commitments/windows/interface.py:32`, `:378`).
- **The template is one id at every rank.** The read's V is the registered vocabulary, not the shard's rows
  (`node_window_embed.py:52-57`, `:69-76`), so the eight ranks' reads share one template.
- **The joined order is by rank.** `join` lays rank r's units after rank r - 1's in every template
  (`node_window.py:889-897`). Each rank's `Ingress` is shifted by `first[r][reader]` (`node_window.py:960`), in rank order.
  - So rank r's prompt reads are units `[first[r] + at, first[r] + at + A*M)` of the one template.
  - `at` is 0 on every rank: the prefill's read is the first unit of the template.
  - The order is asserted in `tests/program/predict/test_node_window_ranks.py:59-61`.
- **The decode steps' reads are not the prompt.** Each rank's decode step's `tok` is wired from that rank's prefill select's
  combine (`tests/program/predict/test_node_window_select.py:156`), so it is no ingress.
- **No other port reads the prompt.** A port that read it beside another source would be a `Free`, since `_ingress` takes
  only a port fed by the prompt alone.
  - `join` leaves no `Free` at one layer (`tests/program/predict/test_node_window_ranks.py`, `j.free == ()`).
  - Run `r20261114-065825-f1eb` (4 layers, at `f253fd8`, read on its node) showed free `{}` and ingress on 8 ports.
  - The 78-layer build at the head (`87dc`) is not read yet.

## The served ids

- **Confirmed: `GreedyCombine_v2{V,N}`'s `o0`, one unit a select a rank.** The select's stages allocate M units per
  stage (`node_window_select.py:133`), and M is 1 for a request.
- **At O outputs and eight ranks, that is 8 * O units.** The select is lowered once per output, and the window has one
  step a select (`node_window.py:172-174`).
  - Rank r's are units `[first[r], first[r] + O)` of the joined template, in output order: the prefill's select first,
    then each decode's.
  - Rank 0's are units `[0, O)`.
- **The eight copies are equal only by what they compute.** Every rank's combine computes the same function of the same
  committed rows: its own lanes, and the other ranks' lanes wired by `join`. No wire states that the eight copies are equal.
- **One place holds the O ids once a rank, as an input, not an output.** That is the `ids` port of `GreedyStopCheck_v1`
  (`node_window_select.py:150-172`).
  - Row i of that port is wired to output i's combine `o0` on the same rank (`node_window_select.py:171`).
  - There is one such unit a rank, so eight in all.
  - No output port holds the O ids once in the whole circuit.

## The request's fields

- **Confirmed: no port reads them.** `max_tokens`, the prompt's length, the model's length, the stop ids and `finish` are
  statics of `GreedyStopCheck_v1`, which takes them from `StopCheck_v1`'s bindings (`node_window_select.py:163`;
  `verity/definitions/registry/greedy_stop.py:46-48`).
  - Its only ports are `logits`, `r<c>` and `ids`.
  - The window's Program has no parameter beyond `prompt`, `weights` and `peers_*`.
- **So each of them is in the stop template's id.** A class signature that takes `b.request` as an input would need a
  stop Definition that reads those fields as values: vllm's `StopClass_v1` or a successor. The join does not do that.

## What naming them costs in the builder

- **Two functions would state them.**
  - One rank: `NodeWindow.circuit` (`node_window.py:539-563`).
  - The joined circuit: `join` (`node_window.py:961-963`).
  - Each would pass `external=(f"{read}/tok", f"{combine}/o0")` to `IF.Circuit`, one line each.
- **`Circuit.external` cannot say "rank 0's units".** It is template and port names only
  (`verity/commitments/windows/interface.py:1331`, `:1339`, `:1364-1365`).
  - Named on the joined circuit, `EmbeddingRead_v1/tok` covers every rank's reads of the prompt: 8 * A * M units, the
    prompt eight times A over.
  - `GreedyCombine_v2/o0` covers 8 * O units, each id eight times.
  - A rank cannot have a template of its own, since a template is one Definition id.
- **What could narrow it, all in `interface.py` (vllm's file).**
  - A unit range in `external`.
  - Or `window_replay` reading the prompt's unit ranges from `Circuit.ingress`, which already holds
    `(reader, port, at, units)` for each rank (`interface.py:822-833`).
- **Nothing can narrow the ids' output today.**

## What each rank's combine o0 feeds, and the other ranks' ids (at `9ce331c`)

- **Built:** each select's combine port, `GreedyCombine_v2{..}/o0`, is in `Circuit.external`.
  - It is added at `node_window_select.py:149` and emitted by `NodeWindow.circuit`.
  - The joined circuit carries every rank's (`node_window_join.py:214`).
  - The prompt stays under `ingress` alone.
- **What rank r's combine o0 feeds inside the joined window: its own rank only.**
  - Its next decode step's embedding read, `tok` (`node_window_embed.py:144`, `w.take(node, 0)`).
  - Its own `GreedyStopCheck_v1` unit's `ids` row (`node_window_select.py:174`).
- **Nothing across ranks.**
  - `join` keeps each rank's own wires within that rank (`node_window_join.py:166`, `shift(w, x.rank, x.rank)`).
  - The only wires it adds between ranks are to `peers_*` inputs (`:175`, `:198`): all-reduce partials and the logits
    all-gather's parts. No combine output is one of them.
- **Nothing holds ranks 1 to 7's ids equal to rank 0's.** No wire, unit or check compares them.
  - Each rank has its own chunk and combine units, and the same Definitions (`node_window_select.py:142`, `:146`).
  - Their logits rows are wired to the same writer rows: a rank's own lanes to its LM head, the others' lanes through
    `join` to the writer rank's.
  - So each rank's ids are equal to rank 0's exactly when each of those units computes its Definition. That is a
    statement about every unit, which only a sampled proof checks, not a constraint in the circuit.
- **If b.egress is tied to rank 0's selects alone, ranks 1 to 7's ids are interior values.**
  - They are explained, not output.
  - A wrong one changes that rank's next step, whose partials reach rank 0 through the all-reduce wires. So it is a
    wrong block like any other, not a separate output.
