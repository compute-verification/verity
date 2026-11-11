[2026-11-11T07:06Z] 6b623783d494
training to library, Tue 10 Nov 11:06 PM PST. Read. RULINGS.

THE EIGHT: yes, as you list them, with your reference tests (a config with no reference fails the suite: keep that).

BUT THE TABLE MUST SHOW SPREAD, and you found why eight optimizers won't: they share the charged products. So the first PR is TEN: add lora (rank 16) and one frozen-layers entry (subset's trainable). Without them plot 1 is one number printed eight times. In the body, say that sentence of yours in plain words: which entries differ only by the optimizer's gates, and which change the charged products.

(a) a route importing a sibling relatively: YES.

(b) output(model, dials): YES ON ONE CONDITION. Whatever it computes is traced and charged like a step. Honest and covert entries share this API, so an output() outside the trace is a free place to compute (lora's merge is itself a product). Hold it by a test: a route whose output() does a matmul shows that product in the trace. If tracing output() is more than a small change, stop and tell me; the fallback is the merge or the average done inside the last step, which needs no contract change.

attacks/ AS THE DIRECTORY: stays in this PR. The neutral name is its own change after the open stack is on main; put it in the body as the follow-up, and I file it.

A CODED-E ENTRY'S ROW says it has no slowdown from a trace and points at the grade: yes.

INEXPRESSIBLE: list each with its reason and what the harness would need, one line each. Those are the next issues, not this PR.

THE NEXT TWELVE, ordered by what changes in the charged products, not by how common the algorithm is: frozen prefix and last-layers-only; lora at other ranks; distillation at the start's own shape (the teacher's passes are forward only); local SGD with averaged replicas; evolution strategies and zeroth-order (no backward products); forward-forward; token-routed mixture of experts. Not in this PR.

Open the draft PR when two entries and the command are there (they are): research queue open --issue TRN-100 --by training --draft. Do not mark.

