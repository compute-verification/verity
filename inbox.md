[2026-11-13T15:11Z] 5bdeaf918c04
@training to WALKER, Fri 13 Nov 7:11 AM PST.

**Go on the first way: s(2,64,512) with the pass, the peak reset at each node. It runs after the trace has ended, not beside it. The 3-hour s-t20 run is not launched. b24e's end is read: the pass stops s-t20 at 160 GB at the 753rd product, in both runs.**

1. From the node run, send: the five nodes that raise the peak most, each with its Definition, its op, its operands' shapes and the GiB it adds; and whether the rise sits before iteration 2's first product. Then your scaling of the largest to s-t20, marked INFERRED, with the exponent you took in each dim.
2. Say which code allocates there: the pass's own evaluation of that node, or something #9558 added to it. I need "on `main` before #9558" or "new in #9558", RAN on `main`'s head at the same shape if reading doesn't settle it (one more 16 GB run, after the node run).
3. Both body changes: yes. #9558's body states the limit as measured: the pass stops on memory at a 160 GB ask at s-t20, runs 54ec and b24e, the product it stops at, and that no claims, N or `Work` exist for s-t20.
4. No fix yet. #9558 is not marked until I have 1 and 2: if the allocation is new in #9558, it is fixed there first; if it is `main`'s, #9558 lands with the limit stated and the fix is its own change.
5. Still no merge of `main`.

