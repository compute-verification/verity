[2026-11-11T13:22Z] 316806b3eafc
REPLOT to training, TRN-111, the second as-if (your 2:53 AM PST message: the stub a, a null Delta_grad, the three
QAT rows). A separate count from the tokens. Nothing published.

RESULT: NO PRINTED CELL MOVES in any of the three as-if builds, on the search table (1,497 rows), the overlay (458)
or the page (197), and what binds changes on no row. Only the stored inputs (rho_inputs) read differently.

Run by me: node job r20261111-125121-50ef (cpu-7, rc 0, peak 42 GB of 96 declared): three builds, each on its own
copy of the scorer's state, replot.py at 225e85e26e, te_core 4.13.6 (tree 7de182a661a4), each compared cell by cell
with the published tables (art:4ba75358e532..., art:b760642b7efe...). The same builds on my VM before it gave the
same answer. An earlier node job (r20261111-113521-a436) is void: its three layers shared one cache name.

a. WHICH ROWS READ WHAT.
 - No field of the table says "stub" for a. `estimated` tags a_bits "bound" for both sources: formalism.STUBS' "the
   model's width x 16" and the trace's "the narrowest cut across the forward's affected path". By each ledger row's
   own label (read by me on the 8,875 rows of the 584 runs traced at 4.13.6): all are the traced cut, none the stub.
 - But the traced cut is itself a residual stream at 16 bits a coordinate on 1,116 of the 1,332 search rows that
   have inputs (147 of the page's 166), and narrower on 198 (19). So QAT's finding is about the recorder's price of
   a stream coordinate, not about a stub, and it reaches those 1,116 rows.
 - Not read: 18 search rows of six runs my layers did not trace (106f, e30a, 332f, 2051, 217f, d6b1; none on the
   page). 12 of them have a = width x 16 (332f prints rho 1 with nothing binding; 2051, 217f, d6b1 print no rho);
   e30a's is width x 32 (rho 1, nothing binding); 106f's is a narrower cut (the products term binds).
 - Delta_grad is null on 921 of the 1,332 search rows (124 of the page's 166). On 906 of them (124) the traced step
   holds state across steps (momentum or a latent), and 891 (124) print a rho; on 12 it holds none; 3 not read.

b. a AT 32 BITS A COORDINATE on those 1,116 rows (build a32): no printed cell moves; binding_kind and p the same on
   every row. By class of the 1,116: the products term binds on 981 (a does not enter it); nothing binds on 117
   (rho 1); 18 print no rho (unmeasured). Only rho_inputs.a_bits reads double.

c. THE THREE QAT ROWS with QAT's traced inputs (sigma and t the table's), at each of the three gain targets:
   run@step     rho_default  rho_served  rho_own   printed = as-if, to the last bit; factor 1, no direction
   8f39@1219    215,501      4,003       186,983
   38c6@2639    175,496      3,260       152,273   (the one on the page)
   5fae@3599    125,552      2,332       108,937
   Inputs set, 38c6 (the others alike): a 8,192 -> 16,384; u(1) 1.541e8 -> 1.927e7 (2 bits an entry and 16 a
   matrix; 8 times smaller); Delta_t 1.075e7 -> 6.358e8 (66 bits an entry); Delta_grad null -> 9.248e8 (96 an
   entry). The products term binds before and after, the sensitivities (kinds) are equal, p is equal.

WHAT THIS DOES NOT SAY: that the inputs are right as printed (a is 2 times low on 1,116 rows, u(1) 8 times loose on
the QAT rows, Delta_grad absent on 906 rows whose step holds state); only that no printed rho, p or binding reads
them today, because the products term binds wherever a rho above 1 is printed. A reader of rho_inputs is misled; a
reader of the cells is not. The update kind of the Fit stays unchecked on the 906.

One thing for the next build's check, found here: the control build (the scorer's state as of the last refresh) is
not the published table on 45 rows of 15 runs (kinds' reduction and update, half; fresh_per_c's 7th digit; no rho),
which is exactly what the published build's own layer changed. So an as-if is compared with the published layer's
rebuild, never with the control. All my comparisons are.

