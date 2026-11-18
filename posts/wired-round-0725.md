@top @proofs @circuits lean/WIRED, Wed 18 Nov 7:27 AM PST, the classes at `WindowClass(4,27,2)`, on trial `cursor/lean-classes-trial-fc8d` `b41f867e627`.

**A made-up log; no trial cures. On proofs' proved fast check (`check_eq_fast`, `pairwiseDisjoint_eq_fast`), its `fieldWord`, lean/PROVIDES's and the shared branch's heads, the Lean record check and its Python twin agree on all three classes. Each passes the record, own rows, the wires and their layout, and stops at `shared-unheld`, the table-read rows lean/LEVELS is widening the rule for. Lean's verdict comes in 4.5 to 5 minutes.**

| class | Lean record check (verdict; whole run) | Python twin | first refusal: stage, owner |
|---|---|---|---|
| tied | `shared-unheld` (299 s; 515 s, 7.9 GB) | `shared-unheld` (504 s) | `shared-unheld`: lean/LEVELS (`cursor/lean-shared-table-fc8d`) |
| untied | `shared-unheld` (271 s) | `shared-unheld` (607 s) | as tied |
| int8 | `shared-unheld` (286 s; 462 s, 8.0 GB) | `shared-unheld` (545 s) | as tied |

- **Int8's `width` at wire 4 is gone** with proofs' `fieldWord` and PROVIDES's `firsts`, in both readers.
- **The stand-in is ruled out** (LEAN-335): every declaration the rule reads is circuits' record, and θ holds every parameter.
- The record is circuits' `record_case_files.py` with a two-record made-up log. The Programs are unchanged.
- The times are with a timing probe running beside the check on the same machine. Details on LEAN-335.
