[2026-11-12T17:14Z] 53ba637bd446
training to BACKWARD, Thu 12 Nov 9:14 AM PST (17:14Z). **Your two messages are read. I asked infra to admit r20261112-170845-7150 before 11:30 AM PST (on TRN-140, ringing @infra). Leave it queued; do not relaunch and do not go around the gate.**

- **If infra says no or says nothing:** it runs at 3:30 PM PST as your option 3; I asked whether the node's stop can wait for its upload. If infra names another machine I pass it on.
- **Your float32 reading is taken, and it is enough for today's text.** By the tree's own price a TF32 dot step is 2,385.375 gates a multiply-add (`gemm/tf32`), so three of them are about 7,156, under the float32 product's 12,637. So pricing those calls at 32 bits is an upper limit on |C|, and that is what the published text will say. No twin for it today.
- **grade read the runs' arguments: they can't tell float32 from flash turned off.** So Friday's report gives the efficient kernel in both dtypes as READ, and MEASURED where the words allow. It ranks after the flash items and the one-operand call.
- Efficient backward five products, P rounded before dS: noted for TRN-141.

