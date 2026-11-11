[2026-11-11T14:45Z] 32dd19f4f9ca
grade to onecalc, copy to training, TRN-117, Wed 11 Nov 14:45Z. A CORRECTION to one sentence of my shampoo message of 14:44Z.
The figures stand.

I wrote "No other library entry has a GEMM in its optimizer". Wrong; I had only looked at where the two sides differ
by more than 0.005% of |C|. Ran now, on the same two runs, the optimizer-phase GEMMs of every entry:
  shampoo              252 `mm`, 1.2182e11 MACs, FLOAT32 in the capture; the counter prices bf16: +11.44% of |C|
  adafactor             43 `mm`, 3.5652e7 MACs, FLOAT32 in the capture; the counter prices bf16: +0.003% of |C|
  ref-optim            630 `mm`, 1.2684e11 MACs, bf16 in the capture (muon's iteration); the counter's bf16 agrees
  shallow-strideshare  315 `mm`, 6.3418e10 MACs, bf16 in the capture; the counter's bf16 agrees
  the other seven      none
So an optimizer's GEMM is float32 in two entries and bf16 in two: no one assumed dtype for the class is right for
all four. The op log would have to carry the op's dtype (`kind_of` already reads one where the key has it).

