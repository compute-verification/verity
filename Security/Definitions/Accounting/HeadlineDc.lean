import Definitions.Accounting.HeadlineDc.Encoding
import Definitions.Accounting.HeadlineDc.Sm90
import Definitions.Accounting.HeadlineDc.Sm90Measured
import Definitions.Accounting.HeadlineDc.Sm90MeasuredI3
import Definitions.Accounting.HeadlineDc.Sm100
import Definitions.Accounting.HeadlineDc.Sm100Measured

/-! The data-centre families' headline instances' exact data: the encoding beside `Headline.Encoding`'s
(`HeadlineDc/Encoding.lean`: a filler, a certificate on several facets) and one file per family and reading, which
`benchmarks/pouw/lean_headline_dc.py` writes (`HeadlineDc/Sm90.lean`: the H200, DS-90 and Dither-2S LI-CP-W with
ffill; `HeadlineDc/Sm90Measured.lean`: the H200 on DC-90-ri; `HeadlineDc/Sm90MeasuredI3.lean`: the same reading at
(I3)'s `q = q_T = 3/5`, its twin; `HeadlineDc/Sm100.lean`: the B200, DS-100 and Dither-2S LI-CP-W heavy with ♭
crediting no LDTM; `HeadlineDc/Sm100Measured.lean`: the B200 on DC-100-ri). -/
