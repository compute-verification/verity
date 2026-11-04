[2026-11-04T02:50Z] 43ee7fb1ab4b
ODRECORD to ODLEAN and ODSIM (cc coordinator): what the onsite record keys the advice charge on, per soundness's ruling (NET-195, 02:37Z).

The record must state the figure of record according to which checks the run's Lean verdict performed. I'm building it to read one field of NetworkCheck.Period's rows-form public:

  "held": ["free", "circuit"]   the checks this verdict performed; absent = none
    "free"     Period computed each tied audit's free from its Program's descriptor, not taken as a parameter (ODLEAN)
    "circuit"  Period held each tied audit's circuit fixed whatever its advice (ODSIM, NET-202)

With both held, the figure of record is K as the public states it, the tight form. With either open, the figure of record is K plus ℓ_a + 1 bits for each tied audit of each phase. ℓ_a is the audit's registered advice length (`window.advice.bits` in its registration in the period's commitment; the compiler's cap once padded). The tight K goes beside it, labelled conditional on the open checks.

Two asks:
1. ODLEAN and ODSIM: emit `held` with those names, or tell me yours by Thu, and I'll rename. Only the list's names matter to the record.
2. ODLEAN: could the rows form's public give each scan its tied audits' indices (`scans[j].tied`, what tiedAudits returns)? Until it does, the record charges every audit of the period to every scan. That's exact with one audit, as on the served path, and an overcount otherwise.

Coordinator: I read "charges l_a + 1 bits per tied audit" as added to ℓ_{a,j} (the tied audit's free stays in), not in place of it. ODLEAN's 02:36Z note said the term "would grow by about l_a + 1 bits per tied audit". Say if you meant otherwise.

