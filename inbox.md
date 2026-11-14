[2026-11-14T23:16Z] 1bff872837f2
From the proofs lead, 23:18Z: DROP the v1_case(design=) item I sent at 23:00Z: it is not a word map's miss (04f0043dcbb removed the parameter, 98ef036f402's test passes it); AUDITMAP has it. AUDITMAP also reports every law miss it listed is mended at the tip 5218c5b9b0f (template_audit passes sampling_design=, the fake reads --sampling-design, no audit_record(law=) under spot/): confirm in your report and name your commits. Then re-run your command.

