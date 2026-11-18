[2026-11-18T10:31Z] 1f9b8af8592f
From proofs lead, 10:42Z: FYI for your merge of the tip: another worker (CROSS) moved the record digest to verity/commitments/docket_digest.py (b5f7c0c7f7b) and made one-line import changes in rec_live, rec_session, rec_vstar (lazy imports kept lazy), and moved population_program to verity.circuits.population at 213 sites (98912e282f0) but LEFT the re-export in one_stage/partition.py because your eight stagers still call P.population_program: after your branch lands, repoint your eight to verity.circuits.population and tell me so the re-export can be deleted.

