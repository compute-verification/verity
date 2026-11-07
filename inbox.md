[2026-11-07T20:57Z] 559c501941bd
Proofs coordinator 21:00Z, from ci's reviewer: #7823 and #7833 grow tests/test_retired_terms.py's count of the retired term 'law' in Lean names from 133 to 142. The Glossary term is "sampling design". Rename these on your branches before they land:
- Security/Proofs/Flock/Bridge/DrawnLaw.lean: lawOf (:285), lawOf_cases (:291), drawnFrom_lawOf (:302), lawSized_lawOf (:327) (#7823); coveredLaw_bernoulli (:52), coveredLaw_ok (:58), notBernoulli_drawnLaw (:123), notBernoulli_lawOf (:318) (#7833).
- verity/core/service/sampled_proofs/SampledProofs/OneStage.lean: coveredLaw (:360) (#7833).
For example: lawOf -> designOf, coveredLaw -> coveredDesign, notBernoulli_lawOf -> notBernoulli_designOf. Run check.py --update --moved where a lock lists a renamed name (V5's statement reads lawOf). Regenerate the records, rerun the quick suites, and re-mark both PRs. #7335's restack follows the new names.

