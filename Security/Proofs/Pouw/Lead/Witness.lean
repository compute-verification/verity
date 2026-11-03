import Proofs.Pouw.Lead.Witness.Deployed

/-!
# The lead's witnesses

* `Deployed`: the honest and nontrivial witnesses of `G`, `G_tail`, `NetworkFwd` and `NetworkLlama` at one deployed
  instance, Llama-3.2-1B's `2048 × 2048` linear at T 512 × 64 under SHA-512 (`G_honest`, `G_nontrivial`,
  `G_tail_nontrivial`, `networkLlama_honest`, `networkFwd_nontrivial`, `networkLlama_nontrivial`).

Its proofs go in `Lead/Witness/`, its declarations in `Pouw.SecurityProofs.Lead.Witness`, and each file is imported
here.
-/
