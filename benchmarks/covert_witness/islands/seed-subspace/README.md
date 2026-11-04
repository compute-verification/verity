# Direction `seed-subspace` (v3)

Inference-shaped training: θ stays frozen and every matrix product the step runs has an operand that doesn't depend on
θ′ (θ, a basis computed once from θ, or a basis drawn from a registered seed); only vectors train, applied elementwise.
`train.py`'s docstring has the forward and E. The mechanisms, each a `--note` tag:

- `scale`: column and row scalings of each adapted matrix of θ (`--scale`) and the norm gains; no extra product.
- `vera`: ΔW = Σ_d z_d b_d a_dᵀ over R seeded rank-1 directions shared by every block (`--vera R`), z = λ_b ⊙ λ_d, applied
  as x·Aᵀ then ·Bᵀ.
- `svf`: θ's top R singular directions of each matrix (`--svf R`), their values shifted by δ.
- `growreg`: each block followed by a frozen copy of itself, gated at 0 through the copy's wo and w_down row scalings
  (`--grow`), with any of the above.

    python benchmarks/covert_witness/launch.py s-t20 --note "vera ..." -- --scale --vera 512 --lr 3e-3
    python benchmarks/covert_witness/launch.py s-t20 --note "growreg ..." -- --grow --vera 128
