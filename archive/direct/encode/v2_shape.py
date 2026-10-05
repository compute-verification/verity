"""The B-Ligero witness shape at the checker-v2 relation (b-v2 lane): counts from the v2 census + the BabyBear repairs.

    uv run --frozen --package verity-numerical python archive/direct/encode/v2_shape.py [--out archive/direct/encode/v2_shape.json]

Layout (``V2_SHAPE.md``): the TRANSPOSED Ligero layout of ``LINTEST.md`` 2.3 -- every AIR column of the unit (a value, a hint,
a lookup output, a product, a helper coordinate) is a row class; row ``(column j, block b)`` holds slot ``j`` of the
``l = 3840`` consecutive units of block ``b``.  Quadratic triples and lookup helper identities are then aligned by unit
index and need no operand copies; the linear test's ``r^T A`` is a per-row scalar times one of ``M = 2`` fixed rows
(the geometric row ``G = (rho^(u C))_u`` for witness and multiplicity rows, the constant row for the helper rows' sum
constraint) except on the table-side helper rows.  Everything the Ligero tests touch is committed: per table row the
multiplicity ``m_v`` (1 ``F_t``) and the table-side inverse ``h_v`` (6 ``F_t`` coordinates), so the 6.26M fixed-table rows
(SHIFT 3.5M, TNORM / TNORM_HI 1M each, ...) cost 7 committed elements each PER PROOF -- the price of v2's lookups, and
why v2 only wins above the batch size computed in ``crossover``.

Lookups (``PROTOCOL.md`` 4.2, logUp with helper columns): per query ``j`` a committed helper ``h_j = 1/(z_T - w_j(beta_T))``
in ``F_c`` (6 coordinates); the identity ``h_j (z_T - w_j(beta_T)) = 1`` is one quadratic triple OVER ``F_c`` (``x = h_j``
committed, ``y = z_T - sum_c beta^c w_{j,c}`` and ``z = 1`` VIRTUAL: linear in committed rows with public coefficients,
evaluated from the opened columns by the verifier), so it costs no committed slot beyond ``h_j`` and no linear constraint.
The sum constraint ``sum_j h_j - sum_v h_v = 0`` per table (6 coordinate equations, constant coefficients) and the
table-side ``h_v (z_T - t_v(beta_T)) = m_T(v)`` (6 linear ``F_t`` equations per table row, coefficients vary with ``v``)
are the linear constraints of the lookup argument.

Variant (b), ``table_side = "public_coefficients"`` (MODELLED, not the protocol as written): the table side of the
fractional sum is ``sum_v m_v / (z_T - t_v)`` with PUBLIC coefficients ``1/(z_T - t_v(beta_T))``: no ``h_v`` rows (1
committed element per table row instead of 7), 1 linear constraint per table instead of ``|T| + 1``, the verifier
computes the ``|T|`` public inverses itself.  Counted for the crossover discussion only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VERITY_CHECKER_VARIANT", "v2")
if os.environ["VERITY_CHECKER_VARIANT"] != "v2":
    raise SystemExit("v2_shape.py describes checker v2: VERITY_CHECKER_VARIANT must be v2 (or unset)")

from verity_numerical.explore import fields as fl
from verity_numerical.explore import tensor as t
from verity_numerical.security import census

K, L, N = fl.K_CODE, fl.L_PAYLOAD, fl.N_CODE
D = 6                                    # BabyBear^6 at 2^-128
UNITS_PER_VU = 96
BLINDING_ROWS_PER_TABLEAU = 3            # LDT, linear-test, quadratic-test blinding rows (PROTOCOL.md section 3)
TABLEAUX = 2                             # round 1 (witness, triples, multiplicities), round 2 (helpers)
V1_ROWS = 310_602                        # tensor.py at v1 (b_measured_4090.md)
V1_QUAD_ROWS = 33_485
V1_HELPERS = 125_895_743
V1_TABLE_ROWS = 66_623


def shape(B: int = 4096, table_side: str = "helper_rows") -> dict:
    u = census.census_unit()
    fs = t.FIELDS["babybear"]
    sfc = t._small_field(fs)                                    # v2 census + BabyBear repairs (fields.small_field_census)
    tables = sfc.tables()
    units = B * UNITS_PER_VU
    blocks = -(-units // L)
    # -- per unit, F_t slots (transposed layout: one slot per AIR column, no operand copies) --------------------------
    repairs = [d.note for d in sfc.deltas]
    extra_cols = sfc.columns_committed_b - u.columns_committed_b   # the repairs' hint / input / range columns
    cols = {"inputs": u.columns_inputs, "hints": u.columns_hints, "lookup_outputs": u.columns_lookup_outputs,
            "products": u.columns_products, "materialised": u.columns_materialised, "babybear_repair_columns": extra_cols}
    witness_slots = u.columns_all + extra_cols
    lookups = sfc.lookup                                        # 89 (v2) + repairs (F4 limb ranges, sign tightening, TNORM_HI)
    helper_slots = D * lookups                                  # h_j coordinates, round 2
    mul = sfc.mul                                               # F_t quadratic triples per unit
    lin = sfc.lin_total                                         # F_t-linear identities per unit (incl. range-chunk splits)
    # -- tables -----------------------------------------------------------------------------------------------------------
    table_rows = sum(tb.rows for tb in tables)
    per_table_row = 7 if table_side == "helper_rows" else 1
    table_elements = table_rows * per_table_row
    # -- rows ----------------------------------------------------------------------------------------------------------------
    witness_rows = witness_slots * blocks                       # tableau 1
    helper_rows = helper_slots * blocks                         # tableau 2
    mult_rows = -(-table_rows // L)                             # tableau 1 (multiplicities)
    hv_rows = -(-D * table_rows // L) if table_side == "helper_rows" else 0   # tableau 2 (table-side inverses)
    blinding = BLINDING_ROWS_PER_TABLEAU * TABLEAUX
    rows = witness_rows + helper_rows + mult_rows + hv_rows + blinding
    committed = units * (witness_slots + helper_slots) + table_elements
    # -- constraints ----------------------------------------------------------------------------------------------------------
    transition_lin = 4                                          # chain linkage per unit (LINTEST.md 2)
    n_linear_unit = units * (lin + transition_lin)
    n_linear_tables = (table_rows * D + D * len(tables)) if table_side == "helper_rows" else D * len(tables)
    n_linear = n_linear_unit + n_linear_tables + 2
    quad_ft_rows = mul * blocks                                 # F_t triples (x, y, z row classes), aligned by unit
    quad_fc_rows = lookups * blocks                             # F_c triples (6 coordinate rows each), y and z virtual
    n_quadratic = units * (mul + lookups)
    # -- linear-test structure ------------------------------------------------------------------------------------------------
    M = 2                                                       # G (witness + multiplicity rows), ONES (helper rows: the sum)
    X = hv_rows if table_side == "helper_rows" else mult_rows   # table-side rows through the general path
    X += 2 * (witness_slots + helper_slots) // L + 2            # boundary units (unit 0, last unit) touch <= 2 rows per class... bounded
    # -- helpers to invert -----------------------------------------------------------------------------------------------------
    helpers = lookups * units + (table_rows if table_side == "helper_rows" else 0)
    verifier_public_inversions = table_rows if table_side == "public_coefficients" else 0
    eb = 4
    return {
        "B": B, "K": 1536, "units": units, "blocks": blocks, "block_units": L, "k": K, "l": L, "n": N, "ext_degree": D,
        "table_side": table_side, "layout": "transposed: row (AIR column j, block b) = slot j of 3840 consecutive units",
        "per_unit": {"witness_slots": witness_slots, "columns": cols, "hints": u.columns_hints, "lookups": lookups,
                     "lookups_by_table": {tb.name: tb.queries_per_unit for tb in tables}, "helper_slots": helper_slots,
                     "mul_ft_triples": mul, "fc_triples": lookups, "lin": lin, "transition_lin": transition_lin,
                     "committed_slots": witness_slots + helper_slots, "babybear_repairs": repairs,
                     "v2_census_unrepaired": {"mul": u.mul, "lin": u.lin_total, "lookups": u.lookup, "hints": u.columns_hints,
                                              "columns_all": u.columns_all}},
        "tables": {tb.name: {"rows": tb.rows, "columns": tb.columns, "queries_per_unit": tb.queries_per_unit} for tb in tables},
        "table_rows_total": table_rows, "committed_per_table_row": per_table_row, "table_elements": table_elements,
        "rows": {"witness": witness_rows, "helper": helper_rows, "multiplicity": mult_rows, "table_inverse": hv_rows,
                 "blinding": blinding, "total": rows, "per_vu_marginal": (witness_rows + helper_rows) / B,
                 "fixed_per_proof": mult_rows + hv_rows + blinding},
        "committed_elements": committed, "committed_bytes_per_vu": committed * eb / B, "encoded_bytes": rows * N * eb,
        "linear_constraints": {"unit": n_linear_unit, "tables": n_linear_tables, "total": n_linear},
        "quadratic": {"ft_triple_rows": quad_ft_rows, "fc_triple_rows": quad_fc_rows, "constraints": n_quadratic,
                      "fc_triple_witness_reads_per_unit": sum(tb.columns * tb.queries_per_unit for tb in tables)},
        "lintest": {"M": M, "exceptional_rows": X, "classes": ["G = (rho^(uC))_u on witness and multiplicity rows",
                                                                "ONES on helper rows (the per-table sum constraint)"]},
        "helpers_to_invert": helpers, "verifier_public_inversions": verifier_public_inversions,
        "t_open": 191, "hvzk": {"randomizers_per_row": K - L, "blinding_rows": blinding},
        "v1": {"rows": V1_ROWS, "quad_rows": V1_QUAD_ROWS, "helpers": V1_HELPERS, "table_rows": V1_TABLE_ROWS,
               "committed_bytes_per_vu": 1_164_743},
        "model_v2_aligned_triples": {"rows": 105_312, "committed_bytes_per_vu": 394_906, "note": "explore.tensor at VERITY_CHECKER_VARIANT=v2 (3 slots per mul, 1 per other column)"},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--B", type=int, default=4096)
    a = ap.parse_args(argv)
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "lane": "b-v2",
           "helper_rows": shape(a.B, "helper_rows"), "public_coefficients": shape(a.B, "public_coefficients")}
    for name, s in (("helper_rows", out["helper_rows"]), ("public_coefficients", out["public_coefficients"])):
        r = s["rows"]
        print(f"{name}: rows {r['total']:,} (witness {r['witness']:,} + helper {r['helper']:,} + mult {r['multiplicity']:,} + "
              f"h_v {r['table_inverse']:,} + blinding {r['blinding']}); committed {s['committed_elements']:,} = "
              f"{s['committed_bytes_per_vu'] / 1e3:.1f} kB/VU; lin {s['linear_constraints']['total']:,}; quad F_t rows "
              f"{s['quadratic']['ft_triple_rows']:,}, F_c rows {s['quadratic']['fc_triple_rows']:,}; helpers {s['helpers_to_invert']:,}; "
              f"M {s['lintest']['M']}, X {s['lintest']['exceptional_rows']:,}")
    print("per unit:", json.dumps(out["helper_rows"]["per_unit"], default=str))
    print("tables:", json.dumps(out["helper_rows"]["tables"]))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
