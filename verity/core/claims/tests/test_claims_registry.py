"""The claims vocabulary: composed assumptions, stable ids, kinds."""

from __future__ import annotations

import pytest

from verity.core.claims import ENTRIES, INSTANCES, PROPERTIES, assumption, get


def test_a_primitive_assumption_is_property_slash_instance_with_its_label_and_standing() -> None:
    e = get("cr/sha-512")
    assert (e.id, e.kind, e.name, e.label, e.standing) == ("cr/sha-512", "assumption", "Collision resistance of SHA-512", "CR · SHA-512", "standard")
    assert assumption("cr", "poseidon2-koalabear").standing == "algebraic" and get("xof/shake-256").label == "XOF · SHAKE-256"
    assert all(get(f"{p}/{i}").id == f"{p}/{i}" for p in PROPERTIES for i in INSTANCES)
    e = get("ecr/sha-512")
    assert (e.name, e.label, e.standing) == ("Expected-time collision resistance of SHA-512", "ECR · SHA-512", "standard")
    e = get("prf/sha-512")
    assert (e.name, e.label, e.standing) == ("Pseudorandom function of SHA-512", "PRF · SHA-512", "standard")
    e = get("cr/turboshake128")
    assert (e.name, e.label, e.standing) == ("Collision resistance of TurboSHAKE128", "CR · TurboSHAKE128", "standard")
    assert get("uniform/python-secrets").name == "Independent uniform output of Python's secrets"
    assert get("uniform/flock-coin-server").name == "Independent uniform output of Flock's coin server"
    e = get("euf-cma/lms-sha512-n64")
    assert (e.label, e.standing) == ("EUF-CMA · LMS over SHA-512", "standard")
    assert "lms-sha512-n32" not in INSTANCES
    e = get("uniform/io-getrandombytes")
    assert (e.name, e.label, e.standing) == ("Independent uniform output of IO.getRandomBytes", "Uniform · IO.getRandomBytes",
                                             "standard")


def test_entries_are_keyed_by_id_and_unknown_ids_raise() -> None:
    assert all(k == e.id for k, e in ENTRIES.items()) and all(k == p.id for k, p in PROPERTIES.items())
    assert all(k == i.id for k, i in INSTANCES.items())
    assert {e.kind for e in ENTRIES.values()} == {"guarantee", "assumption", "model"}
    for bad in ("cr/md5", "nope/sha-256", "nothing"):
        with pytest.raises(KeyError):
            get(bad)


def test_composed_ids_never_shadow_entries() -> None:
    assert not set(ENTRIES) & {f"{p}/{i}" for p in PROPERTIES for i in INSTANCES}


def test_bare_ids_are_kebab_case_without_a_slash() -> None:
    import re

    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", k) for k in ENTRIES)
    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", k) for k in (*PROPERTIES, *INSTANCES))


def test_bandwidth_device_claims_are_measured_and_cited_separately() -> None:
    device, inst = get("bandwidth-device"), get("bandwidth-instantiation")
    assert (device.kind, device.standing) == ("model", None)
    assert all((get(i).kind, get(i).standing) == ("assumption", "measured")
               for i in ("bandwidth-s-chip", "bandwidth-side-cap", "bandwidth-regen-time"))
    assert (inst.kind, inst.standing) == ("assumption", "conjectured") and "uniformly random blocks" in inst.name


def test_rate_audit_claims_are_cited_separately() -> None:
    device, work = get("rate-device"), get("rate-codec-work")
    assert (device.kind, device.standing) == ("model", None) and "L - rho" in device.name and "non-adaptive" in device.name
    assert (work.kind, work.standing) == ("assumption", "measured") and "x * c_enc / b" in work.name
    assert "k >= 2 rounds" in work.name and "false when the payload is known" in work.name
    assert "Incompressible(W)" in work.name
    verbatim = get("rate-sketch-verbatim")
    assert (verbatim.kind, verbatim.standing) == ("model", None) and "only as its blocks" in verbatim.name
    assert "precomputed sketches" in verbatim.name


def test_pous_end_to_end_rings_claims_are_cited_separately() -> None:
    device, speed, inst = get("pous-ring-device"), get("pous-ring-speed"), get("pous-ring-instantiation")
    assert (device.kind, device.standing) == ("model", None) and not any(c.isdigit() for c in device.name)
    assert (speed.kind, speed.standing) == ("assumption", "measured") and "D = 16,446" in speed.name and "Q = 2^20" in speed.name and "L = 2^64" in speed.name
    assert (inst.kind, inst.standing) == ("assumption", "conjectured") and "SHA-512-CTR" in inst.name
    assert get("prf/sha-512").kind == "assumption"


def test_the_sm120_headlines_claims_name_what_their_statements_assume() -> None:
    """HEADLINE-R2's restatement: FOLD-OPT at π*, binding's fork form, A2's catch-all and A1 assumed at the driver's
    report."""
    fold = get("accounting-ts-fold-opt").name
    assert "FoldOptFoldedAt" in fold and "only at their own certificate π*" in fold
    proofs = get("accounting-ts-proofs-sound").name
    assert all(s in proofs for s in ("BindingForkAt", "√δ_fork", "BindingOfFork", "ProofSoundAt", "accounting-ts-proofs-pin"))
    assert "every proof of round w is accepted" in proofs and "both accepted" not in proofs and "ProofsPin" not in proofs
    assert "is not stated there" in proofs
    assert "catch-all" in get("accounting-sheet-dispatch").name and "NOROW" in get("accounting-sheet-dispatch").name
    clock = get("accounting-sheet-clock")
    assert "nvidia-smi" in clock.name and "data sheet's maximum" not in clock.name
    assert clock.label == "A1 clock (assumed)"


def test_the_sm120_headlines_claims_match_their_lean_after_round_3() -> None:
    """RT3's claim-text findings: (I1)'s vector, the writers' ω, the ½-floor reading of 0.591%, and pinning graded on
    its own."""
    facet = get("accounting-ts-facet-ceilings").name
    assert "VF3s-16" in facet and "VF3p-16" not in facet and "(0, f_max]" in facet
    words = get("accounting-ts-words-written").name
    assert "UNI64" in words and "32 for IMMA.8816" in words
    scalar = get("accounting-ts-scalar-route").name
    assert all(s in scalar for s in ("9/8", "0.591%", "7.791%", "10.24% on VF3r-16", "SheetScalarRouteHalf"))
    pin = get("accounting-ts-proofs-pin")
    assert (pin.kind, pin.standing) == ("assumption", "conjectured") and get("accounting-ts-proofs-sound").standing == "standard"
    assert all(s in pin.name for s in ("ProofsPin", "PouwSegCheck_v2 lacks it", "PouwSegCheck_v3", "#2417"))


def test_the_scalar_route_claim_names_the_h200_twin_with_its_premise() -> None:
    """DC-90-ri's (I3) twin takes SCALAR-ROUTE at 3/5 beside sm_120's 9/8, and its citation names the premise, the
    proved ½ floor beside it and the fold menus the credit and the cheater read."""
    scalar = get("accounting-ts-scalar-route").name
    assert all(s in scalar for s in ("Sm90MeasuredI3Headline", "Sm90MeasuredHeadline", "q = q_T = 3/5", "9/8",
                                     "Both are over 1%",
                                     "DC-90-ri 1.086% under (I3) at 3/5; 1.412% at the proved ½ floor; credit against "
                                     "the built ♭, the cheater priced at every known fold, counted ones included"))


def test_the_honest_clock_says_which_form_is_proved() -> None:
    clock = get("honest-clock")
    assert (clock.kind, clock.standing, clock.label) == ("assumption", "standard", "Honest clock")
    assert "Accounting.SecurityProofs.Actual.honestClock" in clock.name and "the Pearl-C form is not proved" in clock.name


def test_the_session_premises_state_their_scope_after_rt4_logic_25_and_27() -> None:
    """RT4-LOGIC-25: no served path verifies with --registered; -27: the window's law also needs the audit coins."""
    live = get("accounting-ts-accepted-live")
    assert live.standing == "standard"
    assert all(s in live.name for s in ("plain flock-verify --zk", "only one_stage/a0.py passes --registered",
                                        "no served path meets it"))
    law = get("accounting-ts-window-law").name
    assert all(s in law for s in ("five facts", "coin tree", "trusted-host", "Under full checking",
                                  "serve --zk --operator loopback", "doesn't meet the fifth fact"))


def test_pouw_gamma_per_coin_is_a_conjecture_at_w1_priced() -> None:
    e = get("pouw-gamma-per-coin")
    assert (e.kind, e.standing) == ("assumption", "conjectured") and "budget per coin" in e.name and "W1 priced" in e.name


def test_option_a_counts_fixed_is_a_conjecture_and_modes_run_standard() -> None:
    assert (get("accounting-ts-counts-fixed").kind, get("accounting-ts-counts-fixed").standing) == ("assumption", "conjectured")
    assert (get("accounting-ts-modes-run").kind, get("accounting-ts-modes-run").standing) == ("assumption", "standard")


def test_the_dc_headlines_claims_name_what_their_statements_assume() -> None:
    """RT3-LOGIC-6's restatement of the H200's and B200's headlines: FOLD-OPT at π* and the proof service's named
    premises, as sm_120's."""
    for e in ("accounting-ts-fold-opt", "accounting-ts-proofs-sound"):
        assert "the H200's and the B200's" in get(e).name, e
    assert "RT3-LOGIC-6" in get("accounting-ts-fold-opt").name


def test_the_sites_nine_assumptions_resolve_and_the_retired_ones_do_not() -> None:
    site = ("cr/sha-256", "cr/sha-512", "cr/blake3", "cr/blake2b", "cr/poseidon2-babybear", "cr/poseidon2-koalabear", "xof/shake-256",
            "random-oracle", "honest-verifier")
    assert all(get(i).id == i and get(i).kind == "assumption" for i in site)
    assert not [k for k in ENTRIES if any(r in k for r in ("beacon", "auditor", "challenger"))]


def test_the_registrant_premise_is_a_model_of_the_provers_covered() -> None:
    e = get("registrant-holds-openings")
    assert (e.kind, e.standing, e.label) == ("model", None, "Registrant holds openings")
    assert "a row and salt under every leaf it registered" in e.name and "outside the guarantee" in e.name
    assert "its own workers behind the firewall hold the salts they committed R's level-0 top rows under" in e.name


def test_the_trusted_host_keeps_the_devices_persistent_state() -> None:
    e = get("trusted-host")
    assert (e.kind, e.standing) == ("assumption", "standard")
    assert e.name == ("The auditor's device runs its stated code, timing included, on a host the prover doesn't control, "
                      "with its CPU isolated from other tenants and a monotonic clock; the host never restores its earlier "
                      "state")


def test_the_latency_ceiling_is_a_measured_device_claim() -> None:
    e = get("latency-ceiling")
    assert (e.kind, e.standing, e.label) == ("assumption", "measured", "Latency ceiling")
    assert "network certifier on its egress" in e.name and "within the stated ceiling" in e.name


def test_the_declared_capacity_rests_on_the_nodes_inventory() -> None:
    """ASK-74 = A: Π's three clauses, per node, trusting the auditor's inventory, with no partition clause."""
    e = get("accounting-declared-device")
    assert (e.kind, e.standing, e.label) == ("assumption", "standard", "Declared device")
    assert all(s in e.name for s in ("every GPU on the node is declared", "the declared board and vBIOS",
                                     "no SM clock exceeds f_max, the top of that vBIOS's clock table",
                                     "MIG and MPS need no clause", "the auditor's inventory of the node"))
