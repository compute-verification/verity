"""A class-sweep record and a probed hardware object shaped as the prover run's (``class_statement.prove``'s LIVE fields)."""
import copy

import pytest

REP = {"t.witness": 0.4, "t.witness_comp": 0.3, "t.witness_units": 0.1, "t.encoding_commitment": 0.9, "t.link": 0.05,
       "t.zerocheck": 0.7, "t.lincheck": 0.2, "t.ring_switch": 0.1, "t.ligerito": 0.5, "t.total": 2.9, "t.arithmetic": 1.5,
       "h.setup": 0.02, "h.stage": 0.05, "h.decode": 0.01, "h.serialize": 0.01, "host_s": 0.1}

LIVE = {"accepted": True, "why": None, "prover": "gpu", "instances": 16, "k_log": 25, "dense_m": 29, "e2e_s": 8.5,
        "witness_s": 0.6, "witness_prebuilt_s": None, "pipeline_depth": 4, "prove_s": [3.0, 3.0], "prove_total_s": 6.0,
        "proof_bytes": [900_000, 900_000], "wire_sent": 1_850_000, "wire_recv": 4_000, "statement_digest": "ab" * 32,
        "buckets": [REP, REP], "prover_peak_gb": 12.5, "verifier_peak_gb": 1.1, "verifier": "loopback",
        "coins": "live-os", "verify_s": 0.8, "timed_runs": 3}

HARDWARE = {"gpu": {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "driver": "580.95.05", "cuda": "13.0", "count": 8,
                    "memory_total_bytes": 97887 << 20},
            "cpu": {"model": "AMD EPYC 9575F 64-Core Processor", "count": 256}, "affinity_cpus": 32}


@pytest.fixture
def record():
    return {"shape": "24dbfd96f26702eb" * 4, "domain": "total", "ands": 123, "live": copy.deepcopy(LIVE), "stage_setup_s": 41.0}


@pytest.fixture
def hardware():
    return copy.deepcopy(HARDWARE)
