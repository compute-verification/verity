"""RT4-INT's witnesses on #3053's served credit (H1 serving through ``verity_int8::scaled_mm_zy``, its z/y records the z
table, `pouw_h1_records`): what a window's reported credit rests on, as the code runs.

* RT4-INT-31: an opaque op that multiplies nothing (y zero, every record zero) is one GEMM to the credit gate's trace and its
  records hold together, so the window reports every served MAC credited, as it does for the honest kernel; the verifier's
  checks of record (`pouw_h1_challenger.trusted_rounds_reason`, `manifest_reason`) accept both windows, and none of them
  reads or recomputes the credit, which is a field of the prover's record under no root.
* RT4-INT-32: the credited set is under no root the draw coin follows: `calls_root`, and with it the draw root, is over the
  manifest rows, which carry no verdict, so a call's credit is withdrawn or granted after the draw with every root the
  same; and the verifier's checks accept a window whose served call has no z table at all.
* RT4-INT-33: the window Program the registered route proves (`h1window`) reads ``a/<K>``, ``b/<K>``, the salt, the
  positions and the roots, and each unit outputs its z and its receipt and no y; a z-table leaf commits a unit's whole
  6,144-byte record (z, then y) as one row under the table's own binding, which the unit's 4,096-byte z output alone
  doesn't open. No unit's proof reads a z-table leaf, and none computes y = dq(z).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import window as Wd  # noqa: E402
from verity.core.protocols.pouw.audit import HM96, HM96_SALT, ROW_SCHEMA, rows_domain  # noqa: E402
from verity.core.protocols.sampled_proofs import remote as R  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as G  # noqa: E402
from verity.ml.commitments import MerkleTree  # noqa: E402
from verity_vllm import protocol_options as PO  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.engine import int8_ops  # noqa: E402
from verity_vllm.protocol_options import pouw  # noqa: E402
from verity_vllm.protocol_options import pouw_h1 as PH  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_challenger as PHC  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_records as ZR  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_requests as RQ  # noqa: E402

from tests.protocol_options import challenger_host as CH  # noqa: E402

K, ROWS, N, M = 64, 32, 64, 3
ENTRY = f"pouw:{PH.SCHEME}"
AUDITOR = "vy-challenger-1:7811"
RESERVED = {"session": "s-1"}
BINDING = bytes(range(64))

_LIB = torch.library.Library("rt4_int_zy", "FRAGMENT")
_LIB.define("idle(Tensor xq, Tensor s_a, Tensor w, Tensor s_w, int rows) -> (Tensor y, Tensor zy)")
_LIB.define("plain(Tensor xq, Tensor s_a, Tensor w, Tensor s_w) -> (Tensor y, Tensor z)")


def _idle_impl(xq, s_a, w, s_w, rows):
    """No product at all: y zero and every record zero, z = 0 with y = dq(0) = +0 on every cell."""
    return (xq.new_zeros((xq.shape[0], w.shape[0]), dtype=torch.bfloat16),
            xq.new_zeros(((rows // 32) * (w.shape[0] // 32), ZR.ZY_BYTES), dtype=torch.uint8))


def _plain_impl(xq, s_a, w, s_w):
    """One opaque GEMM handing over its exact s32 Y and y = dequant(Y), no records."""
    z = E.int_mm(xq, w)
    return E.dequantize(z, s_a, s_w), z


_LIB.impl("idle", _idle_impl, "CPU")
_LIB.impl("plain", _plain_impl, "CPU")


def honest(xq, s_a, w, s_w):
    return int8_ops.scaled_mm_zy(xq, s_a, w, s_w, ROWS)


def idle(xq, s_a, w, s_w):
    return torch.ops.rt4_int_zy.idle.default(xq, s_a, w, s_w, ROWS)


def plain(xq, s_a, w, s_w):
    return torch.ops.rt4_int_zy.plain.default(xq, s_a, w, s_w)


class Arm:
    """A served linear of one kernel after ``row_quant``: its (y, z) handed to ``observe``, y served."""

    int8_arm = E.ARM

    def __init__(self, kernel):
        self.kernel = kernel

    def apply(self, layer, x, bias=None):
        xq, s_a = int8_ops.row_quant(x)
        y, z = self.kernel(xq, s_a, layer.weight, layer.weight_scale)
        self.observe(layer, xq, s_a, y, z)
        return y

    def observe(self, layer, xq, s_a, y, z=None):
        pass


class Linear:
    def __init__(self, arm):
        g = torch.Generator().manual_seed(1)
        self.weight = torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8)
        self.weight_scale = (torch.rand(N, 1, generator=g) * 0.01 + 1e-4).float()
        self.quant_method = arm


class Model:
    def __init__(self, layer):
        self.layers = [layer]

    def modules(self):
        return iter(self.layers)

    def named_modules(self):
        return iter([("q", self.layers[0])])


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))


def _host(path, monkeypatch):
    """The auditor host's challenger for one window: one session over a one-round live epoch, reached at `AUDITOR`."""
    path.mkdir(parents=True, exist_ok=True)
    host = CH.Host(path / "auditor", Wd.live_challenger(0, 1, CH.entropy(1)), 1, gated=True)
    monkeypatch.setattr(R.RemoteChallenger, "connect",
                        classmethod(lambda cls, at, secret=None: host.client if at == AUDITOR else None))
    return host


def _serve(arm, host, out):
    """One served call of M rows through `arm` under H1 (T = `ROWS`, one step), coins from `host`, its record written to
    `out`: (served y, the run, the record)."""
    layer = Linear(arm)
    runner = SimpleNamespace(execute_model=lambda scheduler_output, *a, **kw: None, decode_query_len=1)
    settings = {"pouw": {"rows": ROWS, "steps": 1, "retain_steps": 1, "runner": runner, "transcript_dir": str(out),
                         "challenger": AUDITOR, "session": RESERVED, "gate": lambda session, root: host.open(session, root)}}
    comp = PO.install(Model(layer), PO.Context(stage="serve", enabled=(ENTRY,), settings=settings))
    try:
        req = SimpleNamespace(req_id="r", prompt_token_ids=list(range(M)), sampling_params=SimpleNamespace(max_tokens=1))
        runner.execute_model(SimpleNamespace(scheduled_new_reqs=[req], num_scheduled_tokens={"r": M},
                                             total_num_scheduled_tokens=M))
        x = (torch.randn(M, K, generator=torch.Generator().manual_seed(9)) * 3).bfloat16()
        y = layer.quant_method.apply(layer, x)
        run = pouw.h1_run(ENTRY)
    finally:
        comp.close()
    return y, run, json.loads((out / PH.CALLS_FILE).read_text())


def _verifier(host, record):
    """The verifier's checks of record on a window, once the host's ledger ends its session: its coins the auditor's
    challenger's, and its manifest under them."""
    assert host.ended() is None
    return (PHC.trusted_rounds_reason(record["commitment"], host.directory, [host.ledger.path], host.window, host.key),
            PHC.manifest_reason(record))


def test_an_op_that_multiplies_nothing_is_credited_every_mac_and_its_window_passes_the_verifiers_checks(tmp_path, monkeypatch):
    """RT4-INT-31. The honest kernel and an opaque op that computes no product (y and every record zero) both pass the
    gate: one GEMM by the trace's rule (an op outside aten with no Python implementation counts as one), the records'
    y = dq(z) at the call's scales, the served y the records', the empty slots' z 0. Both windows report the call's
    M·N·K MACs credited, and the verifier's checks accept both; the zero window's served y isn't the product. The credit
    is a field of the prover's record that no check reads: set to anything, the checks still accept."""
    seen = {}
    for name, kernel in (("honest", honest), ("idle", idle)):
        host = _host(tmp_path / name, monkeypatch)
        y, run, record = _serve(Arm(kernel), host, tmp_path / name / "w")
        (v,) = run.verdicts.values()
        assert v["credit"] and v["gemms"] == [v["kernel"]], v
        assert record["calls"][0]["tables"]["z"] and run.coin_source == PH.CHALLENGER
        assert _verifier(host, record) == (None, None)
        seen[name] = (y.view(torch.int16), run.credit()["credited"], record)
    (yh, ch, rh), (yi, ci, ri) = seen["honest"], seen["idle"]
    assert ch == ci == {"rows": M, "macs": M * N * K}
    assert rh["details"]["credit"]["credited"] == ri["details"]["credit"]["credited"] == ch
    assert yh.any() and not yi.any()
    doctored = json.loads(json.dumps(ri))
    doctored["details"]["credit"]["credited"] = {"rows": 10 ** 9, "macs": 10 ** 18}
    assert PHC.manifest_reason(doctored) is None


def test_the_credited_set_is_under_no_root_the_draw_follows_and_a_call_without_a_z_table_passes(tmp_path, monkeypatch):
    """RT4-INT-32. After the draw coin opened, the honest call's credit is withdrawn: the credit drops to 0 and
    `calls_root` and the draw root are the ones the coin followed. A call the gate refuses for handing over no records
    (one opaque GEMM handing over its s32 Y) commits no z table, and its window passes the verifier's checks; granted
    credit after the draw, it counts M·N·K MACs under the same roots."""
    host = _host(tmp_path / "a", monkeypatch)
    _y, run, record = _serve(Arm(honest), host, tmp_path / "a" / "w")
    c = record["commitment"]
    assert _verifier(host, record) == (None, None)
    assert c["draw"]["coin"] and run.credit()["credited"]["macs"] == M * N * K
    run.verdicts[0] = {**run.verdicts[0], "credit": False, "reason": "withdrawn after the draw"}
    assert run.credit()["credited"]["macs"] == 0
    assert run.calls_root().hex() == c["calls_root"]
    assert RQ.draw_root(run.calls_root(), bytes.fromhex(c["draw"]["receipts"]), run.round_log).hex() == c["draw"]["root"]

    host = _host(tmp_path / "b", monkeypatch)
    _y, run, record = _serve(Arm(plain), host, tmp_path / "b" / "w")
    (v,) = run.verdicts.values()
    assert not v["credit"] and v["reason"].startswith("credit: z-records: q its z is not its"), v
    assert record["calls"][0]["tables"]["z"] is None and record["calls"][0]["served"] == M
    assert _verifier(host, record) == (None, None)
    run.verdicts[0] = {k: x for k, x in {**v, "credit": True}.items() if k != "reason"}
    assert run.credit()["credited"] == {"rows": M, "macs": M * N * K}
    assert run.calls_root().hex() == record["commitment"]["calls_root"]


def test_a_z_table_leaf_commits_a_units_z_and_y_as_one_row_which_its_z_alone_doesnt_open():
    """RT4-INT-33, the table: leaf u of a call's z table is unit u's 6,144-byte record (z, then y) and its 192-byte hm96
    salt, hashed as one 49,152-bit row (`audit.Hm96Domain`, which reads no schema: the binding separates trees). The unit's
    z alone, the 4,096 bytes a window Program unit outputs, at the same binding, salt and position is another leaf, so an
    opening of a unit's z output doesn't open its z-table leaf."""
    records = [bytes([u + 1]) * ZR.ZY_BYTES for u in range(2)]
    tree = RQ.hiding_rows(BINDING, 0, records, bytes(range(32)), RQ.TABLES_NONCE + PH.TABLE_NONCES["z"])
    values = tree._values
    assert [len(v) for v in values] == [ZR.ZY_BYTES + HM96_SALT] * 2
    domain = rows_domain(BINDING, 0, 2, HM96)
    assert MerkleTree(domain, dict(enumerate(values)), lambda _: ROW_SCHEMA).commitment.root == tree.commitment.root
    assert all(MerkleTree(domain, dict(enumerate(values)), lambda _, s=s: s).commitment.root == tree.commitment.root
               for s in G.SCHEMAS)
    z_only = {u: v[:4 * 32 * 32] + v[-HM96_SALT:] for u, v in enumerate(values)}
    assert all(domain.leaf(u, u, ROW_SCHEMA, z_only[u]) != domain.leaf(u, u, ROW_SCHEMA, values[u]) for u in range(2))
    assert MerkleTree(domain, z_only, lambda _: ROW_SCHEMA).commitment.root != tree.commitment.root


@pytest.mark.slow
def test_the_window_program_reads_no_z_table_and_its_units_output_z_and_a_receipt_and_no_y():
    """RT4-INT-33, the route: the served 32 × 32 tile's window Program at K = N = 64 reads ``a/64``, ``b/64``, the salt,
    the positions and the roots, and its unit's output is (z, receipt), z 32 × 32 s32 words, 4,096 bytes of a record's
    6,144: no parameter is the z table and no output is y. The real tile traces in about 35 s
    (`test_circuit_h1window`'s real-tile test); a stand-in tile has no output record of its own to show."""
    from verity.core.protocols.pouw.circuit import h1window as HW
    assert HW.OUTPUTS == ("z", "receipt") and HW.roles([K]) == (f"a/{K}", f"b/{K}", "salt", "position", "roots")
    prog = HW.program((("q", K, N),), (("q", 0),), HW.slot(M, PH.TILE), PH.TILE)
    (call,) = prog.fn.signature()[1].fields
    assert call.n == (ROWS // 32) * (N // 32)
    fields = dict(call.elem.fields)
    assert list(fields) == ["z", "receipt"]
    assert fields["z"].leaves == 8 * 4 * 32 * 32 == 8 * (ZR.ZY_BYTES - 2 * 32 * 32)
