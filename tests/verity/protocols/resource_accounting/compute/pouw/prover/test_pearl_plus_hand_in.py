"""Pearl+ asks for the workload's device run by its declaration (`Scheme.kernels`), and the adapter's attach answers it
(`verity_vllm.messages.attach_listed` calls `verity_torch.pouw.hand_in`), so a served run that lists Pearl+ alone
opens its epoch's device run with nothing else imported."""

from __future__ import annotations

from verity import hooks as H
from verity.protocols.compose import schemes as declarations
from verity.protocols.resource_accounting.compute.pouw.prover.serving import tensors as T


def test_attaching_pearl_plus_hands_its_device_run_in(tmp_path, monkeypatch):
    import verity_torch.pouw  # noqa: F401 -- already imported, so importing it again hands nothing in
    from verity_vllm import messages as M

    monkeypatch.setattr(T, "_pearl_c", None)
    M.attach_listed(M.Exchange(), [(declarations.get("pearl-plus"), {"coins": "in-process", "kernel": "stand-in"})],
                    private=tmp_path, process=H.ENGINE, feed=str(tmp_path / "feed"))
    assert hasattr(T.pearl_c_device(), "PearlCDeviceRun")


def test_a_scheme_that_runs_no_kernel_hands_nothing_in(tmp_path, monkeypatch):
    from verity_vllm import messages as M

    monkeypatch.setattr(T, "_pearl_c", None)
    M.attach_listed(M.Exchange(), [(declarations.get("inference-only"), {})], private=tmp_path, process=H.ENGINE,
                    feed=str(tmp_path / "feed"))
    assert T._pearl_c is None
