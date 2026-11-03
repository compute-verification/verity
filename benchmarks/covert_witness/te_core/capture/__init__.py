"""The training-graph capture: one training step of ordinary PyTorch code, forward, autograd backward and optimizer
update, as an ATen graph with shapes and no values (`trace`); the structure a score reads from it in MACs, each GEMM's
operand classes and role (updated, producing, neither), its token dims and the updated MACs upstream of it, φ, and the
narrowest cut between the step's gradient side and its new persistent values, T1's β (`structure`);
a run as the sum of its distinct step graphs (`run`); and the optional lowering onto Verity's Definitions (`lower`)."""
from te_core.capture.run import RunGraph, capture_run, schedule_updated
from te_core.capture.structure import CLASSES, Matmul, StepGraph
from te_core.capture.trace import CaptureRefused, capture_fn, capture_step, phase

__all__ = ["CLASSES", "CaptureRefused", "Matmul", "RunGraph", "StepGraph", "capture_fn", "capture_run", "capture_step", "phase",
           "schedule_updated"]
