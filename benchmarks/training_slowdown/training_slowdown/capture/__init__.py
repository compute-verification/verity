"""The training-graph capture: one training step of ordinary PyTorch code, forward, autograd backward and optimizer
update, as an ATen graph with shapes and no values (`trace`); the structure a score reads from it in MACs, each GEMM's
operand classes and role (updated, producing, neither), its token dims and the updated MACs upstream of it, φ, and the
narrowest cut between the step's gradient side and its new persistent values, T1's β (`structure`)."""
from training_slowdown.capture.structure import CLASSES, Matmul, StepGraph
from training_slowdown.capture.trace import CaptureRefused, capture_step, capture_steps

__all__ = ["CLASSES", "CaptureRefused", "Matmul", "StepGraph", "capture_step", "capture_steps"]
