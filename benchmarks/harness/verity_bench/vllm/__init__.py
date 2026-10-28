"""The shared overhead benchmark that uses the vLLM integration.

The public pieces are deliberately small: presets expand deterministic sweep
points, arms describe stock, protocol-option, or experimental installations,
and :mod:`runner` gates and interleaves them before :mod:`result` and
:mod:`table` report one result.
"""
