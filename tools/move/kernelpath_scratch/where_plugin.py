"""Scratch, never to land: a pytest plugin that prints, in the process that ran the tests, the files the tests' own imports
of `verity`, the adapter and PoUW's prover package came from, and the two computed paths."""
import os
import sys


def pytest_terminal_summary(terminalreporter) -> None:
    w = terminalreporter.write_line
    w("-- inside the pytest process, after the tests ran")
    w(f"cwd: {os.getcwd()}")
    w(f"PYTHONPATH entries: {len([p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p])}, of which under a "
      f"pyshim/: {len([p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if '/pyshim/' in p])}")
    for name in ("verity", "verity_vllm", "verity_vllm.commit.row_tables_device", "verity_vllm.protocols.pouw.device"):
        m = sys.modules.get(name)
        w(f"{name}: {getattr(m, '__file__', None) if m else 'not imported'}"
          + (f" | resolved: {os.path.realpath(m.__file__)}" if m and getattr(m, '__file__', None) else ""))
    for name, attr in (("verity_vllm.commit.row_tables_device", "CUDA_DIR"), ("verity_vllm.protocols.pouw.device", "HERE")):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, "prover"):
            w(f"{name}.prover: {m.prover.__name__} {m.prover.__file__}")
        if m is not None:
            d = getattr(m, attr)
            w(f"{name}.{attr}: {d} | is a directory: {d.is_dir()} | files: {len(list(d.iterdir())) if d.is_dir() else 0}")
    for name, attr in (("verity.hardware.cards", "CARDS"), ("verity.hardware.nodes", "NODES")):
        m = sys.modules.get(name)
        if m is not None:
            d = getattr(m, attr)
            w(f"{name}: {m.__file__}")
            w(f"{name}.{attr}: {d} | is a directory: {d.is_dir()} | sheets: {len(list(d.glob('*.toml')))}")
