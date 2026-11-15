"""The weight store a serving system's adapter holds (`weight_store`), on its own: its P2 store's package loaded from its
files, and the decode at use the store's own module builds.  The adapter's option over it is tested in the vLLM adapter
(`tests/protocols/test_pous_option.py`, `test_pous_refusals.py`)."""
from __future__ import annotations

import importlib
import sys

import pytest
from verity.protocols.resource_accounting.memory.pous.prover import weight_store as ws
from verity.protocols.resource_accounting.memory.pous.verifier import period
from verity.protocols.resource_accounting.memory.pous.verifier.schemes.p2 import P2Params

P2 = "p2-16448/v4"
SMALL = P2Params(period.W, period.C0, 8)


@pytest.fixture
def packages():
    """Three package names for `weight_store.load_package`, each out of ``sys.modules`` again after the test, with its modules."""
    names = ("pous_package_plain", "pous_package_with_init", "pous_package_broken")
    yield names
    for loaded in [m for m in sys.modules if m.split(".")[0] in names]:
        del sys.modules[loaded]


def test_the_stores_package_loads_from_its_files_with_or_without_an_init_and_never_half(tmp_path, packages):
    plain, with_init, broken = packages
    path = list(sys.path)
    for name in packages:
        (tmp_path / name).mkdir()
        (tmp_path / name / "leaf.py").write_text("VALUE = 5\n")
        (tmp_path / name / "reader.py").write_text(f"from {name}.leaf import VALUE\n")
    (tmp_path / with_init / "__init__.py").write_text("FROM_INIT = 1\n")
    (tmp_path / broken / "__init__.py").write_text("raise RuntimeError('half an init')\n")

    for name in (plain, with_init):
        ws.load_package(name, tmp_path / name)
        module = sys.modules[name]
        assert importlib.import_module(f"{name}.reader").VALUE == 5                    # its modules import one another by its name
        assert list(module.__path__) == [str(tmp_path / name)]
        ws.load_package(name, tmp_path / "nowhere")                                      # loaded once: a second call reads no file
        assert sys.modules[name] is module
    assert sys.modules[with_init].FROM_INIT == 1 and not hasattr(sys.modules[plain], "FROM_INIT")
    with pytest.raises(RuntimeError, match="half an init"):
        ws.load_package(broken, tmp_path / broken)
    assert broken not in sys.modules and sys.path == path


def test_the_decode_at_use_on_a_gpu_is_the_stores_own_module():
    """What `weight_store.decoder` builds on a GPU comes from the store's package alone, as far as a CPU reaches: the layout of no
    tensors at the registered point, refused at any other, and rows of another shape refused before the library loads."""
    torch = pytest.importorskip("torch")
    load = ws.p2_module("load")
    assert load.__name__ == "p2_store.load" and "p2_stage" not in sys.modules
    layout = load.StoreLayout(ws.geometry(P2), (), bytes(128))
    with pytest.raises(ValueError, match="the point libp2v1.so is built for"):
        load.StoreLayout(SMALL, (), bytes(128))
    with pytest.raises(ValueError, match=r"a contiguous \[n, 2056\] uint8 tensor"):
        load.EncodedRows(layout, torch.zeros(1, 8, dtype=torch.uint8), 0, "no-such-directory")
