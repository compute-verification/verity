"""The hardware: the registration log (``registration``) and the Lean card tables' writer (``lean``). The card files and the node files are the repository's ``hardware/cards/`` and ``hardware/nodes/``, and the
Site and the readers of those files are ``verity/device.py``. This file holds nothing, so importing one of them runs
none of another's code: import from the owning module (``from verity.device import load_card``)."""
