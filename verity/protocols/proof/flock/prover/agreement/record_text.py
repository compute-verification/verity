"""``verity_flock.record_text``, loaded by path for the agreement scripts, which run where the package may not be
installed and read no other file of it (`check.AGREEMENT_READS`)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "agreement_record_text", Path(__file__).resolve().parent.parent / "python" / "verity_flock" / "record_text.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
session_text = _module.session_text
