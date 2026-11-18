import os
import stat

import pytest
from verity.commitments import salt_file as SF


def test_a_parameters_salts_read_back_by_name_whole_or_by_row_and_the_files_are_the_owners_alone(tmp_path):
    salts = {"weights/b@8": os.urandom(3 * SF.SALT_BYTES), "weights/a@4": os.urandom(5 * SF.SALT_BYTES)}
    d = SF.write(SF.beside(tmp_path / "weights-registration.json"), salts)
    assert d == tmp_path / "weights-registration.salts"
    assert SF.params(d) == {"weights/a@4": 5, "weights/b@8": 3}
    for name, s in salts.items():
        assert SF.read(d, name) == s
    s = salts["weights/a@4"]
    assert SF.read(d, "weights/a@4", [4, 0]) == s[4 * 192:5 * 192] + s[:192]
    assert stat.S_IMODE(d.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(f.stat().st_mode) == 0o600 for f in d.iterdir())


def test_a_rewrite_replaces_every_parameter_and_a_wrong_name_row_or_length_is_refused(tmp_path):
    d = tmp_path / "r.salts"
    SF.write(d, {"x": bytes(2 * 192), "y": bytes(192)})
    SF.write(d, {"z": b"\1" * 192})
    assert SF.params(d) == {"z": 1} and sorted(f.name for f in d.iterdir()) == ["0.bin", "index.json"]
    with pytest.raises(ValueError, match="no salts for 'x'"):
        SF.read(d, "x")
    with pytest.raises(ValueError, match="not row 1"):
        SF.read(d, "z", [1])
    with pytest.raises(ValueError, match="positive multiple of 192"):
        SF.write(d, {"w": bytes(191)})
