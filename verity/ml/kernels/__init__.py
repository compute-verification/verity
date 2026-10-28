"""Registered fast code, by op and device (untrusted): each kernel is checked against its Definition.

``SHA512`` is ``sha512.cuh``, FIPS 180-4 SHA-512 on the device and the host (its round constants, IV and compression
forms), which kernels outside the package include by its path in the tree, and a build from an installed ``verity`` from
here."""
from pathlib import Path

SHA512 = Path(__file__).with_name("sha512.cuh")
