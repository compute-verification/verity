"""The commitments on a GPU (untrusted): the CUDA headers a kernel includes to hash and commit where its values are.

``SHA512`` is ``sha512.cuh``, FIPS 180-4 SHA-512 on the device and the host (its round constants, IV and compression
forms), and ``ROW_COMMIT`` is ``row_commit.cuh``, the row commitment on the device and the host over it: frame-v3-sha512's
leaf, pad and node and the hm96-sha512 leaf, from constants its builders make once a domain and a key. Kernels outside the
package include them by their paths in the tree, and a build from an installed ``verity`` from here."""
from pathlib import Path

SHA512 = Path(__file__).with_name("sha512.cuh")
ROW_COMMIT = Path(__file__).with_name("row_commit.cuh")
