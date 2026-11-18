[2026-11-18T06:39Z] 4deeac46052d
From the lead, 10:41 PM PST, the first-stage keyed-salt places as another worker searched them (verify each): adapters/torch/verity_torch/pouw/h1/tables_device.py callers at 242, 287, 319 (docstring 5-8); serving/h1/requests.py 229 hiding_rows(key, nonce); serving/h1/registered.py 319 Salts(key, nonce, n) and 323 (None unless public); registered_device.py 291, 397, 403 (no row salts means keyed); serving/h1/served.py 406 row_salts default None (likely the fifth caller).

