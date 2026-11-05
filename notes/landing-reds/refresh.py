import time
from research.store.attempt import open_store
s = open_store()
t = time.time()
rep = s.refresh(attempts_only=True, since="r20261021-000000", jobs=32)
print({k: (v if k != "errors" else len(v)) for k, v in rep.items()}, round(time.time() - t), "s")
