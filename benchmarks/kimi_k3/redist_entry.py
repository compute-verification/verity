"""`python3 redist_entry.py MANIFEST NAME`: the linux-x86_64 archive path and sha256 of one component of NVIDIA's
redistrib_<version>.json, as `RELATIVE_PATH SHA256`, for `toolkit.sh`."""
import json
import sys

entry = json.load(open(sys.argv[1]))[sys.argv[2]]["linux-x86_64"]
print(entry["relative_path"], entry["sha256"])
