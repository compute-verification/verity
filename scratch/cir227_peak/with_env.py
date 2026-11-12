"""with_env.py FILE CMD...: run CMD with the variables FILE holds (a JSON object) added to this process's."""
import json
import os
import sys

with open(sys.argv[1]) as f:
    os.environ.update(json.load(f))
os.execvp(sys.argv[2], sys.argv[2:])
