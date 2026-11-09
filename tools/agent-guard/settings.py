"""The agent guard's settings registry: each environment variable the inbox and its hook share is read here, once.

Standard library only, and read at import: every hook run and every `inbox.py` command is a fresh process. The hooks run
this directory's files side by side (`boot.sh` and `hook.sh` copy it whole), so a sibling `import settings` finds it.
"""
from __future__ import annotations

import os

#: where queued messages wait, as `<conversation_id>.md`
INBOX = os.environ.get("AGENT_INBOX", "/tmp/agent-inbox")
#: seconds between the hook's background `inbox.py pull` for one agent; `off` stops it
INBOX_POLL_S = os.environ.get("AGENT_INBOX_POLL_S", "30")
