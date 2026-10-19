# circuits worker inbox

Not code. A running circuits cloud worker can't be messaged mid-turn, so its coordinator appends notes to
`inbox/<agent-id>.md` on this branch. At every hourly checkpoint a worker runs

    git fetch -q origin cursor/circuits-inbox-21f3 && git show origin/cursor/circuits-inbox-21f3:inbox/<agent-id>.md

and acts on anything newer than its last read (each note starts with a `[YYYY-MM-DDTHH:MMZ]` line).
