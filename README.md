Inbox branch for compute-accounting research lane (bc-ebc09a9a) cloud workers. Each worker reads inbox/<its bc id>.md at every hourly checkpoint:
git fetch -q origin cursor/ca-research-inbox-ea24 && git show origin/cursor/ca-research-inbox-ea24:inbox/<bc-id>.md
