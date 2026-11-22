[2026-11-22T18:18Z] 587878642ef3
FROM top, Sun 22 Nov, about 10:25 AM PST. One addition to your brief, from the team's lead: he wants to use VERCEL BOARD as well. Add a section g to your answers and allow yourself 20 more minutes and 200 more words.

What I already read in eve's own docs (`node_modules/eve/docs/channels/board.mdx`): Vercel Board is a team's message board for people and agents (channels, threads, labels, tasks) with two-way sync to Slack, GitHub and Linear; `@eve/board` makes an eve agent a participant: it creates WATCHES (a channel or a subtree, narrowed by labels, author or words) that wake it in the thread; it can work a TASK QUEUE (a task is a thread with a state the board keeps; one holder at a time; a claim of an hour that eve renews; `complete_task`, `block_task`, `fail_task`); it can schedule follow-ups; the deployment reaches the board with its Vercel OIDC token; access is narrowed by grants.

g. VERCEL BOARD. Read that page whole, the package `@eve/board` (npm pack it; its README and types), and Vercel's own pages (https://vercel.com/docs/board, /docs/board/access, /docs/board/concepts/tasks, and whatever they link for an API). Answer with file, line or URL:
  1. What Board IS and its state (beta or generally available, which plans, any price, any limit on messages, tasks or watchers).
  2. Its API for a deployment that is NOT an eve agent: can any Vercel project post, read, register a watcher and claim tasks with its OIDC token (an HTTP API, an SDK), or only through `@eve/board`? This decides whether the team's existing prototype could use Board without eve.
  3. The two-way sync with Slack, GitHub and Linear: what is synced in each direction (messages, threads, issues, pull requests, labels, tasks), and what a person in Slack or Linear sees and can do.
  4. How a watch wakes a deployment (push or poll; what happens to a match while a turn runs; delivery guarantees), and the task claim's rules (lease, renewal, what happens when a holder dies).
  5. Against what the team runs today, one line each on what Board would replace: Slack channels as the coordination surface with 'doorbell' pings to a lead's inbox; Linear issues as work items; a watch process that wakes agents; hourly timers; a 'claim' convention for who works on what; a goals page with owners and due dates.
  6. What Board does NOT do that the team would still need.

