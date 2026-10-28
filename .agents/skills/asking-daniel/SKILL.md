---
name: asking-daniel
description: >
  How to ask Daniel for a decision in Verity: one short Notion page per ask in the Asks for Daniel database, written to a
  fixed template with LaTeX for the math, plus a card in #ask-daniel that notifies him and links the page. Covers what
  needs Daniel (what a guarantee says or rests on, and what only his own accounts can do), the page template, `research msg ask-daniel --page
  FILE.md` (or a page made with the Notion MCP and `--link`), iterating on his page comments, `research msg ask-daniel
  pages` to see his decisions and comments, recording a decision (`resolve`, the page's Status, the friction skill's
  rulings), and when a default may apply. Use when a lane needs Daniel's decision, writes or revises an ask, is woken by
  a decision or a comment on one, or asks how #ask-daniel cards, the Asks database or `ask-daniel` work.
---

# Asking Daniel

An ask is a **page** in the Asks for Daniel Notion database (Notion's search finds it by that title),
where Daniel reads, comments and decides, and a **card** in `#ask-daniel` that mentions him and links the page (the
registry's `#ask-daniel` `pages` holds the database's URL and data source). A coordinator asks, one question a page; a worker asks its coordinator. Every ask gets
a card: a decision waiting in a note or a thread waits forever.

## Before asking

Only claims and his own clicks need him (the team skill's "Working principles"): keys and access between our own
machines and accounts are the lanes' call. A recorded ruling that covers the case (AGENTS.md, the
skills, the friction skill's "Standing rulings") stands until he reverses it. A question the system already answers isn't
an ask (which circuit a templated unit's theorem names: the one its template unrolls to).

An ask that isn't a click only he can make goes on its page decided: Status *Decided by lane*, Choose your pick, Decision
"Provisionally accepted (standing rule): <option>. Re-check: <why it holds>". Resolve its card with
that option and `--provisional` (implied by a `--note` starting "Provisionally accepted"), so it says your lane decided,
and act on it; he reviews these in the "Provisional, to review" view and can reverse any. A page that waits on him leaves
Choose empty: a Choose reads as his pick.

## Write the page

Notion-flavored Markdown: inline math ``$`\gamma \le 0.01`$`` (the inner backticks are required), display math in `$$`
blocks, tables in `<table header-row="true">`, no Slack mrkdwn. `ask-daniel --page` refuses a missing or reordered
section and adds the reply callout if absent; a page made with the Notion MCP needs it too.

~~~markdown
<callout icon="❓" color="gray_bg">
	**Question.** One or two sentences.
	**Recommendation.** The option, and a one-line reason.
	**If you don't answer.** "<the default option> applies at <time PDT, day>." or, blocking: "Nothing happens; <what waits>."
</callout>
<callout icon="✍️" color="yellow_bg">
	**Your reply.** Write below this line: a decision, a question or a change. Top relays it to the lane, which answers under your text starting with its handle.
</callout>
## The problem
Two to four short paragraphs: what is being decided, why now, what goes wrong without a decision.
## Background
What a reader needs to judge the options: the current statement in LaTeX, numbers with units and their source, what is
proved and what is assumed.
## The design the rules want
The design that meets every design rule in AGENTS.md (everything zero-knowledge, one hash, live coins, no form ahead of its
proof), priced even when it's the expensive one, and which option below it is. Or the rule that makes it impossible.
## Options
### A. <label> (recommended)
What changes (the exact new statement or premise in LaTeX), what it costs, what it unblocks, what risk it takes on,
how reversible it is.
### B. <label>
...
## Why the lane recommends A
The argument, and what would change the recommendation.
## What happens next
- **If you approve A:** ...
- **If you choose B:** ...
- **If you say nothing:** ...
## Updates since the card
Dated lines, newest last, or "None.".
## Sources
Links: notes (GitHub research-notes URLs), PRs, branches, Lean declarations, runs, Linear.
~~~

Plain English first, then the math, every term and symbol defined on first use: Daniel knows the field, not this week's
codenames (a red-team round, a row number, a worker). About 400 to 900 words, one screen of options, detail behind links.
A correction or new data edits the page in place, with a dated line under "Updates since the card", never an addendum in
the thread. A claim ask shows the statement or premise as it would enter the lock, in LaTeX, with its Lean name. Price the
design the rules want, not just tweaks to today's.

## Post it

With `$NOTION_API_KEY` (a Notion connection's token, shared on the database), one command makes the page, posts the card
and names the card on the page:

~~~bash
research msg ask-daniel --as memory-accounting --kind default --area 'Claim or assumption' \
  --question 'Accept SteeredLayout as a premise of G?' --why 'G row 6 rests on it; the red team found it minimal.' \
  --option yes='Yes, as a premise' --option hold='Hold until ScheduleDeterministic is checked' --recommend yes --default yes \
  --deadline '2026-10-19T20:00:00Z' --page ask.md
# -> "page <url>" on stderr, then "<id> <ts> <permalink>"; subscribe_slack_thread(channelId=<#ask-daniel id>, threadTs=<ts>)
~~~

Without it, make the page with the Notion MCP (`notion-create-pages`, parent `{"data_source_id":
"9c204dab-e175-407f-b18d-bc3b3bddec61"}`, the template as content, and the properties Ask, Status *Waiting on Daniel*,
Lane (your handle), Kind, Area, Recommendation, If no answer, Deadline, Lane checked (yes once you've read it as Daniel will), Asked by
(your agent's URL)), run the same command with `--link <page url>` for `--page`/`--area`, and set the page's Card ID and
Slack card from its output. If the card route fails, `research msg send --as <you> --to @daniel --text '... <page url>'`
mentions him too.

The card's question is one line and its why a sentence or two. **Blocking** is for a click only he can make: nothing
happens until he makes it. **Default** applies at the deadline unless he comments or decides.

## Iterate and record

- **Every wake**, `ask-daniel pages --as <you>` lists your asks' pages edited in the last day (`--hours H`) with Status,
  Decision and newest comment; keep a 30-minute timer while an ask is open. Top relays these too; don't wait on it.
- **A comment**, or his text in "Your reply": revise the page, reply under his text starting with your handle
  (`@memory-accounting:`; the Notion MCP writes as him), add an "Updates since the card" line, and set Status back to
  Waiting on Daniel if he set Revising.
- **A decision** is the Status: **Approved** takes the recommendation, **Decided otherwise** his Decision text; a button
  or thread reply on the card counts the same, and you set Status to match. Record it with `resolve` (only the asker may)
  and as a ruling for its class of case where the next asker reads it (friction skill §3). `status` or `resolve` on a
  decided card rings @top once for Daniel's decisions log: file no issue to log it.
- **A default** applies at the deadline only if he hasn't commented since posting and Status isn't Revising; then set
  Default applied. Withdrawn is for an ask that events overtook.

~~~bash
research msg ask-daniel status <id> [--wait]                  # the card: "status choice source decided_at"; exit 0 once decided
research msg ask-daniel pages [--as HANDLE] [--hours H] [--json]
research msg ask-daniel resolve <id> --choice <value> --note '<page url or his reply>'
~~~
