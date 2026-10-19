---
name: asking-daniel
description: >
  How to ask Daniel for a decision in Verity: one short Notion page per ask in the Asks for Daniel database, written to a
  fixed template with LaTeX for the math, plus a card in #ask-daniel that notifies him and links the page. Covers what
  needs Daniel (keys, access, and what a guarantee says or rests on), the page template, `research msg ask-daniel --page
  FILE.md` (or a page made with the Notion MCP and `--link`), iterating on his page comments, `research msg ask-daniel
  pages` to see his decisions and comments, recording a decision (`resolve`, the page's Status, the friction skill's
  rulings), and when a default may apply. Use when a lane needs Daniel's decision, writes or revises an ask, is woken by
  a decision or a comment on one, or asks how #ask-daniel cards, the Asks database or `ask-daniel` work.
---

# Asking Daniel

An ask is a **page** in the [Asks for Daniel](https://app.notion.com/p/914ab9399fe44f79ae19ff05db87e1fa) Notion database,
where Daniel reads it, comments to iterate, and decides; and a **card** in `#ask-daniel` that notifies him and links the page
(Daniel, 18 Oct: the cards had too little context, and math doesn't read in Slack). The registry's `#ask-daniel` entry holds the
database's URL and data source (`pages`). A coordinator asks directly, one question per page; a worker asks its coordinator.

## Before asking

Only keys and claims need him (8 Oct, narrowed 13 Oct): access and keys, and what a guarantee says or rests on. Anything else,
spend included, is its owner's call, and his standing rulings stand until he reverses them. Check the recorded rulings first
("Rulings so far" in `.agents/skills/friction/SKILL.md`, `AGENTS.md`, the skills): if one covers the case, follow it.

A decision that waits in a note or a thread waits forever: every ask gets a card, which mentions him (1 Oct).

## Write the page

Write it to this template, in Notion-flavored Markdown: inline math as ``$`\gamma \le 0.01`$`` (the backticks inside the
dollars are required), display math in `$$` blocks, tables in `<table header-row="true">`, no Slack mrkdwn. `research msg ask-daniel --page` refuses a
page that lacks a section or reorders them.

~~~markdown
<callout icon="❓" color="gray_bg">
	**Question.** One or two sentences.
	**Recommendation.** The option, and a one-line reason.
	**If you don't answer.** "<the default option> applies at <time PDT, day>." or, blocking: "Nothing happens; <what waits>."
</callout>
## The problem
Two to four short paragraphs: what is being decided, why now, what goes wrong without a decision.
## Background
What a reader needs to judge the options: the current statement in LaTeX, numbers with units and their source, what is
proved and what is assumed.
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

- **Plain English first, then the math.** Define every term and symbol on first use. Daniel knows the field, not this week's
  codenames: explain or drop a red-team round's name, a row number, a worker's name.
- **Short:** about 400 to 900 words, one screen of options. Detail goes behind a link in Sources.
- **The page is current.** A correction or new data edits the page in place (the numbers, the options, the recommendation)
  and adds a dated line under "Updates since the card". Don't stack addenda in the card's thread.
- **Exact statements.** A claim ask shows the statement or premise as it would enter the lock, in LaTeX, with its Lean name.

## Post it

With `$NOTION_API_KEY` set (a Notion connection's token, shared on the database), one command makes the page, posts the card
linking it, and names the card on the page:

~~~bash
research msg ask-daniel --as memory-accounting --kind default --area 'Claim or assumption' \
  --question 'Accept SteeredLayout as a premise of G?' --why 'G row 6 rests on it; the red team found it minimal.' \
  --option yes='Yes, as a premise' --option hold='Hold until ScheduleDeterministic is checked' --recommend yes --default yes \
  --deadline '2026-10-19T20:00:00Z' --page ask.md
# -> "page <url>" on stderr, then "<id> <ts> <permalink>"; subscribe_slack_thread(channelId=<#ask-daniel id>, threadTs=<ts>)
~~~

Without the token, make the page with the Notion MCP: `notion-create-pages` with parent
`{"data_source_id": "9c204dab-e175-407f-b18d-bc3b3bddec61"}`, the properties below and the template as content. Then post the
same command with `--link <page url>` in place of `--page`/`--area`, and set the page's Card ID and Slack card from its output.
`--link` takes only a Notion page.

The card's fields are its question in one line, why it matters in a sentence or two (the page has the rest), 2 to 5 options
with one recommended, the default, and a deadline in Pacific time at most 7 days out. **Blocking** is for access and keys:
nothing happens until he decides. **Default** applies at the deadline unless he comments or decides.

The page's properties: Ask (the question), Status (Waiting on Daniel), Lane (your handle), Kind, Area (Claim or assumption,
Access or keys, Machines or spend, Process, Research direction), Recommendation, If no answer, Deadline, Lane checked (yes,
once you've read the page as Daniel will), Asked by (your agent's URL), Card ID, Slack card.

## Iterate and record

- **See what changed:** `research msg ask-daniel pages --as <you>` lists your asks' pages edited in the last day (`--hours H`)
  with Status, Decision and the newest comment. Run it on every wake, and keep a 30-minute timer while you have an open ask.
  Top relays decisions and comments to your inbox too, but don't wait on it.
- **A comment** from Daniel: revise the page, reply to the comment, add a line under "Updates since the card", and set Status
  back to Waiting on Daniel if he set Revising. Start a reply with your handle (`@memory-accounting:`): the Notion MCP writes as
  Daniel's own account.
- **A decision** is the page's Status: **Approved** takes the recommendation; **Decided otherwise** takes what he wrote in
  Decision. A button or a thread reply on the card counts the same: set the page's Status to match. Record it on the card with
  `research msg ask-daniel resolve <id> --choice <value> --note '<page url>'` (only the card's asker may), and as a ruling for
  its class of case, where the next asker will read it (friction skill §3). `status` or `resolve` on a decided card rings
  @top's inbox once with the outcome, for Daniel's decisions log: file no TOP issue just to log it (COMM-34).
- **A default** applies at the deadline only if he hasn't commented since the page was posted and Status isn't Revising: a
  comment pauses it. When it applies, set Status to Default applied. Withdrawn is for an ask that events overtook.

~~~bash
research msg ask-daniel status <id> [--wait]                  # the card: "status choice source decided_at"; exit 0 once decided
                                                              # --wait polls 20 min at most: still pending, it exits 79; end your turn
research msg ask-daniel pages [--as HANDLE] [--hours H] [--json]
research msg ask-daniel resolve <id> --choice <value> --note '<page url or his reply>'
~~~

If the card route fails, post the same question with `research msg send --as <you> --to @daniel --text '... <page url>'`,
which mentions him too.
