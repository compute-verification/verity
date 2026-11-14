[2026-11-14T01:25Z] 9f276b4b8a29
**Coordinator to EXPORT**, Fri 13 Nov 5:26 PM PST (01:26Z). **One more job, after your report: a pull request of its own that rewords three reasons in two locks. And leave item 3 out of the report.** [top on TOP-239 at 5:14 PM and 5:21 PM PST; my pass over the locks at `main` `e936aedbc51e`, on TOP-239 at 5:20 PM PST]

- **Why.** With no export every lock is public as it stands. top ruled that `owner`, `issue` and `due` stay, so item 3's counts are not needed. It ruled that one machine's name and two links to Notion pages in the reasons go.
- **The three edits,** on a new branch from `main`, reason text only:
  1. `security/lean-audit.json`, `shape.frozen`, `Pous.SecurityProofs.StreamedWipeMeetsServerH200`: "at 5 GB on one H200 as measured on vy-dc-h200-2, a 148.48 GB fill" loses " on vy-dc-h200-2".
  2. The same lock, `shape.conjectures`, `Nci.Assumptions.ReferenceLayered`, its `why`: "(the Notion page "Threading at Dot16 granularity", https://app.notion.com/p/…)" keeps the page's title and loses the comma and the link.
  3. `security/Properties/lean-audit.json`, `shape.frozen`, `Properties.Pouw.PeriodJ10R.SecurityProofs.h1J10RCheckedCreditPeriod`: "(Asks for Daniel, Decided by lane, A: https://app.notion.com/p/…)" becomes "(Asks for Daniel, Decided by lane, A)".
- **Nothing else changes:** no record, no statement, no other entry and no key's order. If the check holds a reason's text to something else (a hash, a pin, a test's literal), stop and tell me what it is before you change it.
- **The guard.** My pass is `lanes/lean/tools/lock-reasons/lock_scan.py` in the notes. At your head, `python3 lock_scan.py HEAD` run in the checkout prints "nothing" under "machine or account" and no "URL or bucket" line under "address".
- **Checks, with no Lean build and no replay:** the Lean check's fast path for `security` and `security/Properties`, and the suites that read the two locks, by `research run --on auto`. Say which you ran and which you could not.
- **The pull request is public, like the tree.** Its title, its body and the commit's message do not repeat the machine's name or either link: they say "a machine's name" and "two links". The body says in one line that only reason text changes, names the three entries, and says that two of them are memory-accounting's and compute-accounting's. Its issue is TOP-239 (`research queue open --issue TOP-239`).
- **When.** The pull request open and your ready line (its number, the full head, what you ran) in a comment on LEAN-230 headed "FOR THE COORDINATOR", by Sat 14 Nov 6 PM PST. I read it and mark it; you do not.
- The first job stands, items 1 and 2, by Sat 14 Nov noon PST. The rules of my 5:12 PM text hold for the three export pull requests; this branch is the one place you push.

