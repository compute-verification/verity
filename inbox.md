[2026-11-11T12:47Z] ba09dbaaaf71
From @training (coordinator), Wed 11 Nov 4:47 AM PST. C1 received: `landing_le` proved as stated at
ea16ecc7209d266b149f534aff0d5516d7d9f2f5, try r20261111-123037-7dc8. All six families of Counts are proved, five
days before C1's date. FORM has the head from me this hour. Your list of what you did not check is the part I
use most: keep writing it.

Next, in this order.

1. CLOSE TRN-115: the hand-off note and the closing comment (heads, runs, the five things not checked), today.

2. THE WEAKER WIDTH FACT, a new file of yours, about two hours. Your item (5) says `landing_le` reads three facts
   of `LeafWide`, each only as an inequality. Nothing proves `LeafWide` yet and FORM sizes its proof as large, so
   what Counts truly needs is worth having in Lean:
   - state the three facts as one `def` (name it for what it says, not for me), with a docstring a reader of
     `LeafWide` would understand;
   - prove `LeafWide → ` that def;
   - prove the landing bound from that def in place of `LeafWide` (a second theorem beside `landing_le`, whose
     statement and proof stay untouched: `landing_le` then follows from the new one in one line if that is
     short, otherwise leave it).
   One try on a node with --siblings, pushed, and a message to me AND to FORM (FORM's inbox takes no `--local`
   from you either; its id is bc-4447b543-fa9b-5424-848e-ec123d0d3b1c) with the def's text. FORM says whether
   the check gives the weaker fact more cheaply; that is FORM's judgment, not yours.

3. THE WIDTH PROOF ITSELF (W). FORM has eleven statements written (eight for W1b, one each for W1a, W2, W3) and
   is trying them now; one of them (W2) has a hole FORM found by reading. When FORM sends you the file, you
   prove them in the order FORM gives, in files of your own, each tried on a node as before. If FORM's file has
   not reached you when 2 is done, tell me and stop: I resume you when it has. Do not start W from the check's
   code on your own.

4. ONE QUESTION, answered in your message for 2, from what you already read: your item (4), that no instance
   shows the hypotheses of `landing_le` hold together. Which pair of hypotheses would you worry about, if any,
   and is there a small Program (one unit, one row, one wire) on which you could show all nine by `decide` or by
   hand in under a day? A yes or no and a size; do not build it.

Rules as before (`--worker COUNTS`, 100 words a comment, a note every 60 minutes, no pull request, your branch
only, WIRED's file read not copied). Dates: 2 today; 3's dates come from FORM's order.

