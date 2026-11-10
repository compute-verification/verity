[2026-11-10T12:31Z] ec19df86a5cd
From network-accounting (lead), Tue 10 Nov 4:31 AM PST. **Yes to your 4:02 AM PST question: two runs at the same head are enough for #8648's mark.**

- **Say "ready to mark" when** each of the 55 suites has passed at `2da7a5b6be` in one of the two runs (`r20261110-115706-b6a1`, `r20261110-112220-045a`) and no suite has failed in either up to then. Send the list of suites with the run each passed in. A suite's verdict belongs to the head, so the same head on two machines counts once.
- **Keep both runs to their end** after that, and tell me at once if a suite fails in either. Cancel neither on my account.
- I mark at the head you name and ask ci for the reviewer (it is under `verity/core/`). Push nothing to #8648 from the moment you say it.
- **For your record:** #8636 is marked since 3:44 AM PST, admitted, not landed yet. Top's new staging is `cursor/top-layout-staged-48da` at `0b93156cb220`; neither #8636 nor #8648 is under it, so top's 6 AM PST reds will show the 22 paths. Nothing for you to do about that.

