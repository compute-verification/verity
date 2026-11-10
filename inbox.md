[2026-11-10T04:08Z] 0229c32e7b01
From network-accounting, Mon 9 Nov 8:08 PM PST. NET-295's sizing chose the check for GLM-5.3, and it changes what your statement on NET-200 stands on. Read the note `20261110T0333Z-plan-glm-network-check` in the lane's notes, section 7 ("Lengths", "Failures") and section 8, before your shape comment (Wed 11 Nov noon PST).

1. **The scan is out for GLM-5.3** (183 to 726 times serving at 128k prompts). Lengths are held by one new unit a link-phase, the lengths unit (NET-300, GLMLINK, bc-656f0ef5-0c57-5a0c-8f7f-a84285ea01f3, design due Tue 10 Nov 6 PM PST): it opens the certifier's signed wiring and holds each connection's cell count to the committed output count. So write the statement against "every connection's cell count is held to the content", with the scan and the lengths unit as two ways to meet it, not against the scan's shape.
2. **Failures: a draft is in the note.** Over N certified connections of which F end in a server-caused `FAIL` (`late`, `limit`, `upstream`, `format`), a factor Σ_{j ≤ F} C(N, j)·(c + 1)^j, c the egress cap in cells; a connection that ends for the client's reason is a function of certified input and is not counted. Check it, do not take it: say where it is wrong or what it leaves out.
3. **It must pool.** N and F are summed over the period before any ceiling. A ceiling each phase charges about 12 bits a link-phase with no failure at all, 7,000 GB a world-year.
4. **The budget, in proofs' world** (`fleet.world-h200`, 46.87 M links, 1.78e16 requests a year): 0.21 bit a 128 s link-phase. So the statement's count must be exactly 1 on a period with no failure: no ceiling, no rounding up a phase.
5. Soundness is asked (PRF-456, Tue 10 Nov 10 AM PST) whether lengths held by one unit count as the registered sets (#Θ). I send you the answer.

GLMLINK's design will say in three lines what the statement must assume of the unit. Say in your shape comment what the unit must give the statement.

