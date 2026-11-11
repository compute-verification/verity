[2026-11-11T06:48Z] 46fd80e4bc6e
From onecalc (bc-2f024cb4-e0ea-5389-84f4-6c6d22e0f980, TRN-101, branch cursor/training-one-calculator-f980 off 95239b187833; nothing pushed yet, I send you the head when it is).

The one module stays training_slowdown.calculator. calculator.bound(c_gates, k_bits, work) keeps its name, its arguments and work's keys {N, O, beta_w, beta_x, b_y, F, C_gates}; its "slowdown" is the solved count, which becomes the figure the grade prints. formalism.he_of_rows and formalism.he_figure keep their names and signatures and stay in formalism.py in PR 1.

What changes under you: he_figure will choose M by the solved count, and its "rho" will be that count times c/|C| (today it is the closed form), so at the M it returns, calculator.bound(...)["slowdown"] == max(1, he_figure(...)["rho"]). formalism.he_rho, BETA_K, beta_k and SHARE go: beta(k) and F are calculator.beta(k) and calculator.share(3).

Import nothing of verity/protocols/inference_only/rho yourself, by name or by path: a test will refuse that in every module and test of benchmarks/training_slowdown but calculator.py. If you want the work dict the grade builds from a record (grade.rho_of does it inline today) as a function of its own, say so and I add it.

