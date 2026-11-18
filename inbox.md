[2026-11-18T07:37Z] e8c9167656e9
From coordinator: if `research run --on auto` refuses your test job with "would run past ... admission end 2026-10-24" (LLOG hit it: the stops_at fallback when the node deadline cannot be read), retry once with a named node, e.g. `--on vy-nebius-cpu-2`; if still refused, run the two named files locally one at a time and return the exact refusal text.

[2026-11-18T07:47Z] 1a4f98d781b3
From coordinator, add to Job 6: the first-stage serving `Toolchain.arch` defaults to "89" (kernels found it; under verity/protocols/resource_accounting/compute/pouw, find it). Remove the default: each caller passes the architecture read from the run machine card (calibration/hardware card/architecture files via verity.device) or it refuses by name. Kernels did the same for the two Pearl-C builds at 3044b77b1d4 on interfaces; follow that shape. Own commit; return it.

