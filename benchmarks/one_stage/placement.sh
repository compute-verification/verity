# The placement hold's one shell reader, sourced by served_verdicts.sh and served_composite.sh; its rule is
# verity_cli.settings.hold_placement's (test_served_verdicts holds the two to each other).
# Sourcing fails while the retired VERITY_HOLD_PLACEMENT is set, as the Python reader's ValueError does.
if [ -n "${VERITY_HOLD_PLACEMENT+set}" ]; then
  echo "VERITY_HOLD_PLACEMENT is retired: the hold is the default, and VERITY_LIFT_PLACEMENT_HOLD=1 lifts it; unset VERITY_HOLD_PLACEMENT" >&2
  return 2
fi
# true unless VERITY_LIFT_PLACEMENT_HOLD=1, which lifts the holds that rest only on where coins are flipped or a composite made
hold_placement() { [ "${VERITY_LIFT_PLACEMENT_HOLD:-}" != 1 ]; }
# the sampled-proofs piece's challenger is a development session, local or loopback (PP_DEVELOPMENT_UNDER_HOLD=1, its
# hooks'): under the hold its run is a development run (DEVELOPMENT=1); lifted it is simulated and DEVELOPMENT is as given
development_under_hold() { if [ "${PP_DEVELOPMENT_UNDER_HOLD:-}" = 1 ] && hold_placement; then export DEVELOPMENT=1; fi; }
