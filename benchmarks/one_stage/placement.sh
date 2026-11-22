# The placement hold's one shell reader, sourced by served_verdicts.sh and served_composite.sh; its rule is
# verity_cli.settings.hold_placement's (test_served_verdicts holds the two to each other).
# true under VERITY_HOLD_PLACEMENT=1, which restores the holds that rest only on where coins are flipped or a composite made
hold_placement() { [ "${VERITY_HOLD_PLACEMENT:-}" = 1 ]; }
# the sampled-proofs piece's challenger is a development session, local or loopback (PP_DEVELOPMENT_UNDER_HOLD=1, its
# hooks'): under the hold its run is a development run (DEVELOPMENT=1); unheld it is simulated and DEVELOPMENT is as given
development_under_hold() { if [ "${PP_DEVELOPMENT_UNDER_HOLD:-}" = 1 ] && hold_placement; then export DEVELOPMENT=1; fi; }
