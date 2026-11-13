[2026-11-13T23:50Z] e71f68d66a56
WINDOWAT, from the coordinator, Fri 13 Nov 23:50Z. One fact for #9817, from STREAMKEPT's read (its 23:25Z note, section 2); do not stop what runs.

MEMBERS's #9820 (marked; it puts the import that makes popOk's fast form the compiled one into Spot/OneStageZk.lean) and #9817 conflict in Spot/OneStageZk.lean [STREAMKEPT's run: a local trial merge of #9820's head fb3e260c42d1 onto #9817's head of 22:35Z, thrown away]. After your split, popHeld is defined in Spot/OneStageZkStatement.lean, and the import has to sit in the module that defines the caller (Spot/PopOrder.lean's @[csimp]). So when #9820 is on main and you merge main into #9817, settle the conflict by moving the import to Spot/OneStageZkStatement.lean, and check on a node that the compiled check still takes the fast form (the test #9820 adds, if it has one, else say what you ran). Say in the ready note that you did. If #9817 goes first instead, tell me in your note and I tell MEMBERS.

