module

public import NetworkCheck.Period
meta import NetworkCheck.Period

/-!
# The period's check on schedules with connection rows, at a schedule

`plainSchedule`, a schedule whose cells hold an id request's head and first prompt id, and the guards that hold the
period's check (`NetworkCheck.Period`) to its Python mirror there: the schedule's digest with lengths rows, content
rows, a response cap and request words, what the parse reads and refuses of them, and the check's refusal of a
schedule with connection rows.
-/

@[expose] public section

namespace NetworkCheck.Period

open Lean (Json toJson)

/-- A schedule whose cells hold an id request's head and first prompt id, without lengths rows
(`WardenParams(cell_bytes=512, slots=2, bucket_ms=100, delay=5, lanes=2, queue=2, window=16, ingress_cells=4,
ingress_buckets=3, egress_cells=9, every=2)`). -/
def plainSchedule : Schedule :=
  { bucketMs := 100, cellBytes := 512, delay := 5, egressCells := 9, every := 2, ingressBuckets := 3,
    ingressCells := 4, lanes := 2, queue := 2, slots := 2, window := 16 }

-- `WardenParams.digest` of `plainSchedule`, with lengths rows of form 1 and of form 2, with form 1's classes
-- (2, 2, 1) and (4, 8, 16), with form 2's classes (2, 2) and (4, 8), with those under `max_tokens_cap` 4, and with
-- request words beside them: the lengths rows' form, the content rows' classes and the request words are in the
-- digest only of a schedule that sets them
#guard (plainSchedule.digest.map Flock.hexOf).toOption == some
  "b064f2c3e90f5059b7393bcbca02b7b94d514041946423a253fa1cca6de11ee4\
    1898e2e0368e1a03d9e5f8a910fc91ac2f8154924ccd7f669f8b0f4d99964808"
#guard (({ plainSchedule with lengths := some 1 } : Schedule).digest.map Flock.hexOf).toOption == some
  "98280d93904a794e8b443e0c2bb917e0ce0fbb20462a8e99f42fdbe06b0b8510\
    bc65791cc41703428a43976695a2fb75330780df7434190990751efd1202bd13"
#guard (({ plainSchedule with lengths := some 2 } : Schedule).digest.map Flock.hexOf).toOption == some
  "804c6d03b6666a89155a1c4f41c31c7d02c90978cf6d5a28b6a41bbe54131d2c\
    6ea7c345d593c1b204775de7b4bab8f1b264bfed3ec60bb9018d3bee4243335a"
#guard (({ plainSchedule with lengths := some 1, content := some #[#[2, 2, 1], #[4, 8, 16]] } : Schedule).digest.map
  Flock.hexOf).toOption == some
  "f5a916f78cead6daa18dd2913f37aeb53acf2e2557bceb70a33326d786927891\
    4bb9cc7e1c9737ba575ebb86bb5632d1d6d95cfabea982eda3af34081ac1384a"
#guard (({ plainSchedule with lengths := some 2, content := some #[#[2, 2], #[4, 8]] } : Schedule).digest.map
  Flock.hexOf).toOption == some
  "d2217081ba58125d65abf5eef16979e867596c702dea76e3f3d55bb698cb490c\
    dee255fe7b76238dc4b4147b60191d30179391ecca9e29aed228723e822f9811"
#guard (({ plainSchedule with lengths := some 2, content := some #[#[2, 2], #[4, 8]], maxTokensCap := some 4 } :
  Schedule).digest.map Flock.hexOf).toOption == some
  "5c079664862cfe253a9794957b6af18c8d0cd41adf1e2608e624663cde1038f8\
    3f792222b44885914fde2364290676e742cf28c1f93c55f093953fc7f1be6d3e"
#guard (({ plainSchedule with
    lengths := some 2, content := some #[#[2, 2], #[4, 8]], maxTokensCap := some 4, requestWords := true } :
  Schedule).digest.map Flock.hexOf).toOption == some
  "213b8b63d87459e4b76591beab56ea7b2ac6b9c888566ca4cd43b0d511584459\
    b84624493f8e18377327c4402ab301982b8fda184864fb9f2879502707781b47"
-- a slot stride's releases at `plainSchedule` and at its strides of 1, 4, 8 and 16 buckets, at a stride of 16 with
-- one ingress bucket, and at 5 lanes, a stride of 8 and 5 ingress buckets; an execution phase's releases at
-- `plainSchedule`, with one ingress bucket, at a delay of 12, and at 5 lanes and 5 ingress buckets; and a
-- connection's span, at a delay of 5 and of 12 (`WardenParams.stride_releases`, `phase_releases` and `span`)
#guard
  let s := plainSchedule
  s.strideReleases == 2 && ({ s with every := 1 } : Schedule).strideReleases == 2 &&
    ({ s with every := 4 } : Schedule).strideReleases == 2 && ({ s with every := 8 } : Schedule).strideReleases == 4 &&
    ({ s with every := 16 } : Schedule).strideReleases == 8 &&
    ({ s with every := 16, ingressBuckets := 1 } : Schedule).strideReleases == 6 &&
    ({ s with every := 8, lanes := 5, ingressBuckets := 5 } : Schedule).strideReleases == 15 &&
    s.executionPhaseReleases == 6 && ({ s with ingressBuckets := 1 } : Schedule).executionPhaseReleases == 6 &&
    ({ s with delay := 12 } : Schedule).executionPhaseReleases == 4 &&
    ({ s with lanes := 5, ingressBuckets := 5 } : Schedule).executionPhaseReleases == 20 &&
    s.span == 9 && ({ s with delay := 12 } : Schedule).span == 16
-- a schedule's keys are in the order an object lists them, the optional ones only where it sets them
#guard scheduleKeysWith (fun _ => false) == scheduleKeys && (scheduleKeysWith fun _ => true).length == 15 &&
  keysOf (Json.mkObj ((scheduleKeysWith fun _ => true).map (·, Json.null))) == scheduleKeysWith fun _ => true
-- an id request's body after `max_tokens`: `temperature` and `top_p` in 8 characters each, `seed` in 10, no stop
-- strings, streamed, `model` and `user` in 64 each, then the prompt's key and its first id, `[` and 10 characters
#guard maxTokensEnd + (",\"temperature\":".length + 8) + (",\"top_p\":".length + 8) + (",\"seed\":".length + 10) +
  ",\"stop\":[],\"stream\":true".length + (",\"model\":".length + 64) + (",\"user\":".length + 64) +
  ",\"prompt\":".length + (1 + 10) == firstPromptEnd
-- a lengths row's slots, a response row's ids and a schedule's classes, each the most its bound gives
#guard mostSlots == (262144 - 2 * 16) / 96 && mostClasses == (2 ^ 16 - 3) / 3 &&
  mostClassesWords == (2 ^ 16 - 3) / 4 && classSize mostIds && mostIds ≤ (262144 - 16) / 32 - 1 &&
  (262144 - 16) / 32 - 1 < 2 * mostIds
-- the parse reads `plainSchedule` with lengths rows of either form and with content rows of the form's classes, and
-- refuses a third form, lengths rows where ingress ends after the delay, a connection outlasts the window or a cell
-- holds no first prompt id, content rows without lengths rows, no class, a class of the other form's numbers, a size
-- that is no power of two, more ids than a response row has, more prompt ids than a request's bytes, and more rows
-- than the lengths table's 16 slots
#guard
  let n := fun (x : Nat) => toJson x
  let j := fun (delay cell : Nat) (more : List (String × Json)) => Json.mkObj ([("bucket_ms", n 100),
    ("cell_bytes", n cell), ("delay", n delay), ("egress_cells", n 9), ("every", n 2), ("ingress_buckets", n 3),
    ("ingress_cells", n 4), ("lanes", n 2), ("queue", n 2), ("slots", n 2), ("window", n 16)] ++ more)
  let reads := fun (more : List (String × Json)) => (scheduleOf (j 5 512 more)).toOption.isSome
  let classes := fun (cs : Array (Array Nat)) => ("content", toJson cs)
  let form := fun (f : Nat) => ("lengths", n f)
  reads [] && reads [form 1] && reads [form 2] && reads [form 1, classes #[#[2, 2, 1], #[4, 8, 16]]] &&
    reads [form 2, classes #[#[2, 2], #[4, 4096]]] && reads [form 2, classes #[#[1024, 2]], ("max_tokens_cap", n 4)] &&
    (scheduleOf (j 5 512 [form 2, classes #[#[2, 2]]])).toOption.map (·.content) == some (some #[#[2, 2]]) &&
    !reads [form 3] && (scheduleOf (j 2 512 [form 1])).toOption.isNone &&
    (scheduleOf (j 13 512 [form 1])).toOption.isNone && (scheduleOf (j 5 386 [form 1])).toOption.isNone &&
    (scheduleOf (j 5 387 [form 1])).toOption.isSome && (scheduleOf (j 5 386 [])).toOption.isSome &&
    !reads [classes #[#[2, 2]]] && !reads [form 2, classes #[]] && !reads [form 2, classes #[#[2, 2, 1]]] &&
    !reads [form 1, classes #[#[2, 2]]] && !reads [form 2, classes #[#[3, 2]]] && !reads [form 2, classes #[#[2, 1]]] &&
    !reads [form 2, classes #[#[2, 8192]]] && !reads [form 2, classes #[#[4096, 2]]] &&
    !reads [form 1, classes #[#[2, 2, 0]]] && !reads [form 1, classes #[#[2, 2, 17]]]
-- the parse reads `plainSchedule` with request words beside form 2's classes and `max_tokens_cap`, and refuses request
-- words that are not true, under form 1, without content rows, without `max_tokens_cap` and without lengths rows; a
-- schedule with request words lists at most `mostClassesWords` classes, and one without them more
#guard
  let n := fun (x : Nat) => toJson x
  let reads := fun (more : List (String × Json)) => (scheduleOf (Json.mkObj ([("bucket_ms", n 100),
    ("cell_bytes", n 512), ("delay", n 5), ("egress_cells", n 9), ("every", n 2), ("ingress_buckets", n 3),
    ("ingress_cells", n 4), ("lanes", n 2), ("queue", n 2), ("slots", n 2), ("window", n 16)] ++ more)))
  let classes := fun (cs : Array (Array Nat)) => ("content", toJson cs)
  let form := fun (f : Nat) => ("lengths", n f)
  let limit := ("max_tokens_cap", n 4)
  let words := fun (b : Bool) => ("request_words", toJson b)
  let many := fun (k : Nat) => classes (Array.replicate k #[2, 2])
  (reads [form 2, classes #[#[2, 2]], limit, words true]).toOption.map (·.requestWords) == some true &&
    (reads [form 2, classes #[#[2, 2]], limit]).toOption.map (·.requestWords) == some false &&
    (reads [form 2, classes #[#[2, 2]], limit, words false]).toOption.isNone &&
    (reads [form 2, classes #[#[2, 2]], limit, ("request_words", n 1)]).toOption.isNone &&
    (reads [form 1, classes #[#[2, 2, 1]], limit, words true]).toOption.isNone &&
    (reads [form 2, limit, words true]).toOption.isNone &&
    (reads [form 2, classes #[#[2, 2]], words true]).toOption.isNone && (reads [limit, words true]).toOption.isNone &&
    (reads [form 2, many mostClassesWords, limit, words true]).toOption.isSome &&
    (reads [form 2, many (mostClassesWords + 1), limit, words true]).toOption.isNone &&
    (reads [form 2, many (mostClassesWords + 1), limit]).toOption.isSome

-- the check refuses a schedule with connection rows, whatever parse it came from, and no other: the period of no
-- execution phase and no audit under `plainSchedule` at T = 2 is accepted, and with lengths rows it is refused by name
#guard
  let p := fun (s : Schedule) =>
    (⟨#[], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", .rows s .empty ⟨0, 0, 0, 0, 0⟩, 0, none⟩ : Params)
  let reason := fun (s : Schedule) => match check (p s) ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ with
    | .accept .. => "accept"
    | .reject r => r
    | .undecided r => r
  let s : Schedule := { plainSchedule with window := 2 }
  (connectionRowsCheck (p s)).isNone && (connectionRowsCheck (p { s with lengths := some 1 })).isSome &&
    (connectionRowsCheck (p { s with content := some #[] })).isSome &&
    (connectionRowsCheck (p { s with requestWords := true })).isSome && reason s == "accept" &&
    reason { s with lengths := some 2 } == connectionRowsRefusal

end NetworkCheck.Period
