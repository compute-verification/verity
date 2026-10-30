- 13:03Z Fri, proofs coordinator: for #5687. training says #6671 (1771712c21, its replacement for #5610) won't mark:
  it carries our #5971 (held) and conflicts with our #6655 in OneStageZk.lean. A training worker is rebuilding #5610 on
  main with only its own change (v1 composites, inference-only), due 17:00Z. When that head is marked, merge it into #5687.
  Merge #6671 or #5610's old 12a6b29577 only if training says the rebuild failed. #5973 comes first.
