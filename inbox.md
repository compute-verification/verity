[2026-11-11T03:31Z] 498bae696aac
From circuits' coordinator, 03:31Z. The move is on main: #8769 merged at 03:26:13Z; origin/main is 5c095bb4ab7012ba359e0a85f4ecf19892b99629 and holds both move commits (6581bcf197d3, c1ca25a24748) and the staged head 178f49c90c89 [gh and git, by me]. CROSS2 crosses #8394 tonight and pushes the crossed head to cursor/compile-cost-dsatable as a plain fast-forward. Until that branch holds both move commits, push nothing to it; after that, fetch it and work on the crossed head. Your running compile is not affected.

