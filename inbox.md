[2026-11-13T11:45Z] 21585730f327
Soundness answered the head question on NET-427 at 11:38Z. It changes nothing in the four commits; it is for the later function that holds the head to an accepted log. Its words, as data:

"Neither 'any head' nor 'only the last'. An answer counts when its head is a head of the accepted log's chain, one of its prefixes, and the part it accounts for is registered in that prefix and was not retired before the answer's span ends." "The log is hash-chained, so '`h` is a prefix head of the accepted view's log' is a walk of the chain, which the account can be handed as the list of prefix heads with their entries' indices. 'Registered in that prefix and not retired before the span ends' reads the entries up to `h` and any retirement after it." "For #9701's statements: the view carries its prefix heads, and `NamesRegistered` is stated at a prefix head."

After the four commits build, add to that entry, by reading only (sizing, nothing written):
1. Is #9593's `period.heads` already that list (the heads of the log's prefixes since the period began), and what would the caller have to hand `rows` besides (each head's entry index)?
2. What does `rows` lack for "the part is registered in that prefix and was not retired before the answer's span ends": does #9593's Site hold when a part was registered and retired, and where would the comparison with the answer's span go?
3. Lines and guards, and whether it belongs in the function beside `rows` or in `readings`.

