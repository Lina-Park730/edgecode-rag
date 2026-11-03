# Question revision record

All 20 questions are development data. Four questions per original category remain.
Wording revised after observing a dense-only optimum; no retrieval results were used to select among successive rewrites. No BM25 improvement is guaranteed.
Original mixed split labels are retained only in questions_original_reference.json, which no runner loads.

## q01_exact_context_invoke
Before: How does Context.invoke behave when its callback argument is a Command, including callback validation, default parameter filling, and context management?
After: Locate Context.invoke. How does it validate a Command callback, fill missing parameter defaults, and manage the context?

## q02_exact_option_parse_decls
Before: What validation and name-normalization rules does Option._parse_decls apply to option declarations?
After: Locate Option._parse_decls. How does it validate declarations and derive the Python parameter name?

## q03_exact_lazyfile_open_close
Before: How do LazyFile.open and LazyFile.close_intelligently handle deferred opening, open failures, and streams that the wrapper does not own?
After: Locate LazyFile.open and LazyFile.close_intelligently. How do they handle deferred opening, opening failures, and streams the wrapper does not own?

## q04_exact_resolve_incomplete
Before: What precedence does _resolve_incomplete use to choose the object responsible for shell completion?
After: Locate _resolve_incomplete. In what order does it choose the object responsible for shell completion?

## q09_cross_decorated_parameter_flow
Before: Trace how an @option-created parameter is stored on a decorated function and transferred into the final Command.params list. Why is the temporary list reversed?
After: Where is __click_params__ populated and consumed? Trace how its contents become Command.params and explain why the temporary list is reversed.

## q13_error_unknown_long_option
Before: A user misspells a long option. How are suggestions selected and how is the final error message formatted when there is one suggestion versus several?
After: An unknown option produces "Did you mean" or "Possible options". Where are suggestions selected, and how does the formatter choose between these messages?

## q15_error_invalid_option_combinations
Before: Which incompatible Option constructor settings are rejected by the debug-time validation block, and what exception is raised?
After: I see "nargs=-1 is not supported for options." Where is this raised, and what other incompatible settings does the same debug-time Option validation block reject?


Q01 expected answer expanded; Q02 prefix-priority error corrected. Gold spans unchanged. Source review is not independent human validation.
