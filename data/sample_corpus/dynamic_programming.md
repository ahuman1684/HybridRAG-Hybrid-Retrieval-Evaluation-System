# Dynamic Programming

Dynamic programming (DP) is a technique for solving problems by breaking
them into overlapping subproblems, solving each subproblem once, and
storing its result so it doesn't need to be recomputed. DP applies when a
problem has two properties: optimal substructure (an optimal solution to
the problem can be built from optimal solutions to its subproblems) and
overlapping subproblems (the same subproblems recur many times during a
naive recursive solution).

## Memoization vs. Tabulation

There are two standard ways to implement DP. Memoization (top-down) keeps
the natural recursive structure of the problem but caches the result of
each subproblem the first time it's computed, so later calls with the same
arguments return instantly from the cache instead of recomputing. Tabulation
(bottom-up) instead builds a table iteratively, starting from the smallest
subproblems and working up to the final answer, avoiding recursion overhead
and stack depth limits entirely. Tabulation is usually faster in practice
because it avoids function-call overhead, but memoization can be easier to
write correctly since it mirrors the problem's natural recursive
definition.

## Classic Example: Fibonacci

The canonical illustration is computing the nth Fibonacci number. A naive
recursive implementation has O(2^n) time complexity because it recomputes
fib(k) exponentially many times for smaller k. Memoizing the recursive calls
(caching fib(k) the first time it's computed) drops this to O(n) time and
O(n) space. Tabulating it bottom-up achieves the same O(n) time and can be
further optimized to O(1) space, since computing fib(n) only ever needs the
previous two values, not the entire table.

## Classic Example: Knapsack

The 0/1 knapsack problem - given items with weights and values, and a
weight capacity, maximize total value without exceeding capacity, where
each item can be taken at most once - is a standard DP exercise with
O(n * W) time and space complexity, where n is the number of items and W is
the capacity. The subproblem is: what is the best value achievable using
the first i items with capacity w? Each state depends only on smaller i and
w, which is exactly the optimal-substructure property DP requires.

## Recognizing When DP Applies

The most common interview signal for a DP problem is a request to find a
maximum, minimum, or count of ways to do something, combined with the
ability to express the answer for size n in terms of answers for smaller
sizes. If a greedy choice at each step can be proven optimal without
needing to consider alternatives, a greedy algorithm will be simpler and
faster than DP; DP is the right tool specifically when greedy choices are
not provably optimal and multiple subproblem outcomes must be compared.
