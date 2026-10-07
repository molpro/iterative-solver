Consistency audit playbook
==========================

This is the playbook followed for the consistency-audit work in PRs #559–#564, #566 and #569
(May 2026). It is intended for anyone who wants to redo or extend that audit, ideally driven by an
LLM with a human steering and verifying.

## Setup

1. **Fresh clone**, current `master`, working build under your normal compiler.
2. **A strict compiler for one verification build**: `gcc 16+` or recent `clang`. Many bugs that the
   first round found compile silently under older toolchains. Keep one build directory configured
   against a strict one.
3. **An LLM** with file access and shell access to the repo (e.g. Claude Code). The audit is too
   tedious to do by hand; the bottleneck is reviewer time, not the LLM's.

## The shape of a pass

A *pass* is one trip through the codebase looking for **one specific kind of problem**. Not "find
bugs" — "find this kind of bug." Mixing kinds makes the diff impossible to review.

Useful lenses, roughly in order from least invasive to most:

| # | Lens | What to look for |
|---|---|---|
| 1 | Build-system errors | undefined CMake vars (e.g. `${PROJECT_TEST}`), loop-variable typos, options referenced but not declared, missing `PUBLIC_HEADER` installs, files in `CMakeLists.txt` that don't exist, duplicate `SOURCES` / `PUBLIC_HEADER` keywords |
| 2 | Doc / README rot | `.h` filenames in docs that no longer exist, API examples that won't compile against current headers, broken badge / link refs |
| 3 | Dead code | files not referenced from any `CMakeLists.txt`, stub functions never called, members never read or written, `extern` declarations with no definition, commented-out targets |
| 4 | Header hygiene | `using` declarations at *file* scope in public headers (leak into every consumer), duplicate `extern template`, missing includes that compile only because of a transitive |
| 5 | Python / Cython binding bugs | rebound parameters that should have been mutated in-place, `np.array([n])` vs `np.zeros(n)`, missing `__del__` that leaks C-side state, unreachable `pass` |
| 6 | Fortran correctness | implicit-typed loop variables, dropped-letter typos that fall back to `IMPLICIT INTEGER`, locals shadowing module functions, dead module-private routines that never bind to a type |
| 7 | Subspace / linear-algebra core | inverted-bound comparisons (`oX >= i` vs `i >= oX`), dereffing `max_element` on a possibly empty container, popping the wrong end of a deque, off-by-one between `>` and `>=` on past-the-end iterators |
| 8 | DistrArray / `std::filesystem` family | size_t underflow (`current - offset` when `current < offset`), stray `;` after `if` making the body unconditional, file-scope globals without `static` |
| 9 | HDF5 / PHDF5 ownership | rvalue-reference parameters used as lvalues inside the body (named rvalues are lvalues), copy/move assignment asymmetry, leaked HDF5 IDs |
| 10 | `extern "C"` and C wrapper layer | declarations in headers as `extern "C"` but definitions in `.cpp` without it, unchecked `dynamic_cast` followed by `->`, top-of-stack assumptions when stack might be empty |
| 11 | Destructors that can throw | `~T()` is implicitly `noexcept`. Any call inside that can throw (`future::wait`, file I/O, `profiler->dotgraph`, `options->parameter`) terminates the program. Wrap in `try { } catch (...) {}` or guard the call. |

A later round added two more lenses that are worth keeping in mind, since both turned up bugs that
had survived every earlier pass:

| # | Lens | What to look for |
|---|---|---|
| 12 | Hardcoded scalar types | `double` where the surrounding code is templated — working buffers, thresholds, `Matrix<double>` in a generic container, LAPACK calls bound to one precision. See the series under #587. |
| 13 | Row- versus column-major confusion | an `Eigen::Map` whose storage order disagrees with how the caller filled the buffer. Invisible for a symmetric matrix and wrong for anything else, so it hides until the data stops being symmetric. |

## How to drive a single pass

1. **Prompt the LLM** with a tight description of what to look for, plus the path to scan. Be
   specific about the *kind* of bug. Example:

   > "Scan `src/molpro/linalg/array/` for headers that declare `using ...` at file scope (outside
   > any namespace). For each one, identify the symbols leaked and which downstream files use those
   > symbols unqualified. List the fixes needed (both header-side and consumer-side). Don't change
   > any code yet."

2. **Read the report**. Spot-check 2–3 findings against the actual source. LLMs hallucinate file
   paths and symbol names; verify before fixing.

3. **Ask for a patch** as a unified diff or have it apply edits directly. Keep the scope tight — one
   lens per branch.

4. **Build** in your normal config and the strict config. **Run the tests** if any exist for the
   touched files.

5. **Commit** with a message that explains *why* each fix matters, not just *what* changed. Group
   commits within the PR by sub-theme.

6. **Open the PR**. Title format that worked well: `<Component> <kind of fix>: <short summary>`. Body
   lists each commit and explains the audit method.

## Lessons from the first round

A few traps you'll hit that aren't obvious from the lens list:

* **The leaked-`using` cleanup is not mechanical.** `ArrayHandler.h` declared
  `using subspace::Matrix;` at file scope while a downstream `OptimizeBFGS.h` referenced
  `Matrix<double>` unqualified, so *moving* the `using` into the right namespace also broke the
  consumer — which under GCC 16 fails compilation and under older GCCs is a warning. The PR has to
  fix both sides at once. (This is what initially broke compilation in #556 and was caught by #562
  plus the follow-up `b359deb3`. Both sides are fixed now: the declarations sit inside
  `namespace molpro::linalg::array`, and the consumer qualifies the name.)
* **The strict compiler catches a different bug class** than your daily one. The two-stage
  name-lookup errors from `-Wtemplate-body` only surface in templated code instantiated by the test
  suite — not by configure.
* **Don't trust commit-message claims.** When splitting an existing PR, re-read each diff line by
  line.
* **Keep CMake and packaging out of the same PR as code logic.** Maintainers can rubber-stamp a
  CMake-hygiene PR in a minute. A mixed PR forces them to read every line.
* **Skip nothing.** If a pass turns up findings you don't want to fix (API-breaking, requires design
  discussion), file them as a tracker issue rather than silently dropping. #557 is the example from
  the first round.

## Lessons from later rounds

* **A test that has never failed proves nothing.** For any fix, verify the accompanying test fails
  without it. Several bugs in the #587 series had sat under a passing test that only asserted the
  result contained no NaNs, which a wrong-but-finite answer satisfies.
* **Make a tolerance relative to the precision it is checking.** A test that hardcodes `1e-17` is
  really asserting that `long double` is wider than `double`, which is false on arm64 macOS, where
  both are IEEE binary64. Express such bounds in units of `std::numeric_limits<T>::epsilon()`, and
  skip explicitly where the comparison is vacuous.
* **A no-op abstraction is not free in a debug build.** Wrapping a hot loop's body in a lambda that
  calls a function which is the identity for the type in question costs nothing at `-O3` and roughly
  a factor of two at `-O0`. Branch on the type with `if constexpr` so the common path is the code
  that was there before.
* **Reproduce the reviewer's platform rather than reasoning about it.** `-mlong-double-64` makes
  x86-64 GCC use the same `long double` as arm64 macOS, which turned a round of guesses about a
  reviewer's failing tests into a local reproduction.

## Suggested PR shape

Aim for PRs in the range **30–200 changed lines**, one lens per PR, *each rebased on current
`master`*. PRs over ~300 lines tend to attract "I can't review this" comments. The split from the
first series (#559–#564, #566 and #569) is the upper end of what's still reviewable — finer-grained
is fine.

Note that GitHub will not accept a base branch that lives on a fork, so a stack of
interdependent PRs from a fork all have to target `master`. Each diff then also shows the commits
below it until those merge, which is worth saying in the PR body so reviewers are not surprised.

## What to expect from the LLM

It will:

- Hallucinate file paths (verify with `git ls-files`).
- Sometimes claim a function is unused when there's one indirect caller through Cython,
  `extern template` or a Fortran ISO binding.
- Generally do an excellent job at *finding* candidate issues in a single lens, but mediocre at
  *judging* which ones are worth fixing. Keep a human in the loop for judgement.

A reasonable session length is one pass per session. Trying to do all the passes in one session leads
to context exhaustion and dropped findings.
