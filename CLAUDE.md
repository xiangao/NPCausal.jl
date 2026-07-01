# NPCausal.jl — project notes for Claude

Julia port of the estimators the author uses from the R package `npcausal`:
influence-function-based causal estimators with cross-fitting, sped up with
`MLJ.jl`/`EvoTrees.jl` and threaded cross-fitting folds.

## What's where

- `src/ate.jl` — `ate()`: doubly-robust ATE via cross-fitting, one propensity
  (classifier) + one outcome model (regressor) fit per fold per treatment
  level, threaded over folds (`Base.Threads.@threads`).
- `src/att.jl` — `att()`: average treatment effect on the treated, same
  cross-fitting scaffold as `ate.jl`.
- `src/ctseff.jl` — `ctseff()`: dose-response curve for a continuous
  treatment; local polynomial regression / kernel weighting + bandwidth
  selection helpers (`kern`, `locpoly`, `interp1d`, `kde`) live at the top of
  the file.
- `src/ivlate.jl` — `ivlate()`: complier average treatment effect (LATE) with
  a binary instrument; has a one-sided-noncompliance fast path
  (`onesided = sum(a[z .== 0]) == 0`).
- `src/ivbds.jl` — `ivbds()`: Manski-style effect bounds under an
  instrument, when point identification fails.
- `src/ipsi.jl` — `ipsi()`: incremental propensity score interventions
  (longitudinal, needs `time`/`id` columns and a grid `delta_seq`).
- `src/ensemble.jl` — `superlearner()`: builds an `MLJ.Stack` (GLM + random
  forest + EvoTrees + constant baseline) as a drop-in `mu_model`/`pi_model`
  for any of the estimators above.
- `src/NPCausal.jl` — module root. Exports the estimators plus five
  `admg_*` function *stubs* that only work once `CausalGraphs.jl` is loaded
  (see below); calling them without it raises a clear `error()` telling the
  user to install/load CausalGraphs.jl.
- `ext/NPCausalCausalGraphsExt.jl` — the real ADMG bridge, loaded as a Julia
  package extension (`[weakdeps]`/`[extensions]` in `Project.toml`) only
  when `CausalGraphs.jl` is also loaded. It calls `CausalGraphs.identify` to
  pick a strategy, then routes: `:a_fixable` → `NPCausal.ate`, `:p_fixable`
  → `CausalGraphs.nps_tmle_a` (NPS TMLE), `:nested_fixable` →
  `CausalGraphs.nested_anipw_a`, everything else → `CausalGraphs.estimate_id`
  (finite-support ID-algorithm plug-in with EIF confidence intervals).

## Relationship to CausalGraphs.jl

Direction: **CausalGraphs.jl identifies, NPCausal.jl estimates.**
CausalGraphs.jl does graph-based effect identification on ADMGs (backdoor /
front-door / nested-fixable / general ID); NPCausal.jl supplies the
influence-function estimators that turn an identified functional into a
number with a confidence interval.

The dependency is a mutual **weak dependency**, wired as two separate
package extensions, not a hard dependency in either direction:

- `NPCausal.jl`'s own extension is `ext/NPCausalCausalGraphsExt.jl`
  (`NPCausalCausalGraphsExt`, triggered by `CausalGraphs` in `[weakdeps]`).
  This is where `admg_estimate_causal` etc. actually get implemented.
- `CausalGraphs.jl` has its own mirror-image extension,
  `ext/CausalGraphsNPCausalExt.jl` (`CausalGraphsNPCausalExt`), which is a
  thin wrapper that just calls back into `NPCausal.admg_estimate_causal`.
  It exists so a user who loads `CausalGraphs` first (instead of `NPCausal`
  first) gets the same functionality under CausalGraphs' namespace.
  Its testset is `"optional NPCausal bridge"` in
  `~/projects/software/CausalGraphs.jl/test/runtests.jl`.

Neither package lists the other as a normal (non-weak) dependency — loading
either one alone works fine; the ADMG functions only activate when both are
`using`'d in the same session.

## Main exported API

```
ate, att, ctseff, ivlate, ivbds, ipsi, superlearner
admg_estimate_causal, admg_nps_tmle_a, admg_nested_anipw_a,
admg_id_plugin_a, admg_estimate_id   # all require CausalGraphs.jl loaded
```

## Tests — known thin-coverage gap

`test/runtests.jl` has **one** `@testset` ("NPCausal.jl") with **4**
`@test` assertions, all exercising `ate()` only on a single simulated DGP
(n=10000, 3-level categorical treatment, 5-fold cross-fitting). Confirmed
by running the suite (2026-07-01): `Test Summary: NPCausal.jl | Pass 4 Total 4`.

This is thin for a package implementing seven estimators plus a
graph-identification bridge:
- `att`, `ctseff`, `ivlate`, `ivbds`, `ipsi`, `superlearner` have **no**
  tests at all.
- The ADMG bridge (`ext/NPCausalCausalGraphsExt.jl`) has no tests in this
  repo (only exercised from CausalGraphs.jl's side, and only via one
  "optional NPCausal bridge" testset there).
- No tests check the doubly-robust property under model misspecification,
  no tests check `ivlate`/`ivbds` against a known compliance-rate DGP, no
  coverage-rate check on the reported confidence intervals.

Treat new estimator code as unverified until you add a DGP-based test with
a known closed-form target (mirror the pattern already in `runtests.jl`:
simulate, then check the contrast is close to the true effect within a
tolerance). This is a good first thing to improve in this repo.

Run tests with:
```bash
cd ~/projects/software/NPCausal.jl
julia --project=. test/runtests.jl
```
Takes about a minute — most of it is MLJ/EvoTrees/StatisticalMeasures
precompilation, not actual test time (~30s of that is the real run).

## CI gotcha: Julia version matrix

`.github/workflows/CI.yml` only tests `version: ["1"]` (latest stable), even
though `Project.toml` declares `julia = "1"` as the compat lower bound
(i.e. claims support back to Julia 1.0/LTS in principle). The workflow has
an explicit comment explaining why: the committed `Manifest.toml` was
resolved against a recent Julia and fails to instantiate on older versions
like 1.10 (stdlib UUID/version mismatches). So **1.10 (and any non-latest
"1.x") is untested in CI despite the compat bound implying it should work**.
If you touch `Project.toml`/`Manifest.toml` compat, either regenerate the
manifest so it resolves on 1.10 too, add 1.10 back to the CI matrix and
fix what breaks, or tighten `julia = "1"` to something CI actually verifies
(e.g. `julia = "1.10"` pinned to the tested version) so the claim in
`Project.toml` matches reality.

## Docs

Documenter.jl, source in `docs/src/` (`index.md` + two vignettes under
`docs/src/vignettes/`), built by `docs/make.jl`, deployed via
`.github/workflows/docs.yml` to GitHub Pages
(https://xiangao.github.io/NPCausal.jl/). Vignettes:
`01_getting_started` (ATE/ATT with cross-fitting) and
`02_advanced_estimators` (continuous treatment, IV, policy/incremental
propensity score interventions). `docs/build/` is committed (pre-rendered
output) — re-render with `julia --project=docs docs/make.jl` after any
docstring or vignette change, matching the "render locally, CI just
uploads" pattern used across the other Julia packages in this workspace.
