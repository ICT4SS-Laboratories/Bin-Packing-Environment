# `solver_364130_354977_356856_359530` — How the solver works, step by step

3-D bin packing: place **every** item into containers (vehicles) of heterogeneous
types, minimizing the **total cost** of the containers used, subject to: no
overlaps, items inside the container, `maxWeight`, `maxValue` (may be `+∞`),
allowed rotations, and the `gravityStrength` support constraint. Unlimited
containers per type; cost is paid per container used regardless of fill.

The solver combines a fast geometric placement engine with a portfolio of
constructive heuristics, large-neighborhood search, column generation, and an
exact set-partition MILP. Everything is **strict-improvement / keep-the-best**,
so adding heuristics can only help, never regress a result.

---

## 0. Data model (`parse_items`, `parse_vehicles`)

- **Item**: `id, w, d, h, weight, value`, the set of **unique rotations**
  `urots` (each allowed `orient` → permuted dims, deduplicated), plus derived
  keys: `vol`, `maxdim`, `base_area`, `density_w`, `density_v`.
- **Vehicle**: `W, D, H, max_weight, max_value (∞ if blank), cost, gravity`,
  plus efficiency ratios `cpv` (cost/volume), `wpc` (cost/weight),
  `vpc` (cost/value).
- **Rotation table** (`_ROT_IDX`) is identical to the official checker's
  `get_dims`, so an `orient` code means the same box dimensions in the solver
  and in the checker.

### Coordinate convention
Internally the box at `(x,y,z)` spans `[x,x+iw]×[y,y+id_]×[z,z+ih]` with the
x-axis bounded by vehicle **width**, y by **depth**, z by **height**. On output
the CSV swaps `x_origin = internal y`, `y_origin = internal x` so the checker's
constraints (`x+depth ≤ vehicle.depth`, `y+width ≤ vehicle.width`) hold exactly.

---

## 1. The placement engine — `Bin3D` (Extreme Points)

A single container. Placement uses **Extreme Points** (Crainic et al. 2008):

- EPs start at `{(0,0,0)}`. Placing a box adds up to 6 child EPs (the 3 axis
  corners plus 3 diagonal corners), tried sorted by `(z, x, y)` — i.e. bottom-up,
  which keeps placements gravity-friendly.
- `find_ep` returns the first feasible EP for a given rotation, checking in
  order: capacity (`cap_ok` for weight/value), bin bounds, overlap, gravity.
- `try_add` tries every unique rotation and keeps the **best** EP by a score
  that prefers low z (bottom), then a tight residual fit, then large footprint.
- Overlap uses a short-circuit pure-Python scan; gravity uses an index of boxes
  bucketed by their **top-z**, so the support scan only visits boxes that could
  actually support the new one.

### Float-safety margins (`_SAFE`, `_SAFE_GRAV`) + output repair
The official checker compares geometry with **strict, zero-tolerance**
arithmetic. With **non-integer dimensions** (e.g. `968.9`), touching faces and
stacks accumulate ~`1e-12…1e-9` binary-float error, which the checker would flag
as spurious overlaps / gravity shortfalls. So `solve()` detects whether the
instance has any non-integer dimension and, if so, packs with a tiny clearance:

- `_SAFE = 1e-7`: child EPs are offset by `_SAFE`, overlap requires `_SAFE`
  clearance, and bounds reserve `_SAFE`. `_SAFE` is **below** the checker's
  `1e-6` gravity z-window (so a box placed `_SAFE` above its support is still
  counted as supported) and **above** the worst float noise (~`2e-9`).
- `_SAFE_GRAV = 1e-6`: a box supported by *multiple* partial supporters must
  exceed the gravity requirement by this margin (absorbs seam noise; the slack
  exists only when `gravity < 100%`).
- A box supported by a **single supporter that fully covers its footprint** is
  accepted at any gravity (including 100%), because its support equals the box's
  own base — which the **output repair** then makes checker-exact.
- **Integer instances** (e.g. datasets A–J) set both margins to `0.0`, so their
  behaviour is byte-for-byte unchanged.

**Output repair** (`_repair_bin_for_checker`, non-integer instances only): a
box's own width as the checker computes it, `(origin+dim)-origin`, can be off by
±½ULP, so a stacked 100%-gravity box can read as unsupported by machine epsilon.
Before writing, each bin is replayed through an **exact replica of the checker**
(`_checker_box_ok`: bounds + overlap + gravity) in internal coords. Boxes that
fail are *snapped*: coordinates shared by several boxes are nudged together (over
a sub-ULP grid via `math.nextafter`) to a value where the checker's arithmetic
rounds favourably, preserving every support/edge relationship. Any box that
still can't be made feasible is split off to its own floor bin (always
feasible). For non-integer instances the CSV is written at **full precision**
(no 9-decimal rounding, which would erase the sub-ULP snaps); the repair is
verified, so the output provably passes the official checker.

This is a **generic, data-driven** robustness switch, not dataset tuning.

---

## 2. Vehicle orderings & item orderings (diversification)

- `vehicle_orderings` builds several priority lists over vehicle types:
  `cost`, `cpv`, `capacity` (weight/value efficiency), `large`, `big_eff`
  (volume per cost), `balanced`. A `vehicle_cycle` rotates through them so
  different parts of the search prefer different container shapes.
- `build_sequence(items, mode, …)` produces item orderings: `vol`, `weight`,
  `value`, **`vmax`** (dominant normalised resource pressure — see below),
  `maxdim`, `footprint`, `densw`, `densv`, `hard`, `mixed`, `shuffle`.
- Per-item **`hardness`** and **`vec_pressure`** are precomputed from how many
  vehicles accept the item and how tight the fit is — used only to drive generic
  ordering, never dataset-specific branching.

---

## 3. Constructive heuristics

- **GRASP** (`grasp`): greedy first/best-fit over an item sequence with a
  randomized restricted candidate list (`alpha`), packing into `open_bin`-chosen
  containers.
- **Resource-targeted FFD constructors**, each picks the most cost-efficient
  "primary" vehicle and fills it to a cap, then **retypes** every bin down to
  its cheapest viable vehicle:
  - `construct_weight_packed` — weight-binding instances (LPT distribution).
  - `construct_volume_packed` — volume-binding instances.
  - **`construct_knapsack_packed`** — tight-fill FFD for resource-bound
    fleets: routes each item to the vehicle type cheapest *per unit of the
    item's binding resource* among those that accept it (two-tier structure
    emerges automatically, e.g. light/low-value items into the small efficient
    vehicle, the rest into the next class), packs each tier first-fit in
    descending binding share, squeezes zero-demand items into spare geometry.
  - **`construct_towers`** — tower/shelf constructor: groups identical-dims
    items, stacks them into full-support towers (identical footprints ⇒ 100%
    support ⇒ valid at any gravityStrength) and shelf-packs the floor; tries
    3 height policies × 2 shelf openers × top-2 primaries, feeding ALL covers
    to the column pool. On repeated-shape instances it additionally mines the
    per-bin **patterns** (group-count vectors + placement templates) and solves
    a small cutting-stock cover IP (`scipy.milp`) so mono-type and mixed
    patterns from different runs can be combined — greedy runs alone partition
    the items and the set-partition MILP could never mix them.
  - **`construct_vector_packed`** — generic multi-resource constructor. It fills
    the gap left by the two above: when weight **and/or value** (and volume)
    bind together, it balances all *active* (finite-cap, non-zero-demand)
    resources at once. It degrades gracefully — if `maxValue` is `∞` the value
    term vanishes; if only weight binds it behaves like weight-FFD — so it is
    not tuned to any particular fleet.
  - **`construct_layered`** — layer/shelf builder: items tall-first then by
    footprint, growing each bin in stable bottom layers. Reaches different,
    often denser, volume-bound packings than the other constructors.
  - `construct_weight_packed_diverse` — multiple primaries for column diversity.

---

## 4. Large Neighborhood Search — `lns` (ALNS + VNS)

Adaptive LNS over a current solution. `op_elim` (empty the emptiest bin, repack
elsewhere) is always tried first; the rest are chosen by a **roulette wheel**
whose weights adapt to recent success and decay toward uniform for exploration.
On stagnation it escalates "ruin-and-recreate" kicks (VNS). It always returns the
**best** solution seen. Operators (all strict-improvement) include:

`op_relocate`, `op_retype` / `op_retype_all` (downsize containers),
`op_shake`, `op_merge3`, `op_eject`, `op_swap_pair`, `op_consolidate_pair`
(merge two bins into one cheaper bin), `op_ruin_recreate` (+ a stronger variant),
`op_weight_pair_repack` (pairs on the *binding* normalised resource — weight
or value), `op_bin_split`, `op_redistribute_then_retype`, and
**`op_drain_retype`**: partially drains an under-filled expensive bin into the
slack of ANY other bin (cross-type receivers; falls back to load-reducing
item swaps when no direct move fits), then retypes the drained bin to a
cheaper vehicle. This captures 'tail' bins whose load sits just above a
cheaper vehicle's capacity — e.g. on DatasetI it turns the V5 tail into a V6
(−785.45) where retype/redistribute alone can never fire.

`op_eject` / `op_swap_pair` re-pack bins item-by-item; both now verify every
`try_add` (a failed re-add used to silently DROP the item — the cost looked
lower, `update_best` accepted it, and the end-of-run repair re-opened a fresh
bin for it, a net loss). `op_swap_pair` also works on a copy per swap
candidate so a half-applied swap can't duplicate an item across two bins.

---

## 5. Column generation + exact set partition

- Every feasible bin discovered anywhere is stored as a **column** in a
  thread-safe `ColumnPool` (deduplicated by item-set, cheapest kept).
- Targeted generators add structurally diverse columns: per-item, pair-seeded,
  cost-targeted (repack expensive bins into cheaper vehicles), dual-guided, and
  shadow-mode Dantzig–Wolfe cycles (committed only if they actually help).
- **`generate_columns_resource_knapsack`**: for the vehicle types most
  cost-efficient per unit of each active finite resource, builds bins filled
  as close to the binding cap as possible (largest seed + largest
  complementary top-ups, jittered sweeps). Supplies the tight pair/triple
  columns the MILP needs on weight-/value-bound instances; skips vehicles
  where no item uses ≥5% of the caps (geometry-bound — nothing to gain).
- **`generate_columns_pair_matching`**: deterministic two-pointer
  max-cardinality pair matching on the binding normalised share (optimal pair
  count for one capacity) — one column per pair on the most efficient vehicle
  types. The wholesale alternative to singleton bins.
- **Instance-adaptive budget split**: after Phase 1, if the incumbent's
  weight/value fills dominate its volume fill (resource-bound instance),
  Phase 2 (GRASP restarts — they plateau early on such fleets) is shortened
  and every Phase-3 CG/MILP window is doubled. MILP column caps also scale
  with the incumbent's bin count (≥4× bins), since a pool capped near the
  solution size leaves the set-partition no combinatorial freedom.
- `solve_set_partition` runs an exact **MILP** (HiGHS, fallback CP-SAT / SciPy)
  that selects the cheapest subset of columns covering every item exactly,
  warm-started with the incumbent. The pool is LP-filtered down to a column cap
  before solving. The MILP can only pick a subset — it never invents placements —
  so it is always followed by a polish pass.

---

## 6. Orchestration — `solve()` (10-minute budget, ≤4 threads)

1. **Setup** — parse, set float-safety, compute hardness/vec_pressure, build
   vehicle/item orderings, seeds.
2. **Phase 1 — parallel construction** (~10% of time): a portfolio of GRASP
   configs + the resource FFD constructors (weight / volume / **vector** /
   **layered**) run in a thread pool; every result feeds the column pool and the
   incumbent.
3. **Phase 2 — GRASP restarts + LNS** (bulk of time): one deep-LNS worker plus
   restart workers cycling orderings/vehicle priorities (including **`vmax`**),
   harvesting columns continuously.
4. **Phase 3 — column generation + MILP**: pre-seed (weight / volume / **vector**
   / diverse), targeted CG (pair-seeded, item-centric, cost-targeted), then the
   set-partition MILP; an intensification round mines columns near the incumbent
   and re-runs the MILP.
5. **Phase 4 — final polish**: path relinking, more LNS bursts (cost- and
   balanced-vehicle ordered, including `vmax`), deterministic
   `post_optimize_bins` (strict-improvement only), a final MILP over the full
   pool, and a last polish.
6. **Output**: build the CSV (swapping coordinates as in §0), repair any missing
   item by opening a fresh bin, and write `results/sol_<Dataset>_solver_364130.csv`.

The incumbent is guarded by `update_best` (accepts only strictly cheaper
feasible solutions), so none of the phases can worsen the result.

---

## 7. Tooling (not part of the graded solver)

- `tools/eval_solutions.py A-J | K-Z` — replays the **exact** official checker
  logic over many datasets and prints cost, feasibility, a valid lower bound and
  the gap. Use it after every run to compare against held-out datasets.
- `tools/run_all.py A-J | K-Z` — runs the solver over a range of datasets
  back-to-back without modifying the graded `main.py`.

---

## 8. Design principles (anti-overfitting)

- No `if dataset == …`, no hardcoded vehicle picks, no constants eyeballed from
  the visible datasets. Every decision is computed at runtime from the instance
  and **degrades gracefully** (e.g. value handling disappears when `maxValue` is
  infinite; float-safety margins are `0` for integer instances).
- Additive, strict-improvement structure: new constructors / orderings /
  operators can only be kept if they help, so the solver stays robust on unseen
  (hidden) instances.
