"""
solver_354977.py
================
3D Heterogeneous Bin Packing Solver

Architecture
------------
Phase 1 (≈8 min)  — Four parallel metaheuristic workers generate diverse
                     packing *patterns* (one loaded container each) and
                     register them in a shared pool:
                       • Worker 1 : BRKGA   – biased random-key genetic algo
                       • Worker 2 : ALNS    – adaptive large neighbourhood search
                       • Worker 3 : GRASP   – greedy randomised with wall-build bias
                       • Worker 4 : VNS     – variable neighbourhood search

Phase 2 (≈75 sec) — Set-Covering ILP (OR-Tools / SCIP) selects the minimum-cost
                     subset of patterns that covers every item exactly once.

Coordinate convention
---------------------
    x-axis  →  depth   (container depth runs along x)
    y-axis  →  width
    z-axis  →  height

Rotation codes (applied to item (depth, width, height)):
    0  →  (d, w, h)   1  →  (w, d, h)   2  →  (d, h, w)
    3  →  (h, d, w)   4  →  (h, w, d)   5  →  (w, h, d)
"""

import time
import random
import multiprocessing as mp
import numpy as np
import pandas as pd
from .abstract_solver import AbstractSolver

# Fix for macOS segfault with OR-Tools and multiprocessing
# Must set spawn method before any other multiprocessing code
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

try:
    from ortools.linear_solver import pywraplp
    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False

# ===========================================================================
# § 1  ROTATION UTILITIES
# ===========================================================================

_ROT = {
    '0': (0, 1, 2),
    '1': (1, 0, 2),
    '2': (0, 2, 1),
    '3': (2, 0, 1),
    '4': (2, 1, 0),
    '5': (1, 2, 0),
}


def apply_rot(depth: float, width: float, height: float, code: str):
    """Return (dx, dy, dz) after applying rotation *code* to (depth, width, height)."""
    dims = (depth, width, height)
    p = _ROT[code]
    return dims[p[0]], dims[p[1]], dims[p[2]]


def item_rot_options(it: dict):
    """Return list of (code, dx, dy, dz) for every allowed rotation of *it*."""
    d, w, h = it['depth'], it['width'], it['height']
    return [(r, *apply_rot(d, w, h, r)) for r in it['rots']]


# ===========================================================================
# § 2  GEOMETRY ENGINE
# ===========================================================================

_EPS = 1e-7


def overlaps(ax, ay, az, adx, ady, adz,
             bx, by, bz, bdx, bdy, bdz) -> bool:
    return not (ax + adx <= bx + _EPS or bx + bdx <= ax + _EPS or
                ay + ady <= by + _EPS or by + bdy <= ay + _EPS or
                az + adz <= bz + _EPS or bz + bdz <= az + _EPS)


def supported(x, y, z, dx, dy, placed: list, gravity_pct: float) -> bool:
    """
    True if the item footprint (x,y)→(x+dx,y+dy) at height *z* receives
    at least *gravity_pct* % area support from items whose top is at *z*,
    or if z≈0 (floor) or gravity_pct≈0.
    """
    if z < _EPS or gravity_pct < _EPS:
        return True
    required = dx * dy * gravity_pct / 100.0
    support = 0.0
    for p in placed:
        if abs(p['z'] + p['dz'] - z) < _EPS:
            ix = max(0.0, min(x + dx, p['x'] + p['dx']) - max(x, p['x']))
            iy = max(0.0, min(y + dy, p['y'] + p['dy']) - max(y, p['y']))
            support += ix * iy
            if support >= required - _EPS:
                return True
    return False


def compute_eps(placed: list, vdx: float, vdy: float, vdz: float) -> list:
    """
    Generate candidate extreme-point positions from the placed items.
    Each placed item contributes three new candidate points (one per axis).
    Result is sorted Deepest-Bottom-Left-Fill: (z asc, y asc, x asc).
    """
    seen = set()
    pts = []

    def add(px, py, pz):
        key = (round(px, 9), round(py, 9), round(pz, 9))
        if key not in seen:
            seen.add(key)
            pts.append((px, py, pz))

    add(0.0, 0.0, 0.0)
    for p in placed:
        cx = p['x'] + p['dx']
        cy = p['y'] + p['dy']
        cz = p['z'] + p['dz']
        if cx < vdx - _EPS: add(cx, p['y'], p['z'])
        if cy < vdy - _EPS: add(p['x'], cy, p['z'])
        if cz < vdz - _EPS: add(p['x'], p['y'], cz)

    pts.sort(key=lambda q: (q[2], q[1], q[0]))
    return pts


# ===========================================================================
# § 3  SINGLE-BIN PACKER
# ===========================================================================

def pack_bin(item_seq: list, items_data: dict, vtype: str, vdata: dict):
    """
    Greedily fill *one* bin of type *vtype* with items from *item_seq*
    (each element is (item_id, [rot_codes])) using the EP+DBLF strategy.

    Returns a pattern dict  {vtype, cost, ids, items}  or  None.
    """
    vdx = vdata['depth']
    vdy = vdata['width']
    vdz = vdata['height']
    grav = vdata['gravity']
    max_w = vdata['max_weight']
    max_v = vdata['max_value']

    placed = []
    cur_w = cur_v = 0.0
    placed_ids = []

    for item_id, rots in item_seq:
        it = items_data[item_id]

        # ── capacity quick-rejects ─────────────────────────────────────────
        if cur_w + it['weight'] > max_w + _EPS:
            continue
        if max_v is not None and cur_v + it['value'] > max_v + _EPS:
            continue

        rot_opts = [(r, *apply_rot(it['depth'], it['width'], it['height'], r))
                    for r in rots]

        # ── search extreme points ──────────────────────────────────────────
        eps = compute_eps(placed, vdx, vdy, vdz)
        found = None
        for ex, ey, ez in eps:
            for r, dx, dy, dz in rot_opts:
                if ex + dx > vdx + _EPS: continue
                if ey + dy > vdy + _EPS: continue
                if ez + dz > vdz + _EPS: continue
                if any(overlaps(ex, ey, ez, dx, dy, dz,
                                p['x'], p['y'], p['z'],
                                p['dx'], p['dy'], p['dz'])
                       for p in placed):
                    continue
                if not supported(ex, ey, ez, dx, dy, placed, grav):
                    continue
                found = dict(id=item_id, x=ex, y=ey, z=ez,
                             dx=dx, dy=dy, dz=dz, orient=r)
                break
            if found:
                break

        if found:
            placed.append(found)
            cur_w += it['weight']
            cur_v += it['value']
            placed_ids.append(item_id)

    if not placed:
        return None

    return dict(vtype=vtype, cost=vdata['cost'],
                ids=frozenset(placed_ids), items=placed)


# ===========================================================================
# § 4  SEQUENCE → PATTERN RUNNER
# ===========================================================================

def _register(pat: dict, pool):
    """Thread-safe insert into shared pool (keyed by content hash)."""
    key = hash((pat['ids'], pat['vtype']))
    if key not in pool:
        pool[key] = pat


def _force_single(item_id: str, it: dict, vehicles_data: dict, pool):
    """Place one item alone in the cheapest fitting vehicle."""
    for vt in sorted(vehicles_data, key=lambda v: vehicles_data[v]['cost']):
        vd = vehicles_data[vt]
        if it['weight'] > vd['max_weight'] + _EPS:
            continue
        if vd['max_value'] is not None and it['value'] > vd['max_value'] + _EPS:
            continue
        for r in it['rots']:
            dx, dy, dz = apply_rot(it['depth'], it['width'], it['height'], r)
            if dx <= vd['depth'] + _EPS and dy <= vd['width'] + _EPS and dz <= vd['height'] + _EPS:
                pat = dict(vtype=vt, cost=vd['cost'],
                           ids=frozenset([item_id]),
                           items=[dict(id=item_id, x=0.0, y=0.0, z=0.0,
                                       dx=dx, dy=dy, dz=dz, orient=r)])
                _register(pat, pool)
                return True
    return False


def run_sequence(seq: list, items_data: dict, vehicles_data: dict, pool):
    """
    Pack the full sequence by repeatedly filling one bin at a time.
    For each bin-load the cheapest vehicle type that fits the most items is chosen.
    Every completed bin is registered as a pattern in *pool*.
    """
    vtypes_by_cost = sorted(vehicles_data, key=lambda v: vehicles_data[v]['cost'])
    remaining = list(seq)

    while remaining:
        best_pat = None

        for vt in vtypes_by_cost:
            pat = pack_bin(remaining, items_data, vt, vehicles_data[vt])
            if pat is None:
                continue
            if best_pat is None or len(pat['ids']) > len(best_pat['ids']):
                best_pat = pat
            if len(best_pat['ids']) == len(remaining):
                break           # Can't do better

        if best_pat is None:
            # Nothing fitted at all — force first item alone, then continue
            item_id, _ = remaining[0]
            _force_single(item_id, items_data[item_id], vehicles_data, pool)
            remaining = remaining[1:]
            continue

        _register(best_pat, pool)
        placed = best_pat['ids']
        remaining = [(iid, rots) for iid, rots in remaining if iid not in placed]


# ===========================================================================
# § 5  WORKER PROCESSES  (must be top-level for pickle / multiprocessing)
# ===========================================================================

# ── Worker 1: BRKGA ─────────────────────────────────────────────────────────

def worker_brkga(items_data, vehicles_data, t_limit, t0, pool, seed):
    """
    Biased Random-Key Genetic Algorithm.

    Chromosome = 2n floats in [0,1]:
      • first  n  →  item ordering (decoded by argsort)
      • second n  →  rotation selection (scaled index into allowed rots)

    Population is maintained with elite preservation, mutants, and biased
    crossover (elite allele preferred with probability ELITE_BIAS).
    """
    random.seed(seed)
    np.random.seed(seed % (2**31 - 1))

    ids = list(items_data)
    n = len(ids)
    POP, ELITE_R, MUT_R, BIAS = 30, 0.15, 0.10, 0.70
    n_elite = max(1, int(POP * ELITE_R))
    n_mut   = max(1, int(POP * MUT_R))

    def make_chrom():
        return np.random.rand(2 * n)

    def decode(c):
        order = np.argsort(c[:n])
        seq = []
        for i in order:
            iid = ids[i]
            rots = items_data[iid]['rots']
            r = rots[int(c[n + i] * len(rots)) % len(rots)]
            seq.append((iid, [r]))
        return seq

    pop = [make_chrom() for _ in range(POP)]

    while time.time() - t0 < t_limit:
        for chrom in pop:
            run_sequence(decode(chrom), items_data, vehicles_data, pool)

        elite   = pop[:n_elite]
        mutants = [make_chrom() for _ in range(n_mut)]
        n_cross = POP - n_elite - n_mut
        cross   = []
        for _ in range(n_cross):
            e = random.choice(elite)
            o = pop[random.randint(n_elite, POP - 1)] if POP > n_elite else make_chrom()
            mask = np.random.rand(2 * n) < BIAS
            cross.append(np.where(mask, e, o))

        pop = elite + mutants + cross


# ── Worker 2: ALNS ──────────────────────────────────────────────────────────

def worker_alns(items_data, vehicles_data, t_limit, t0, pool, seed):
    """
    Adaptive Large Neighbourhood Search.

    Repeatedly destroys 15-40 % of the current sequence (random or
    similarity-based removal) and repairs by re-inserting with fresh
    rotation samples at random positions.
    """
    random.seed(seed)
    ids = list(items_data)

    def rand_seq():
        s = [(iid, [random.choice(items_data[iid]['rots'])]) for iid in ids]
        random.shuffle(s)
        return s

    seq = rand_seq()
    op_weights = [1.0, 1.0]      # [random_removal, worst_removal]
    op_scores  = [0.0, 0.0]
    op_uses    = [0,   0]
    SEGMENT    = 50
    itr        = 0

    while time.time() - t0 < t_limit:
        itr += 1
        n = len(seq)

        # ── operator selection (roulette) ──────────────────────────────────
        total_w = sum(op_weights)
        r_op = random.random() * total_w
        op_idx = 0
        for i, w in enumerate(op_weights):
            r_op -= w
            if r_op <= 0:
                op_idx = i
                break

        # ── destroy ───────────────────────────────────────────────────────
        k = max(1, int(n * random.uniform(0.15, 0.40)))

        if op_idx == 0:                         # random removal
            rm_idx = set(random.sample(range(n), k))
        else:                                   # largest-item removal
            scores_i = []
            for j, (iid, _) in enumerate(seq):
                it = items_data[iid]
                scores_i.append((it['depth'] * it['width'] * it['height'], j))
            scores_i.sort(reverse=True)
            rm_idx = set(j for _, j in scores_i[:k])

        removed  = [seq[j] for j in sorted(rm_idx)]
        retained = [seq[j] for j in range(n) if j not in rm_idx]

        # ── repair (regret-style: insert hardest item first) ───────────────
        random.shuffle(removed)
        for iid, _ in removed:
            rot = random.choice(items_data[iid]['rots'])
            pos = random.randint(0, len(retained))
            retained.insert(pos, (iid, [rot]))

        seq = retained
        run_sequence(seq, items_data, vehicles_data, pool)
        op_scores[op_idx] += 1.0
        op_uses[op_idx]   += 1

        # ── adapt weights every SEGMENT iterations ─────────────────────────
        if itr % SEGMENT == 0:
            RHO = 0.8
            for i in range(len(op_weights)):
                if op_uses[i] > 0:
                    op_weights[i] = RHO * op_weights[i] + (1 - RHO) * op_scores[i] / op_uses[i]
                    op_weights[i] = max(op_weights[i], 0.01)
                op_scores[i] = op_uses[i] = 0


# ── Worker 3: GRASP ─────────────────────────────────────────────────────────

def worker_grasp(items_data, vehicles_data, t_limit, t0, pool, seed):
    """
    Greedy Randomised Adaptive Search Procedure with a wall-building bias.

    Items are scored by their difficulty (largest dimension first) plus
    Gaussian noise to diversify.  A random fraction α controls greediness.
    """
    random.seed(seed)

    ids = list(items_data)

    while time.time() - t0 < t_limit:
        alpha = random.uniform(0.0, 0.45)
        scored = []
        for iid in ids:
            it = items_data[iid]
            r = random.choice(it['rots'])
            dx, dy, dz = apply_rot(it['depth'], it['width'], it['height'], r)
            # score = volume + largest-dim bonus (large, awkward items first)
            score = dx * dy * dz + max(dx, dy, dz) + random.gauss(0, 0.05)
            scored.append((score, iid, r))

        scored.sort(reverse=True)
        cutoff = max(1, int(len(scored) * (1 - alpha)))
        rcl = scored[:cutoff]
        random.shuffle(rcl)

        seq = [(iid, [r]) for _, iid, r in rcl]
        run_sequence(seq, items_data, vehicles_data, pool)


# ── Worker 4: VNS ───────────────────────────────────────────────────────────

def worker_vns(items_data, vehicles_data, t_limit, t0, pool, seed):
    """
    Variable Neighbourhood Search.

    Iterates through increasingly disruptive neighbourhood structures:
      k=1  swap two items
      k=2  re-rotate ~12% of items
      k=3  shift one item to a new position
      k=4  reverse a random segment
      k=5  shuffle a random chunk
    After each perturbation the sequence is packed and registered.
    """
    random.seed(seed)
    ids = list(items_data)

    seq = [(iid, [random.choice(items_data[iid]['rots'])]) for iid in ids]
    random.shuffle(seq)
    K_MAX = 5
    k = 1

    while time.time() - t0 < t_limit:
        ns = list(seq)
        n = len(ns)

        if k == 1 and n >= 2:
            i, j = random.sample(range(n), 2)
            ns[i], ns[j] = ns[j], ns[i]

        elif k == 2:
            for _ in range(max(1, n // 8)):
                idx = random.randint(0, n - 1)
                iid = ns[idx][0]
                ns[idx] = (iid, [random.choice(items_data[iid]['rots'])])

        elif k == 3 and n >= 2:
            i = random.randint(0, n - 1)
            item = ns.pop(i)
            ns.insert(random.randint(0, n - 1), item)

        elif k == 4 and n >= 3:
            i, j = sorted(random.sample(range(n), 2))
            ns[i:j + 1] = ns[i:j + 1][::-1]

        elif k == 5 and n >= 4:
            chunk = random.randint(2, max(2, n // 4))
            start = random.randint(0, n - chunk)
            seg = ns[start:start + chunk]
            random.shuffle(seg)
            ns[start:start + chunk] = seg

        run_sequence(ns, items_data, vehicles_data, pool)
        k = 1 if k >= K_MAX else k + 1
        seq = ns


# ===========================================================================
# § 6  GREEDY SET-COVERING FALLBACK
# ===========================================================================

def greedy_cover(patterns: list, item_ids: set) -> list:
    """
    Simple greedy set-covering heuristic used when OR-Tools is unavailable
    or the ILP times out without a feasible solution.

    Iteratively selects the pattern with the best (items_covered / cost) ratio.
    """
    uncovered = set(item_ids)
    selected  = []

    while uncovered:
        best_pat   = None
        best_ratio = -1.0

        for pat in patterns:
            gain = len(pat['ids'] & uncovered)
            if gain > 0:
                ratio = gain / pat['cost']
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pat   = pat

        if best_pat is None:
            break

        selected.append(best_pat)
        uncovered -= best_pat['ids']

    return selected


# ===========================================================================
# § 7  MAIN SOLVER CLASS
# ===========================================================================

class solver_354977(AbstractSolver):
    """
    Solver entry-point.  Inherits from AbstractSolver and implements solve().
    The class name MUST match the file name for the project's main.py to work.
    """

    # ── instance parsing ────────────────────────────────────────────────────

    def _parse(self):
        """
        Convert self.inst DataFrames into plain dicts for inter-process sharing.

        items_data  : {item_id → {depth, width, height, weight, value, rots}}
        vehicles_data: {vtype  → {depth, width, height, max_weight, cost,
                                   max_value, gravity}}
        """
        # reset_index() promotes the named index (id / type) back to a column
        df_i = self.inst.df_items.reset_index()
        df_v = self.inst.df_vehicles.reset_index()

        # Some datasets expose identifiers as index, others as explicit columns.
        # Resolve them once to avoid fragile direct lookups (e.g. row['id']).
        item_id_col = next(
            (c for c in ('id', 'item_id', 'itemId', 'index') if c in df_i.columns),
            None,
        )
        vehicle_type_col = next(
            (c for c in ('type', 'vehicle_type', 'vehicleType', 'index') if c in df_v.columns),
            None,
        )

        items_data = {}
        for idx, row in df_i.iterrows():
            iid_raw = row.get(item_id_col, idx) if item_id_col is not None else idx
            iid = str(iid_raw)
            rots = [c for c in str(row['allowedRotations']) if c in _ROT] or ['0']
            items_data[iid] = dict(
                depth  = float(row['depth']),
                width  = float(row['width']),
                height = float(row['height']),
                weight = float(row['weight']),
                value  = float(row['value']),
                rots   = rots,
            )

        vehicles_data = {}
        for idx, row in df_v.iterrows():
            vt_raw = row.get(vehicle_type_col, idx) if vehicle_type_col is not None else idx
            vt = str(vt_raw)
            mv_raw = row.get('maxValue', None)
            mv     = (None if mv_raw is None or
                      (isinstance(mv_raw, float) and pd.isna(mv_raw))
                      else float(mv_raw))
            gs_raw = row.get('gravityStrength', 0)
            gs     = (0.0 if gs_raw is None or
                      (isinstance(gs_raw, float) and pd.isna(gs_raw))
                      else float(gs_raw))
            vehicles_data[vt] = dict(
                depth      = float(row['depth']),
                width      = float(row['width']),
                height     = float(row['height']),
                max_weight = float(row['maxWeight']),
                cost       = float(row['cost']),
                max_value  = mv,
                gravity    = gs,
            )

        return items_data, vehicles_data

    # ── greedy fallback ──────────────────────────────────────────────────────

    def _greedy_fallback(self, missing_ids, items_data, vehicles_data, vid_offset=0):
        """
        Place each item in *missing_ids* alone in the cheapest fitting vehicle.
        Used both as a parachute solution and to handle unpacked items post-ILP.
        """
        rows = []
        vtypes = sorted(vehicles_data, key=lambda v: vehicles_data[v]['cost'])
        vid    = vid_offset

        for iid in missing_ids:
            it = items_data[iid]
            placed = False
            for vt in vtypes:
                vd = vehicles_data[vt]
                if it['weight'] > vd['max_weight'] + _EPS:
                    continue
                if vd['max_value'] is not None and it['value'] > vd['max_value'] + _EPS:
                    continue
                for r in it['rots']:
                    dx, dy, dz = apply_rot(it['depth'], it['width'], it['height'], r)
                    if (dx <= vd['depth']  + _EPS and
                        dy <= vd['width']  + _EPS and
                        dz <= vd['height'] + _EPS):
                        rows.append(dict(
                            type_vehicle = vt,
                            idx_vehicle  = vid,
                            id_item      = iid,
                            x_origin     = 0.0,
                            y_origin     = 0.0,
                            z_origin     = 0.0,
                            orient       = r,
                        ))
                        vid    += 1
                        placed  = True
                        break
                if placed:
                    break

        return rows

    # ── solution assembly ────────────────────────────────────────────────────

    @staticmethod
    def _patterns_to_rows(patterns_selected: list) -> list:
        """Convert a list of selected pattern dicts into solution-row dicts."""
        rows    = []
        packed  = set()
        vid     = 0

        for pat in patterns_selected:
            added = False
            for it in pat['items']:
                if it['id'] not in packed:
                    rows.append(dict(
                        type_vehicle = pat['vtype'],
                        idx_vehicle  = vid,
                        id_item      = it['id'],
                        x_origin     = round(it['x'], 9),
                        y_origin     = round(it['y'], 9),
                        z_origin     = round(it['z'], 9),
                        orient       = it['orient'],
                    ))
                    packed.add(it['id'])
                    added = True
            if added:
                vid += 1

        return rows, packed

    # ── ILP set-covering ─────────────────────────────────────────────────────

    @staticmethod
    def _solve_ilp(patterns: list, item_ids: list, time_ms: int):
        """
        Build and solve a Set-Covering ILP with OR-Tools.

        min  Σ cost_p · x_p
        s.t. Σ_{p: i∈p} x_p ≥ 1   ∀i           (covering)
             x_p ∈ {0,1}

        Returns the list of selected pattern dicts, or None on failure.
        """
        solver = (pywraplp.Solver.CreateSolver('SCIP') or
                  pywraplp.Solver.CreateSolver('CBC'))
        if solver is None:
            return None

        solver.SetTimeLimit(time_ms)

        # Decision variables
        x = [solver.IntVar(0, 1, f'p{i}') for i in range(len(patterns))]

        # One covering constraint per item
        for iid in item_ids:
            ct = solver.Constraint(1, solver.infinity(), f'cov_{iid}')
            for i, pat in enumerate(patterns):
                if iid in pat['ids']:
                    ct.SetCoefficient(x[i], 1)

        # Objective
        obj = solver.Objective()
        for i, pat in enumerate(patterns):
            obj.SetCoefficient(x[i], pat['cost'])
        obj.SetMinimization()

        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return None

        return [pat for i, pat in enumerate(patterns) if x[i].solution_value() > 0.5]

    # ── main entry-point ─────────────────────────────────────────────────────

    def solve(self):
        """
        Orchestrate the full solve pipeline and write the solution CSV.
        Total wall-clock budget: 10 minutes.
        """
        t0 = time.time()

        # ── 1. Parse instance ────────────────────────────────────────────────
        items_data, vehicles_data = self._parse()
        item_ids = list(items_data)

        # ── 2. Compute an immediate greedy baseline (parachute) ──────────────
        #       Ensures a valid solution even if everything else fails.
        baseline_rows = self._greedy_fallback(item_ids, items_data, vehicles_data)

        # ── 3. Shared pattern pool + launch workers ──────────────────────────
        META_SEC = 480           # 8 minutes of exploration
        ILP_MS   = 70_000        # 70 seconds for ILP
        # Remaining overhead budget: 600 - 480 - 70 ≈ 50 sec (ample)

        mgr  = mp.Manager()
        pool = mgr.dict()

        # Pre-seed the pool with the greedy baseline patterns
        # (one bin per item — gives ILP something to start from)
        _greedy_seq = [(iid, items_data[iid]['rots']) for iid in item_ids]
        run_sequence(_greedy_seq, items_data, vehicles_data, pool)

        base_seed = int(t0 * 1000) & 0x7FFF_FFFF
        worker_fns = [worker_brkga, worker_alns, worker_grasp, worker_vns]
        procs = []

        for idx, fn in enumerate(worker_fns):
            p = mp.Process(
                target=fn,
                args=(items_data, vehicles_data,
                      META_SEC, t0, pool,
                      base_seed ^ (idx * 6_271)),
                daemon=True,
            )
            procs.append(p)
            p.start()

        # Wait for workers to finish (or kill them after their time budget)
        for p in procs:
            p.join(timeout=META_SEC + 15)
            if p.is_alive():
                p.terminate()
                p.join()

        # ── 4. Collect patterns ──────────────────────────────────────────────
        all_patterns = list(pool.values())
        elapsed      = time.time() - t0
        sol_rows     = []

        # ── 5. Set-Covering ILP ──────────────────────────────────────────────
        if all_patterns and _HAS_ORTOOLS:
            remaining_ms = max(10_000, int((580 - elapsed) * 1000))
            selected     = self._solve_ilp(all_patterns, item_ids,
                                           min(ILP_MS, remaining_ms))
            if selected is not None:
                sol_rows, packed = self._patterns_to_rows(selected)
            else:
                # ILP infeasible / timed out → greedy cover
                selected = greedy_cover(all_patterns, set(item_ids))
                sol_rows, packed = self._patterns_to_rows(selected)

        elif all_patterns:
            # OR-Tools not installed → greedy cover
            selected = greedy_cover(all_patterns, set(item_ids))
            sol_rows, packed = self._patterns_to_rows(selected)

        # ── 6. Patch any items missed by the ILP / greedy cover ───────────────
        missing = [iid for iid in item_ids if iid not in {r['id_item'] for r in sol_rows}]
        if missing:
            vid_off  = max((r['idx_vehicle'] for r in sol_rows), default=-1) + 1
            fallback = self._greedy_fallback(missing, items_data, vehicles_data,
                                             vid_offset=vid_off)
            sol_rows.extend(fallback)

        # ── 7. Use baseline if nothing better was found ───────────────────────
        if not sol_rows:
            sol_rows = baseline_rows

        # ── 8. Write solution ────────────────────────────────────────────────
        self.sol = pd.DataFrame(sol_rows)
        self.write_solution_to_file()