"""
solver_356856.py  ─  3-D Bin Packing  ·  Column Generation + Set Partition
════════════════════════════════════════════════════════════════════════════

═══ WHY THIS WINS ════════════════════════════════════════════════════════════

Standard solvers: FFD → local-search → hope.
This solver:      FFD/GRASP → harvest ALL bin-packings as "columns" →
                  solve SET PARTITION ILP (scipy HiGHS) → optimal selection.

The Set Partition formulation:
  • Variable  x_j ∈ {0,1}  = 1 if bin-packing j is used
  • Minimise  Σ  cost_j · x_j
  • s.t.      Σ  a_ij · x_j  =  1   ∀ item i   (each item in exactly one bin)

This is the textbook approach for bin-packing used in industrial solvers.
Given enough good columns (bins), the ILP is tight and finds the optimum.

═══ ALGORITHM ════════════════════════════════════════════════════════════════

Phase 1  (0 → 20 s)   Parallel construction: 8 FFD/BFD/GRASP workers.
                       Each found bin is stored as a column.

Phase 2  (20 → T-60 s) Intensive GRASP restarts on 4 threads.
                       Every restart produces new feasible bins → columns.
                       LNS is run on promising solutions to refine them.

Phase 3  (T-60 → T-15 s) Set Partition ILP via scipy.optimize.milp.
                       Uses HiGHS (bundled in scipy) — open source, no licence.
                       Time budget: up to 45 seconds.

Phase 4  (T-15 → T-3 s)  LNS polish on ILP solution.

═══ SUBMISSION ═══════════════════════════════════════════════════════════════
  1. Replace EVERY occurrence of XXXXXX with your student-ID (e.g. 123456).
  2. Rename file   →  solver_123456.py
  3. Rename folder →  solver_123456/
  4. Update __init__.py and main.py as per README.
"""

import os, sys, time, math, random, threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix
from .abstract_solver import AbstractSolver

# ─── AbstractSolver import (try several paths) ───────────────────────────────
_SD = os.path.dirname(os.path.abspath(__file__))
_RD = os.path.dirname(_SD)
for _p in (_RD, _SD):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ══════════════════════════════════════════════════════════════════════════════
#  ROTATION TABLE
#  Derived by reading results_checker.py → get_dims():
#    rotations[orient] = (w_out, d_out, h_out)
#    w_out → x-axis  (checked against vehicle["width"])
#    d_out → y-axis  (checked against vehicle["depth"])
#    h_out → z-axis  (checked against vehicle["height"])
# ══════════════════════════════════════════════════════════════════════════════

_ROT_IDX = [
    (0, 1, 2),   # 0: (w, d, h)
    (1, 0, 2),   # 1: (d, w, h)
    (2, 1, 0),   # 2: (h, d, w)
    (1, 2, 0),   # 3: (d, h, w)
    (0, 2, 1),   # 4: (w, h, d)
    (2, 0, 1),   # 5: (h, w, d)
]


def rotate(w, d, h, rot):
    t = _ROT_IDX[rot]; s = (w, d, h)
    return s[t[0]], s[t[1]], s[t[2]]


def unique_rots(w, d, h, allowed):
    seen, out = set(), []
    for r in allowed:
        dims = rotate(w, d, h, r)
        if dims not in seen:
            seen.add(dims); out.append((r, *dims))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  BIN3D  —  Extreme-Point placement
# ══════════════════════════════════════════════════════════════════════════════

class Bin3D:
    """
    Single physical container.
    Placement uses Extreme Points (Crainic et al. 2008):
      · EPs start at {(0,0,0)}.
      · Placing at (x,y,z)+(iw,id_,ih) adds  (x+iw,y,z), (x,y+id_,z), (x,y,z+ih).
      · EPs tried sorted (z asc, x asc, y asc) → bottom-up = gravity-safe.
    """
    _MAX_EPS = 400

    __slots__ = ('vtype','W','D','H','max_weight','max_value','gravity','cost',
                 'items','_boxes','weight','value','vol_used',
                 '_eps','_eps_set','_dirty')

    def __init__(self, vtype, W, D, H, mw, mv, grav, cost):
        self.vtype = vtype
        self.W = W; self.D = D; self.H = H
        self.max_weight = mw; self.max_value = mv
        self.gravity = grav; self.cost = cost
        self.items = []; self._boxes = []
        self.weight = self.value = self.vol_used = 0.0
        self._eps = [(0.,0.,0.)]; self._eps_set = {(0.,0.,0.)}; self._dirty = True

    # ── capacity ─────────────────────────────────────────────────────────────
    def cap_ok(self, w, v):
        return (w + self.weight <= self.max_weight + 1e-9 and
                v + self.value  <= self.max_value  + 1e-9)

    # ── geometry ──────────────────────────────────────────────────────────────
    def _overlaps(self, x, y, z, iw, id_, ih):
        x2,y2,z2 = x+iw, y+id_, z+ih
        for (bx1,by1,bz1,bx2,by2,bz2) in self._boxes:
            if x<bx2 and x2>bx1 and y<by2 and y2>by1 and z<bz2 and z2>bz1:
                return True
        return False

    def _grav_ok(self, x, y, z, iw, id_, ih):
        if self.gravity < 1e-9 or z < 1e-9: return True
        need = iw * id_ * self.gravity / 100.
        sup  = 0.; x2,y2 = x+iw, y+id_
        for (bx1,by1,bz1,bx2,by2,bz2) in self._boxes:
            if abs(bz2-z) < 1e-9:
                ox = min(x2,bx2)-max(x,bx1); oy = min(y2,by2)-max(y,by1)
                if ox>1e-12 and oy>1e-12:
                    sup += ox*oy
                    if sup >= need-1e-9: return True
        return False

    def _seps(self):
        if self._dirty:
            self._eps.sort(key=lambda p:(p[2],p[0],p[1])); self._dirty=False
        return self._eps

    # ── placement ────────────────────────────────────────────────────────────
    def find_ep(self, iw, id_, ih, weight, value):
        if not self.cap_ok(weight, value): return None
        for (ex,ey,ez) in self._seps():
            if ex+iw>self.W+1e-9 or ey+id_>self.D+1e-9 or ez+ih>self.H+1e-9: continue
            if self._overlaps(ex,ey,ez,iw,id_,ih): continue
            if not self._grav_ok(ex,ey,ez,iw,id_,ih): continue
            return ex,ey,ez
        return None

    def place(self, iid, x, y, z, iw, id_, ih, rot, wt, vl):
        self.items.append((iid,x,y,z,iw,id_,ih,rot))
        self._boxes.append((x,y,z, x+iw,y+id_,z+ih))
        self.weight+=wt; self.value+=vl; self.vol_used+=iw*id_*ih
        for ep in ((x+iw,y,z),(x,y+id_,z),(x,y,z+ih)):
            if ep[0]<self.W-1e-9 and ep[1]<self.D-1e-9 and ep[2]<self.H-1e-9 and ep not in self._eps_set:
                self._eps.append(ep); self._eps_set.add(ep); self._dirty=True
        if len(self._eps)>self._MAX_EPS:
            self._eps.sort(key=lambda p:(p[2],p[0],p[1]))
            self._eps=self._eps[:self._MAX_EPS]; self._eps_set=set(self._eps); self._dirty=False

    def try_add(self, item):
        if not self.cap_ok(item['weight'],item['value']): return False
        for (rot,iw,id_,ih) in item['urots']:
            if iw>self.W+1e-9 or id_>self.D+1e-9 or ih>self.H+1e-9: continue
            pos = self.find_ep(iw,id_,ih,item['weight'],item['value'])
            if pos:
                self.place(item['id'],pos[0],pos[1],pos[2],iw,id_,ih,rot,item['weight'],item['value'])
                return True
        return False

    def rem_vol(self): return self.W*self.D*self.H - self.vol_used

    def copy(self):
        b = Bin3D(self.vtype,self.W,self.D,self.H,
                  self.max_weight,self.max_value,self.gravity,self.cost)
        b.items=list(self.items); b._boxes=list(self._boxes)
        b.weight=self.weight; b.value=self.value; b.vol_used=self.vol_used
        b._eps=list(self._eps); b._eps_set=set(self._eps_set); b._dirty=self._dirty
        return b


# ══════════════════════════════════════════════════════════════════════════════
#  VEHICLE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def item_fits(item, v):
    for (_,iw,id_,ih) in item['urots']:
        if iw<=v['W']+1e-9 and id_<=v['D']+1e-9 and ih<=v['H']+1e-9: return True
    return False

def open_bin(item, vehicles):
    for v in vehicles:
        if item['weight']>v['max_weight']+1e-9: continue
        if item['value'] >v['max_value'] +1e-9: continue
        if not item_fits(item,v): continue
        b=Bin3D(v['type'],v['W'],v['D'],v['H'],v['max_weight'],v['max_value'],v['gravity'],v['cost'])
        if b.try_add(item): return b
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  PACKING  —  First-Fit / Best-Fit
# ══════════════════════════════════════════════════════════════════════════════

def pack(seq, vehicles, best_fit=False, t_end=float('inf')):
    bins, unp = [], []
    for item in seq:
        if time.monotonic()>t_end: unp.append(item); continue
        chosen=None; best_rem=float('inf')
        for b in bins:
            if not b.cap_ok(item['weight'],item['value']): continue
            if best_fit:
                for (_,iw,id_,ih) in item['urots']:
                    if iw>b.W+1e-9 or id_>b.D+1e-9 or ih>b.H+1e-9: continue
                    if b.find_ep(iw,id_,ih,item['weight'],item['value']):
                        rem=b.rem_vol()-iw*id_*ih
                        if rem<best_rem: best_rem=rem; chosen=b
                        break
            else:
                if b.try_add(item): chosen=b; break
        if chosen:
            if best_fit: chosen.try_add(item)
        else:
            nb=open_bin(item,vehicles)
            if nb: bins.append(nb)
            else:  unp.append(item)
    return bins, unp


# ══════════════════════════════════════════════════════════════════════════════
#  COLUMN POOL  —  thread-safe collection of candidate bin packings
# ══════════════════════════════════════════════════════════════════════════════

class ColumnPool:
    """
    Stores all discovered feasible bin packings as "columns".
    Deduplicates by frozenset of item IDs (same set = same column value).
    Thread-safe.
    """
    def __init__(self):
        self._lock  = threading.Lock()
        self._cols  = []          # list of Bin3D
        self._seen  = set()       # frozenset(item_ids) already stored

    # Hard cap: once we have many columns, only add if they look promising
    _MAX_POOL = 3000

    def add_bin(self, b: Bin3D):
        key = frozenset(r[0] for r in b.items)
        if not key: return
        with self._lock:
            if key not in self._seen:
                if len(self._cols) >= self._MAX_POOL:
                    # Replace a random existing column if new one is cheaper
                    import random as _rnd
                    worst_idx = max(range(len(self._cols)),
                                   key=lambda i: self._cols[i].cost)
                    if b.cost >= self._cols[worst_idx].cost:
                        return
                    old_key = frozenset(r[0] for r in self._cols[worst_idx].items)
                    self._seen.discard(old_key)
                    self._cols[worst_idx] = b.copy()
                    self._seen.add(key)
                else:
                    self._seen.add(key)
                    self._cols.append(b.copy())

    def add_solution(self, bins):
        for b in bins: self.add_bin(b)

    def get_columns(self):
        with self._lock:
            return list(self._cols)

    def size(self):
        with self._lock: return len(self._cols)


# ══════════════════════════════════════════════════════════════════════════════
#  LNS  —  Large Neighbourhood Search
# ══════════════════════════════════════════════════════════════════════════════

def _cost(bins): return sum(b.cost for b in bins)

def _repack(to_place, existing, vehicles, t_end):
    """Try to fit *to_place* into *existing* (already copied). Returns (bins, success)."""
    work = [b.copy() for b in existing]
    for item in sorted(to_place, key=lambda x:-x['vol']):
        if time.monotonic()>t_end: return work, False
        placed=False
        for b in work:
            if b.try_add(item): placed=True; break
        if not placed:
            nb=open_bin(item,vehicles)
            if nb: work.append(nb)
            else:  return work, False
    return work, _cost(work)<=_cost(existing)+1e-9

def op_elim(bins, ilookup, vehicles, t_end):
    if len(bins)<=1: return bins,False
    order=sorted(range(len(bins)),key=lambda i:len(bins[i].items))
    for idx in order[:min(8,len(bins))]:
        if time.monotonic()>t_end: break
        to_move=[ilookup[r[0]] for r in bins[idx].items]
        others=[bins[i] for i in range(len(bins)) if i!=idx]
        new,ok=_repack(to_move,others,vehicles,t_end)
        if ok and _cost(new)<_cost(bins)-1e-9: return new,True
    return bins,False

def op_shake(bins, ilookup, vehicles, rng, t_end):
    if len(bins)<2: return bins,False
    i1,i2=rng.sample(range(len(bins)),2)
    pool=sorted([ilookup[r[0]] for r in bins[i1].items]+
                [ilookup[r[0]] for r in bins[i2].items], key=lambda x:-x['vol'])
    for v in vehicles:
        if time.monotonic()>t_end: break
        nb=Bin3D(v['type'],v['W'],v['D'],v['H'],v['max_weight'],v['max_value'],v['gravity'],v['cost'])
        if all(nb.try_add(it) for it in pool):
            if nb.cost<bins[i1].cost+bins[i2].cost-1e-9:
                rest=[bins[k] for k in range(len(bins)) if k!=i1 and k!=i2]
                return rest+[nb],True
    return bins,False

def op_eject(bins, ilookup, vehicles, rng, t_end):
    if len(bins)<2: return bins,False
    bi=rng.randrange(len(bins)); b=bins[bi]
    if not b.items: return bins,False
    n=max(1,len(b.items)//3)
    eject_recs=rng.sample(b.items,min(n,len(b.items)))
    eject_ids={r[0] for r in eject_recs}
    eject_items=[ilookup[r[0]] for r in eject_recs]
    src=Bin3D(b.vtype,b.W,b.D,b.H,b.max_weight,b.max_value,b.gravity,b.cost)
    for rec in b.items:
        if rec[0] not in eject_ids: src.try_add(ilookup[rec[0]])
    work=[bins[i].copy() for i in range(len(bins)) if i!=bi]
    work.insert(bi,src)
    others=[work[i] for i in range(len(work)) if i!=bi]
    ok=True
    for it in sorted(eject_items,key=lambda x:-x['vol']):
        if time.monotonic()>t_end: return bins,False
        placed=any(ob.try_add(it) for ob in others)
        if not placed: ok=False; break
    if ok:
        result=[work[i] for i in range(len(work)) if i!=bi] if not src.items else work
        if _cost(result)<_cost(bins)-1e-9: return result,True
    return bins,False

def lns(bins, ilookup, vehicles, t_end, rng=None, verbose=False, pool=None):
    if rng is None: rng=random.Random()
    cur=list(bins); stag=0
    while time.monotonic()<t_end:
        imp=False
        new,ok=op_elim(cur,ilookup,vehicles,t_end)
        if ok:
            cur=new; stag=0; imp=True
            if pool: pool.add_solution(cur)
            if verbose: print(f'    [ELIM]  bins={len(cur):3d}  cost={_cost(cur):.2f}')
            continue
        new,ok=op_shake(cur,ilookup,vehicles,rng,t_end)
        if ok:
            cur=new; stag=0; imp=True
            if pool: pool.add_solution(cur)
            if verbose: print(f'    [SHAKE] bins={len(cur):3d}  cost={_cost(cur):.2f}')
        new,ok=op_eject(cur,ilookup,vehicles,rng,t_end)
        if ok:
            cur=new; stag=0; imp=True
            if pool: pool.add_solution(cur)
            if verbose: print(f'    [EJECT] bins={len(cur):3d}  cost={_cost(cur):.2f}')
        if not imp:
            stag+=1
            if stag>=40: break
    return cur


# ══════════════════════════════════════════════════════════════════════════════
#  GRASP construction
# ══════════════════════════════════════════════════════════════════════════════

def grasp(items, vehicles, alpha, rng, t_end, best_fit=False):
    scored=sorted(items,key=lambda x:-x['vol'])
    if alpha>1e-9:
        n=len(scored); nd=max(1,int(n*(1-alpha)))
        tail=scored[nd:]; rng.shuffle(tail); scored=scored[:nd]+tail
    return pack(scored,vehicles,best_fit=best_fit,t_end=t_end)


# ══════════════════════════════════════════════════════════════════════════════
#  SET PARTITION  —  scipy HiGHS MILP  (with LP-guided column filtering)
# ══════════════════════════════════════════════════════════════════════════════

def _build_coverage_matrix(columns, all_item_ids):
    """Build sparse coverage matrix A and cost vector. Returns (A, costs, item_idx)."""
    n_items  = len(all_item_ids)
    n_cols   = len(columns)
    item_idx = {iid: i for i, iid in enumerate(all_item_ids)}
    rows_r, cols_r = [], []
    costs = np.empty(n_cols)
    for j, b in enumerate(columns):
        costs[j] = b.cost
        for (iid, *_) in b.items:
            if iid in item_idx:
                rows_r.append(item_idx[iid])
                cols_r.append(j)
    A = csc_matrix(
        (np.ones(len(rows_r), dtype=np.float64), (rows_r, cols_r)),
        shape=(n_items, n_cols))
    return A, costs, item_idx


def _filter_columns_lp(columns, all_item_ids, max_cols=700, lp_budget=8.):
    """
    Reduce column pool to max_cols using a two-step strategy:
      1. LP Set-Cover relaxation (>= 1, not = 1) to identify fractionally
         useful columns — these span the binding LP basis.
      2. Mandatory inclusion of the cheapest column per item (ensures every
         item remains coverable after filtering).
      3. Fill to max_cols by cheapest cost-per-covered-item ratio.
    """
    if len(columns) <= max_cols:
        return columns

    n_items = len(all_item_ids)
    A, costs, item_idx = _build_coverage_matrix(columns, all_item_ids)

    keep = set()

    # Step 1: LP relaxation with INEQUALITY (set cover — always feasible)
    try:
        lp = milp(
            costs,
            constraints=LinearConstraint(A,
                                         lb=np.ones(n_items),
                                         ub=np.full(n_items, np.inf)),
            bounds=Bounds(lb=np.zeros(len(columns)), ub=np.ones(len(columns))),
            integrality=np.zeros(len(columns)),
            options={'time_limit': lp_budget, 'disp': False}
        )
        if lp.success and lp.x is not None:
            for j, xj in enumerate(lp.x):
                if xj > 1e-8:
                    keep.add(j)
    except Exception:
        pass

    # Step 2: Mandatory — cheapest column per item (guarantees coverage)
    n_items_col = np.array(A.sum(axis=0)).flatten()
    # For each item i, find the column j with max items (best fill) covering i
    A_arr = A.toarray()   # small: max_cols x n_items
    for i in range(n_items):
        covering = np.where(A_arr[i, :] > 0.5)[0]
        if covering.size == 0:
            continue
        # Prefer cheapest, break ties by most items covered
        best_j = covering[np.argmin(costs[covering])]
        keep.add(int(best_j))

    # Step 3: Fill remainder by cost/n_items_covered ratio
    score  = costs / np.maximum(n_items_col, 1.)
    ranked = np.argsort(score)
    for j in ranked:
        if len(keep) >= max_cols:
            break
        keep.add(int(j))

    return [columns[j] for j in sorted(keep)]


def solve_set_partition(columns, all_item_ids, t_budget=45.0):
    """
    Solve the Set Partition ILP:
      min  Σ cost_j · x_j
      s.t. Σ a_ij · x_j = 1  ∀ i   (each item in exactly one bin)
           x_j ∈ {0,1}

    Returns (list[Bin3D], float) — or (None, inf) on failure.
    """
    if not columns:
        return None, float('inf')

    all_set = set(all_item_ids)

    # ── Filter to tractable size via LP ───────────────────────────────────
    lp_budget = min(8., t_budget * 0.18)
    columns   = _filter_columns_lp(columns, all_item_ids,
                                   max_cols=700, lp_budget=lp_budget)

    # ── Check every item is covered by at least one column ────────────────
    A, costs, item_idx = _build_coverage_matrix(columns, all_item_ids)
    row_sums = np.array(A.sum(axis=1)).flatten()
    if row_sums.min() < 0.5:
        return None, float('inf')   # some item uncoverable in this pool

    n_items = len(all_item_ids)
    n_cols  = len(columns)

    constraints = LinearConstraint(A,
                                   lb=np.ones(n_items),
                                   ub=np.ones(n_items))
    bounds      = Bounds(lb=np.zeros(n_cols), ub=np.ones(n_cols))
    integrality = np.ones(n_cols)

    milp_budget = max(5., t_budget - lp_budget - 1.)

    try:
        result = milp(
            costs,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': milp_budget, 'disp': False,
                     'mip_rel_gap': 0.005}
        )

        # result.x is None when HiGHS times out before finding any integer solution
        if result.x is None:
            return None, float('inf')

        if result.status in (0, 1):   # 0=optimal, 1=time-limited with solution
            selected = [columns[j] for j in range(n_cols) if result.x[j] > 0.5]
            covered  = set()
            for b in selected:
                for (iid, *_) in b.items:
                    covered.add(iid)
            if covered == all_set:
                return selected, float(result.fun)

    except Exception as e:
        print(f'  [MILP] error: {e}')

    return None, float('inf')


# ══════════════════════════════════════════════════════════════════════════════
#  TARGETED COLUMN GENERATION  —  item-centric bin filling
# ══════════════════════════════════════════════════════════════════════════════

def generate_columns_for_item(seed_item, items, vehicles, n_cols,
                               rng, t_end):
    """
    Generate *n_cols* diverse bin-packings that all contain *seed_item*.
    This enriches the column pool for items that are hard to cover.
    Returns list of Bin3D.
    """
    out = []
    others = [it for it in items if it['id'] != seed_item['id']]

    for _ in range(n_cols):
        if time.monotonic() > t_end: break
        for v in vehicles:
            b = Bin3D(v['type'],v['W'],v['D'],v['H'],
                      v['max_weight'],v['max_value'],v['gravity'],v['cost'])
            if not b.try_add(seed_item): continue
            shuffled = list(others); rng.shuffle(shuffled)
            for it in shuffled:
                if not b.cap_ok(it['weight'],it['value']): continue
                b.try_add(it)
            out.append(b)
            break

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  DATA PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_items(df):
    out=[]
    for iid,row in df.iterrows():
        rots=[int(c) for c in str(row['allowedRotations'])]
        w,d,h=float(row['width']),float(row['depth']),float(row['height'])
        out.append({'id':iid,'w':w,'d':d,'h':h,
                    'weight':float(row['weight']),'value':float(row['value']),
                    'urots':unique_rots(w,d,h,rots),
                    'vol':w*d*h,'maxdim':max(w,d,h)})
    return out

def parse_vehicles(df):
    out=[]
    for vtype,row in df.iterrows():
        mv=row.get('maxValue',float('nan'))
        if mv is None or (isinstance(mv,float) and math.isnan(mv)): mv=float('inf')
        out.append({'type':vtype,
                    'W':float(row['width']),'D':float(row['depth']),'H':float(row['height']),
                    'max_weight':float(row['maxWeight']),'max_value':float(mv),
                    'cost':float(row['cost']),'gravity':float(row['gravityStrength']),
                    'vol':float(row['width'])*float(row['depth'])*float(row['height'])})
    out.sort(key=lambda v:v['cost'])   # cheapest first
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SOLVER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class solver_356856(AbstractSolver):
    """
    3-D Bin Packing: EP + GRASP + LNS + Column Generation + Set Partition MILP.
    ► Replace XXXXXX with your student ID before submission. ◄
    """

    SOLVE_SECONDS = 545
    N_THREADS     = 4
    VERBOSE       = True

    def __init__(self, inst):
        super().__init__(inst)
        self.name = 'solver_356856'

    def solve(self):
        t0   = time.monotonic()
        tend = t0 + self.SOLVE_SECONDS
        log  = print if self.VERBOSE else (lambda *a,**k:None)

        log(f"\n{'═'*65}")
        log(f"  Dataset  :  {self.inst.name}")
        log(f"{'═'*65}")

        items    = parse_items(self.inst.df_items)
        vehicles = parse_vehicles(self.inst.df_vehicles)
        ilookup  = {it['id']:it for it in items}
        all_ids  = sorted(ilookup.keys())

        log(f"  Items    : {len(items)}")
        for v in vehicles:
            mv = f"{v['max_value']:.0f}" if v['max_value']<1e14 else "∞"
            log(f"  Vehicle  : {v['type']:14s}  {v['W']}×{v['D']}×{v['H']}"
                f"  maxW={v['max_weight']:.0f}  maxV={mv}"
                f"  cost={v['cost']:.2f}  grav={v['gravity']}%")

        # Shared state
        pool      = ColumnPool()
        best_bins = [None]
        best_cost = [float('inf')]
        lock      = threading.Lock()

        def update_best(bins, label=''):
            c=_cost(bins)
            with lock:
                if c<best_cost[0]:
                    best_cost[0]=c; best_bins[0]=[b.copy() for b in bins]
                    log(f'    ★ NEW BEST {label}: cost={c:.2f}  bins={len(bins)}')
                    return True
            return False

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 1 — Parallel construction (8 workers)
        # ─────────────────────────────────────────────────────────────────────
        log(f"\n  Phase 1  —  parallel construction")
        t_p1 = t0 + min(20.0, self.SOLVE_SECONDS*0.04)

        by_vol    = sorted(items, key=lambda x:-x['vol'])
        by_weight = sorted(items, key=lambda x:(-x['weight'],-x['vol']))
        by_maxdim = sorted(items, key=lambda x:(-x['maxdim'],-x['vol']))
        by_value  = sorted(items, key=lambda x:(-x['value'],-x['vol']))

        def _worker(seq, bf, alpha, seed):
            rng=random.Random(seed)
            bins,unp=grasp(seq,vehicles,alpha,rng,t_p1,bf)
            if unp: return None,float('inf')
            bins=lns(bins,ilookup,vehicles,t_p1,rng,verbose=False,pool=pool)
            pool.add_solution(bins)
            return bins,_cost(bins)

        configs=[
            (by_vol,    False, 0.00, 1001),(by_vol,    True,  0.00, 1002),
            (by_weight, False, 0.00, 1003),(by_maxdim, False, 0.00, 1004),
            (by_value,  False, 0.00, 1005),(by_value,  True,  0.00, 1006),
            (items,     False, 0.12, 1007),(items,     True,  0.12, 1008),
        ]

        with ThreadPoolExecutor(max_workers=self.N_THREADS) as ex:
            fs={ex.submit(_worker,seq,bf,alpha,seed):seed
                for (seq,bf,alpha,seed) in configs}
            for f in as_completed(fs):
                try:
                    bins,c=f.result()
                    if bins: update_best(bins,f'p1-{fs[f]}')
                except Exception as e:
                    log(f'    p1 worker error: {e}')

        if best_bins[0] is None:
            log('  WARN: emergency fallback — 1 item per bin')
            fb=[]
            for it in items:
                b=open_bin(it,vehicles)
                if b: fb.append(b); pool.add_bin(b)
            best_bins[0]=fb; best_cost[0]=_cost(fb)

        log(f"\n  After Phase 1: cost={best_cost[0]:.2f}  bins={len(best_bins[0])}"
            f"  columns={pool.size()}")

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 2 — LNS + GRASP restarts (columns harvested into pool)
        # ─────────────────────────────────────────────────────────────────────
        log(f"\n  Phase 2  —  GRASP restarts + LNS  (column harvesting)")
        t_p2 = tend - 65.0
        restarts=[0]

        def _deep_lns():
            with lock: snap=[b.copy() for b in best_bins[0]]
            rng=random.Random(42)
            improved=lns(snap,ilookup,vehicles,t_p2,rng,
                         verbose=self.VERBOSE,pool=pool)
            update_best(improved,'lns-deep')

        def _restart_worker(seed_base):
            rng=random.Random(seed_base)
            while time.monotonic()<t_p2:
                alpha=rng.uniform(0.05,0.35); bf=rng.random()<0.5
                seq=list(items); rng.shuffle(seq)
                bins,unp=grasp(seq,vehicles,alpha,rng,t_p2,bf)
                if unp: restarts[0]+=1; continue
                pool.add_solution(bins)
                with lock: thresh=best_cost[0]*1.08
                if _cost(bins)<=thresh:
                    tl=min(time.monotonic()+30.,t_p2)
                    bins=lns(bins,ilookup,vehicles,tl,rng,verbose=False,pool=pool)
                update_best(bins,'restart')
                restarts[0]+=1

        with ThreadPoolExecutor(max_workers=self.N_THREADS) as ex:
            futures=[ex.submit(_deep_lns)]
            for k in range(self.N_THREADS-1):
                futures.append(ex.submit(_restart_worker,3000+k*1337))
            for f in as_completed(futures):
                try: f.result()
                except Exception as e: log(f'  Phase 2 error: {e}')

        log(f"\n  After Phase 2: cost={best_cost[0]:.2f}  bins={len(best_bins[0])}"
            f"  columns={pool.size()}  restarts={restarts[0]}")

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 3 — Targeted column generation + Set Partition MILP
        # ─────────────────────────────────────────────────────────────────────
        log(f"\n  Phase 3  —  Set Partition ILP  ({int(tend-time.monotonic())} s left)")

        # Add item-centric columns (ensure each item has good solo-bin options)
        rng_cg=random.Random(5555)
        t_cg  =min(time.monotonic()+20., tend-45.)
        for item in sorted(items, key=lambda x:-x['vol']):
            if time.monotonic()>t_cg: break
            for b in generate_columns_for_item(item,items,vehicles,
                                               8,rng_cg,t_cg):
                pool.add_bin(b)

        cols = pool.get_columns()
        log(f"  Column pool size: {len(cols)}")

        milp_budget = min(45., tend - time.monotonic() - 15.)
        if milp_budget > 5. and cols:
            log(f"  Running MILP  (budget={milp_budget:.0f}s) ...")
            selected, milp_cost = solve_set_partition(cols, all_ids, milp_budget)
            if selected is not None:
                log(f"  MILP solution: cost={milp_cost:.2f}  bins={len(selected)}")
                update_best(selected,'milp')
            else:
                log('  MILP: no feasible solution found in column pool')
        else:
            log(f'  MILP skipped (budget too small or no columns)')

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 4 — Final LNS polish
        # ─────────────────────────────────────────────────────────────────────
        remaining=int(tend-time.monotonic())
        log(f"\n  Phase 4  —  final LNS polish  ({remaining} s left)")
        with lock: snap=[b.copy() for b in best_bins[0]]
        final=lns(snap,ilookup,vehicles,tend-2,random.Random(9999),
                  verbose=self.VERBOSE,pool=pool)
        update_best(final,'final-lns')

        # ─────────────────────────────────────────────────────────────────────
        # Build output
        # ─────────────────────────────────────────────────────────────────────
        with lock: result=best_bins[0]; total=best_cost[0]
        elapsed=time.monotonic()-t0

        log(f"\n{'═'*65}")
        log(f"  FINAL  cost={total:.2f}  bins={len(result)}  time={elapsed:.1f}s")
        log(f"{'═'*65}\n")

        idx_v=0
        for b in result:
            for (iid,x,y,z,iw,id_,ih,orient) in b.items:
                self.sol['type_vehicle'].append(b.vtype)
                self.sol['idx_vehicle'].append(idx_v)
                self.sol['id_item'].append(iid)
                self.sol['x_origin'].append(round(x,9))
                self.sol['y_origin'].append(round(y,9))
                self.sol['z_origin'].append(round(z,9))
                self.sol['orient'].append(orient)
            idx_v+=1

        self.write_solution_to_file()
        log(f"  → results/sol_{self.inst.name}_{self.name}.csv")
