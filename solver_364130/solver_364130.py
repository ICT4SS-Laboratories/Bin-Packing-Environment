"""
solver_364130.py
================
Heuristic 3D bin packing focused on:
1) always returning feasible placements,
2) minimizing container cost within 10 minutes,
3) combining fast constructive heuristics with exact final selection.

Pipeline:
- build many diverse feasible bins (columns),
- improve them with destroy/repair neighborhoods,
- pick the cheapest exact cover of items via set partition MILP.
"""
import os, time, math, random, threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix
from .abstract_solver import AbstractSolver

try:
    import highspy
    _HAS_HIGHSPY = True
except Exception:
    highspy = None
    _HAS_HIGHSPY = False

try:
    from ortools.sat.python import cp_model
    _HAS_ORTOOLS = True
except Exception:
    cp_model = None
    _HAS_ORTOOLS = False


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
    _MAX_EPS = 700

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
        # Six-EP variant: original three axis-aligned EPs plus three diagonal
        # corner EPs (Crainic et al. 2008 full variant). Catches placements
        # at the corners between two stacked surfaces that the 3-EP pruning
        # would miss.
        for ep in ((x+iw, y,     z),
                   (x,     y+id_, z),
                   (x,     y,     z+ih),
                   (x+iw, y+id_, z),
                   (x+iw, y,     z+ih),
                   (x,     y+id_, z+ih)):
            if ep[0]<self.W-1e-9 and ep[1]<self.D-1e-9 and ep[2]<self.H-1e-9 and ep not in self._eps_set:
                self._eps.append(ep); self._eps_set.add(ep); self._dirty=True
        if len(self._eps)>self._MAX_EPS:
            self._eps.sort(key=lambda p:(p[2],p[0],p[1]))
            self._eps=self._eps[:self._MAX_EPS]; self._eps_set=set(self._eps); self._dirty=False

    def try_add(self, item):
        if not self.cap_ok(item['weight'],item['value']): return False
        best = None
        best_score = None
        for (rot,iw,id_,ih) in item['urots']:
            if iw>self.W+1e-9 or id_>self.D+1e-9 or ih>self.H+1e-9: continue
            pos = self.find_ep(iw,id_,ih,item['weight'],item['value'])
            if pos:
                x, y, z = pos
                rem_x = self.W - (x + iw)
                rem_y = self.D - (y + id_)
                rem_z = self.H - (z + ih)
                # Prefer bottom placements, then tighter residual space.
                score = (z, rem_x + rem_y + 0.20 * rem_z, x + y, -(iw * id_))
                if best_score is None or score < best_score:
                    best_score = score
                    best = (rot, x, y, z, iw, id_, ih)
        if best is None:
            return False
        rot, x, y, z, iw, id_, ih = best
        self.place(item['id'], x, y, z, iw, id_, ih, rot, item['weight'], item['value'])
        return True

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

def _vehicle_accepts_item(item, v):
    if item['weight'] > v['max_weight'] + 1e-9:
        return False
    if item['value'] > v['max_value'] + 1e-9:
        return False
    return item_fits(item, v)

def open_bin(item, vehicles, top_k=8):
    # Bounded candidate scan improves stability on large instances while
    # still allowing harder items to explore a wider set of vehicles.
    k_eff = top_k
    if item.get('hardness', 0.0) >= 2.2:
        k_eff = max(top_k, 12)
    cand = []
    for v in vehicles:
        if _vehicle_accepts_item(item, v):
            cand.append(v)
            if len(cand) >= k_eff:
                break
    if not cand:
        return None
    best, best_score = None, None
    for v in cand:
        b = Bin3D(v['type'], v['W'], v['D'], v['H'],
                  v['max_weight'], v['max_value'], v['gravity'], v['cost'])
        if not b.try_add(item):
            continue
        vol_cap = max(1.0, b.W * b.D * b.H)
        rem_vol = (vol_cap - b.vol_used) / vol_cap
        rem_w = (b.max_weight - b.weight) / max(1.0, b.max_weight)
        if b.max_value >= 1e18:
            rem_val = 0.0
        else:
            rem_val = (b.max_value - b.value) / max(1.0, b.max_value)
        score = (b.cost, 0.55 * rem_w + 0.30 * rem_vol + 0.15 * rem_val)
        if best_score is None or score < best_score:
            best_score = score
            best = b
    if best is not None:
        return best
    for v in vehicles:
        if not _vehicle_accepts_item(item, v):
            continue
        b = Bin3D(v['type'], v['W'], v['D'], v['H'],
                  v['max_weight'], v['max_value'], v['gravity'], v['cost'])
        if b.try_add(item):
            return b
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
        self._idx   = {}          # key=frozenset(item_ids) -> index in _cols

    # Hard cap: once we have many columns, only add if they look promising
    _MAX_POOL = 7000

    def add_bin(self, b: Bin3D):
        key = frozenset(r[0] for r in b.items)
        if not key: return
        cand = b.copy()
        with self._lock:
            old_idx = self._idx.get(key)
            if old_idx is not None:
                # Same item-set: keep the cheapest known column.
                if cand.cost + 1e-9 < self._cols[old_idx].cost:
                    self._cols[old_idx] = cand
                return

            if len(self._cols) < self._MAX_POOL:
                self._idx[key] = len(self._cols)
                self._cols.append(cand)
                return

            # Full pool: replace the most expensive column only if improved.
            worst_idx = max(range(len(self._cols)), key=lambda i: self._cols[i].cost)
            if cand.cost >= self._cols[worst_idx].cost - 1e-9:
                return

            old_key = frozenset(r[0] for r in self._cols[worst_idx].items)
            self._cols[worst_idx] = cand
            self._idx.pop(old_key, None)
            self._idx[key] = worst_idx

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

def op_eject(bins, ilookup, vehicles, rng, t_end, destroy_rate=0.33):
    if len(bins)<2: return bins,False
    bi=rng.randrange(len(bins)); b=bins[bi]
    if not b.items: return bins,False
    n=max(1, int(len(b.items) * destroy_rate))
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

def op_relocate(bins, ilookup, vehicles, rng, t_end):
    """Try to fully empty worst bin, relocating its items into others. If all relocate, bin removed -> cost savings."""
    if len(bins) < 2:
        return bins, False
    def _quality(b):
        util = b.weight / max(1.0, b.max_weight) + b.vol_used / max(1.0, b.W*b.D*b.H)
        return (util / max(1.0, b.cost), len(b.items))
    order = sorted(range(len(bins)), key=lambda i: _quality(bins[i]))
    n_try = min(8, len(order))
    for ai in order[:n_try]:
        if time.monotonic() > t_end:
            return bins, False
        a = bins[ai]
        if not a.items:
            new = [bins[k].copy() for k in range(len(bins)) if k != ai]
            return new, True
        items_in_a = [ilookup[r[0]] for r in a.items]
        items_in_a.sort(key=lambda x: -x['vol'])
        others = [bins[i].copy() for i in range(len(bins)) if i != ai]
        ok = True
        for it in items_in_a:
            if time.monotonic() > t_end:
                return bins, False
            placed = False
            for bo in sorted(others, key=lambda bb: bb.rem_vol()):
                if bo.try_add(it):
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok and _cost(others) < _cost(bins) - 1e-9:
            return others, True
    return bins, False

def op_swap_pair(bins, ilookup, vehicles, rng, t_end):
    """Pairwise item swaps between bins. Accepts only strict cost decreases."""
    if len(bins) < 2:
        return bins, False
    n = len(bins)
    pairs_tried = 0
    max_pairs = 12 if n < 30 else (16 if n < 100 else 20)
    expensive = sorted(range(n), key=lambda i: bins[i].cost, reverse=True)
    for ai in expensive[:min(6, n)]:
        if pairs_tried >= max_pairs:
            break
        bi_choices = [i for i in range(n) if i != ai]
        rng.shuffle(bi_choices)
        for bi in bi_choices[:min(4, len(bi_choices))]:
            if time.monotonic() > t_end:
                return bins, False
            pairs_tried += 1
            a, b = bins[ai], bins[bi]
            if not a.items or not b.items:
                continue
            ra_pool = rng.sample(a.items, min(6, len(a.items)))
            rb_pool = rng.sample(b.items, min(6, len(b.items)))
            for ra in ra_pool:
                ia = ilookup[ra[0]]
                a_new = Bin3D(a.vtype,a.W,a.D,a.H,a.max_weight,a.max_value,a.gravity,a.cost)
                for r2 in a.items:
                    if r2[0] != ra[0]:
                        a_new.try_add(ilookup[r2[0]])
                for rb in rb_pool:
                    if time.monotonic() > t_end:
                        return bins, False
                    if ra[0] == rb[0]:
                        continue
                    ib = ilookup[rb[0]]
                    b_new = Bin3D(b.vtype,b.W,b.D,b.H,b.max_weight,b.max_value,b.gravity,b.cost)
                    for r2 in b.items:
                        if r2[0] != rb[0]:
                            b_new.try_add(ilookup[r2[0]])
                    if not a_new.try_add(ib):
                        continue
                    if not b_new.try_add(ia):
                        continue
                    test_bins = [bins[k].copy() for k in range(n)]
                    test_bins[ai] = a_new
                    test_bins[bi] = b_new
                    if _cost(test_bins) < _cost(bins) - 1e-9:
                        return test_bins, True
    return bins, False


def op_consolidate_pair(bins, ilookup, vehicles, rng, t_end):
    """
    Deterministic pairwise bin merge.
    Scans pairs (i, j) ordered by combined cost (most expensive first) and
    attempts to repack their items into ONE bin across ALL vehicle types.
    Accepts the first strict-improvement.

    More systematic than op_shake (which samples randomly) and complements
    op_merge3 (which targets triples).
    """
    n = len(bins)
    if n < 2:
        return bins, False

    # Order bins by cost desc — focus on expensive bins first
    ranked = sorted(range(n), key=lambda i: -bins[i].cost)
    head = ranked[:min(12, n)]  # cap workload

    # Build candidate pairs (i, j) with j != i
    pairs = []
    for ai in head:
        for bj in ranked:
            if bj == ai:
                continue
            pair = (min(ai, bj), max(ai, bj))
            pairs.append(pair)
    seen_pairs = set()
    uniq_pairs = []
    for p in pairs:
        if p not in seen_pairs:
            seen_pairs.add(p)
            uniq_pairs.append(p)
    # Score by combined cost desc — expensive merges have biggest payoff
    uniq_pairs.sort(key=lambda p: -(bins[p[0]].cost + bins[p[1]].cost))
    uniq_pairs = uniq_pairs[:min(40, len(uniq_pairs))]

    for ai, bj in uniq_pairs:
        if time.monotonic() > t_end:
            return bins, False
        a = bins[ai]
        b = bins[bj]
        old_cost = a.cost + b.cost
        ids = list(dict.fromkeys([rec[0] for rec in a.items] + [rec[0] for rec in b.items]))
        items = [ilookup[iid] for iid in ids]
        # Quick capacity-feasibility filter: total weight/value/vol
        tw = sum(it['weight'] for it in items)
        tv = sum(it['value'] for it in items)
        tvol = sum(it['vol'] for it in items)
        # Try cheapest viable vehicle that fits totals
        viable = []
        for v in vehicles:
            if v['cost'] >= old_cost - 1e-9:
                continue
            if tw > v['max_weight'] + 1e-9 or tv > v['max_value'] + 1e-9:
                continue
            if tvol > v['vol'] + 1e-9:
                continue
            if not all(item_fits(it, v) for it in items):
                continue
            viable.append(v)
        if not viable:
            continue
        # Try cheapest first; if it fails, try next 2 cheapest
        viable.sort(key=lambda v: v['cost'])
        for v in viable[:3]:
            if time.monotonic() > t_end:
                return bins, False
            nb = _pack_exact(items, v, t_end, rng)
            if nb is None:
                continue
            if nb.cost < old_cost - 1e-9:
                keep = [bins[k].copy() for k in range(n) if k != ai and k != bj]
                keep.append(nb)
                return keep, True
            break  # next cheaper vehicle won't help if this denser one already failed cost-wise
    return bins, False


def op_ruin_recreate(bins, ilookup, vehicles, rng, t_end, ruin_frac=0.30):
    """
    Strong global ruin-and-recreate.
    Removes ~ruin_frac fraction of items spread across multiple bins, then
    rebuilds: first try to fit removed items into surviving bins, otherwise
    open new bins with cheapest-fit vehicle selection.
    Accepts only strict improvements.

    Differs from op_eject (which targets one bin and only repacks into others)
    by scattering removal across the solution and allowing new bins to open —
    this escapes local optima that single-bin moves can't.
    """
    n = len(bins)
    if n < 2:
        return bins, False
    total_items = sum(len(b.items) for b in bins)
    if total_items < 4:
        return bins, False
    n_remove = max(2, int(total_items * ruin_frac))
    n_remove = min(n_remove, total_items - 1)

    # Bias removal toward items in expensive/under-utilized bins
    cand = []  # (score, bin_idx, item_record)
    for bi, b in enumerate(bins):
        util = b.vol_used / max(1.0, b.W * b.D * b.H)
        # higher score = more likely to remove
        s = b.cost / max(0.05, util)
        for rec in b.items:
            jitter = rng.random() * 0.5
            cand.append((s + jitter, bi, rec))
    cand.sort(key=lambda x: -x[0])
    chosen = cand[:n_remove]

    # Build surviving bins (with items kept)
    keep_per_bin = {bi: [] for bi in range(n)}
    removed_set = set()
    for _, bi, rec in chosen:
        removed_set.add((bi, rec[0]))
    for bi, b in enumerate(bins):
        for rec in b.items:
            if (bi, rec[0]) not in removed_set:
                keep_per_bin[bi].append(ilookup[rec[0]])

    # Reconstruct surviving bins from scratch (re-pack kept items)
    surv = []
    for bi, b in enumerate(bins):
        kept = keep_per_bin[bi]
        if not kept:
            continue
        nb = Bin3D(b.vtype, b.W, b.D, b.H, b.max_weight, b.max_value, b.gravity, b.cost)
        # large-first to keep packing density
        ok = True
        for it in sorted(kept, key=lambda x: -x['vol']):
            if time.monotonic() > t_end:
                return bins, False
            if not nb.try_add(it):
                ok = False
                break
        if not ok:
            return bins, False
        surv.append(nb)

    removed_items = [ilookup[rec[0]] for _, _, rec in chosen]
    removed_items.sort(key=lambda x: -x['vol'])

    # Place removed items: first into existing bins (best-fit by remaining vol),
    # then open new bins with cheapest-cost vehicle that fits.
    work = surv
    for it in removed_items:
        if time.monotonic() > t_end:
            return bins, False
        placed = False
        # best-fit among existing
        best_b = None
        best_rem = float('inf')
        for bb in work:
            if not bb.cap_ok(it['weight'], it['value']):
                continue
            for (rot, iw, id_, ih) in it['urots']:
                if iw > bb.W + 1e-9 or id_ > bb.D + 1e-9 or ih > bb.H + 1e-9:
                    continue
                pos = bb.find_ep(iw, id_, ih, it['weight'], it['value'])
                if pos:
                    rem = bb.rem_vol() - iw * id_ * ih
                    if rem < best_rem:
                        best_rem = rem
                        best_b = bb
                    break
        if best_b is not None and best_b.try_add(it):
            placed = True
        if not placed:
            nb = open_bin(it, vehicles)
            if nb is None:
                return bins, False
            work.append(nb)
    if _cost(work) < _cost(bins) - 1e-9:
        return work, True
    return bins, False


def op_weight_pair_repack(bins, ilookup, vehicles, rng, t_end):
    """
    Pair single-item bins of the SAME vehicle type into 2-item (or 3-item) bins.
    Uses 2-pointer (lightest + heaviest) which is optimal for 1D weight matching.

    Targets the case where a vehicle's weight capacity is the binding constraint
    and items are being placed alone wasting capacity. Pairing reduces bin count.

    Feasibility: each new bin is built via Bin3D.try_add() which enforces
    overlap, weight/value caps, gravity and dim fit. Strict-improvement only.
    """
    n = len(bins)
    if n < 4:
        return bins, False

    # Index single-item bins by vehicle type
    single = {}
    for i, b in enumerate(bins):
        if len(b.items) == 1:
            single.setdefault(b.vtype, []).append(i)
    if not any(len(v) >= 2 for v in single.values()):
        return bins, False

    # Build vehicle-spec lookup
    vmap = {v['type']: v for v in vehicles}

    consumed = set()
    new_bins = []
    for vtype, indices in single.items():
        if len(indices) < 2 or vtype not in vmap:
            continue
        if time.monotonic() > t_end:
            break
        v_spec = vmap[vtype]
        # Items with their bin index, sorted by weight ASC
        items_with_idx = [(idx, ilookup[bins[idx].items[0][0]]) for idx in indices]
        items_with_idx.sort(key=lambda x: x[1]['weight'])

        lo, hi = 0, len(items_with_idx) - 1
        while lo < hi:
            if time.monotonic() > t_end:
                break
            ia, item_a = items_with_idx[lo]  # lightest unprocessed
            ib, item_b = items_with_idx[hi]  # heaviest unprocessed
            if ia in consumed or ib in consumed:
                if ia in consumed: lo += 1
                if ib in consumed: hi -= 1
                continue
            nb = Bin3D(v_spec['type'], v_spec['W'], v_spec['D'], v_spec['H'],
                       v_spec['max_weight'], v_spec['max_value'],
                       v_spec['gravity'], v_spec['cost'])
            # Heavy first — usually larger, deserves bottom-left placement
            if not nb.try_add(item_b):
                # Heavy doesn't even fit alone (shouldn't happen since it was a single bin)
                hi -= 1
                continue
            if not nb.try_add(item_a):
                # Pair doesn't fit — even the lightest can't go with this heavy.
                # Heavy stays alone (no benefit pairing it with anything heavier).
                hi -= 1
                continue
            # Optionally try to fit a 3rd item from the still-unprocessed range.
            # Walk from lo+1 upward looking for a tiny item that still fits.
            mid = lo + 1
            added_third = False
            while mid < hi:
                im, item_m = items_with_idx[mid]
                if im in consumed:
                    mid += 1
                    continue
                if nb.try_add(item_m):
                    consumed.add(im)
                    added_third = True
                    break
                mid += 1
            new_bins.append(nb)
            consumed.add(ia)
            consumed.add(ib)
            lo += 1
            hi -= 1

    if not consumed:
        return bins, False

    result = [b.copy() for i, b in enumerate(bins) if i not in consumed]
    result.extend(new_bins)
    if _cost(result) < _cost(bins) - 1e-9:
        return result, True
    return bins, False


def op_bin_split(bins, ilookup, vehicles, rng, t_end):
    """
    Try to SPLIT an expensive multi-item bin into 2 cheaper bins.
    For each candidate bin B (cost c_B, vehicle V_B), find a pair (V', V'')
    of vehicles with c_V' + c_V'' < c_B such that B's items can be split
    between fresh bins of types V' and V''. Sort items by weight desc and
    greedy-assign to whichever bin has more remaining weight capacity (or
    fits dimensionally).

    Feasibility: every placement uses Bin3D.try_add() — gravity, weight,
    value, dim, overlap all checked. Strict-improvement only.
    """
    n = len(bins)
    if n < 1:
        return bins, False
    # Sort vehicles by cost ascending for cheapest-first iteration
    veh_sorted = sorted(vehicles, key=lambda v: v['cost'])

    expensive = sorted(range(n), key=lambda i: -bins[i].cost)
    for ai in expensive[:min(8, n)]:
        if time.monotonic() > t_end:
            break
        b = bins[ai]
        if len(b.items) < 2:
            continue
        items = [ilookup[rec[0]] for rec in b.items]

        # Search: pair of vehicle types (cheaper combined)
        for vi, v1 in enumerate(veh_sorted):
            if v1['cost'] >= b.cost - 1e-9:
                break
            for v2 in veh_sorted[vi:]:
                if v1['cost'] + v2['cost'] >= b.cost - 1e-9:
                    break
                if time.monotonic() > t_end:
                    return bins, False
                # Try to split items between fresh v1 and v2 bins
                # Sort items by weight desc, greedy assign
                ordered = sorted(items, key=lambda x: -x['weight'])
                nb1 = Bin3D(v1['type'], v1['W'], v1['D'], v1['H'],
                            v1['max_weight'], v1['max_value'],
                            v1['gravity'], v1['cost'])
                nb2 = Bin3D(v2['type'], v2['W'], v2['D'], v2['H'],
                            v2['max_weight'], v2['max_value'],
                            v2['gravity'], v2['cost'])
                ok = True
                for it in ordered:
                    # Prefer bin with more remaining weight capacity
                    rem1 = nb1.max_weight - nb1.weight
                    rem2 = nb2.max_weight - nb2.weight
                    if rem1 >= rem2:
                        if not nb1.try_add(it):
                            if not nb2.try_add(it):
                                ok = False
                                break
                    else:
                        if not nb2.try_add(it):
                            if not nb1.try_add(it):
                                ok = False
                                break
                if not ok:
                    continue
                if not nb1.items or not nb2.items:
                    # All items went into one bin — better handled by op_retype
                    continue
                # We have a feasible 2-way split with cheaper combined cost
                cand = [bins[k].copy() for k in range(n) if k != ai]
                cand.append(nb1)
                cand.append(nb2)
                if _cost(cand) < _cost(bins) - 1e-9:
                    return cand, True
                break  # next v2 won't reduce cost since v_sorted ascending
    return bins, False


def _can_host_all(items, v):
    tw = sum(it['weight'] for it in items)
    tv = sum(it['value']  for it in items)
    if tw > v['max_weight'] + 1e-9: return False
    if tv > v['max_value']  + 1e-9: return False
    if sum(it['vol'] for it in items) > v['vol'] + 1e-9: return False
    return all(item_fits(it, v) for it in items)

def _pack_exact(items, v, t_end, rng):
    if not _can_host_all(items, v):
        return None
    seqs = [
        sorted(items, key=lambda x: -x['vol']),
        sorted(items, key=lambda x: (-x['maxdim'], -x['vol'])),
        sorted(items, key=lambda x: (-x['weight'], -x['vol'])),
    ]
    shuffled = list(items)
    rng.shuffle(shuffled)
    seqs.append(shuffled)

    for seq in seqs:
        if time.monotonic() > t_end:
            break
        nb = Bin3D(v['type'], v['W'], v['D'], v['H'],
                   v['max_weight'], v['max_value'], v['gravity'], v['cost'])
        ok = True
        for it in seq:
            if not nb.try_add(it):
                ok = False
                break
        if ok:
            return nb
    return None

def _pack_in_one_or_two_bins(items, vehicles, t_end):
    if not items:
        return []
    ordered = sorted(items, key=lambda x: (-x['vol'], -x['maxdim']))
    candidate_vs = vehicles[:min(5, len(vehicles))]

    # Try to repack everything in a single cheap bin.
    for v in candidate_vs:
        if time.monotonic() > t_end:
            return None
        if not _can_host_all(ordered, v):
            continue
        b = Bin3D(v['type'], v['W'], v['D'], v['H'],
                  v['max_weight'], v['max_value'], v['gravity'], v['cost'])
        ok = True
        for it in ordered:
            if not b.try_add(it):
                ok = False
                break
        if ok:
            return [b]

    # Try to repack in two bins.
    for v1 in candidate_vs:
        if time.monotonic() > t_end:
            return None
        b1 = Bin3D(v1['type'], v1['W'], v1['D'], v1['H'],
                   v1['max_weight'], v1['max_value'], v1['gravity'], v1['cost'])
        rem = []
        for it in ordered:
            if not b1.try_add(it):
                rem.append(it)
        if not rem:
            return [b1]
        for v2 in candidate_vs:
            if time.monotonic() > t_end:
                return None
            if not _can_host_all(rem, v2):
                continue
            b2 = Bin3D(v2['type'], v2['W'], v2['D'], v2['H'],
                       v2['max_weight'], v2['max_value'], v2['gravity'], v2['cost'])
            ok = True
            for it in rem:
                if not b2.try_add(it):
                    ok = False
                    break
            if ok:
                return [b1, b2]
    return None

def op_merge3(bins, ilookup, vehicles, rng, t_end):
    if len(bins) < 3:
        return bins, False
    ranked = sorted(
        range(len(bins)),
        key=lambda i: (bins[i].cost, len(bins[i].items)),
        reverse=True,
    )

    for a in ranked[:min(6, len(ranked))]:
        if time.monotonic() > t_end:
            break
        others = [i for i in range(len(bins)) if i != a]
        rng.shuffle(others)
        others = others[:min(12, len(others))]
        pairs = []
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                pairs.append((others[i], others[j]))
        rng.shuffle(pairs)

        for b, c in pairs[:8]:
            if time.monotonic() > t_end:
                return bins, False
            old_cost = bins[a].cost + bins[b].cost + bins[c].cost
            ids = []
            for idx in (a, b, c):
                ids.extend(rec[0] for rec in bins[idx].items)
            uniq_ids = list(dict.fromkeys(ids))
            items = [ilookup[iid] for iid in uniq_ids]
            packed = _pack_in_one_or_two_bins(items, vehicles, t_end)
            if not packed:
                continue
            new_cost = sum(nb.cost for nb in packed)
            if new_cost < old_cost - 1e-9:
                keep = [bins[k].copy() for k in range(len(bins)) if k not in (a, b, c)]
                keep.extend(packed)
                return keep, True
    return bins, False

def op_retype(bins, ilookup, vehicles, rng, t_end):
    if not bins:
        return bins, False
    order = sorted(range(len(bins)), key=lambda i: bins[i].cost, reverse=True)
    for bi in order[:min(10, len(order))]:
        if time.monotonic() > t_end:
            break
        src = bins[bi]
        cheaper = [v for v in vehicles if v['cost'] < src.cost - 1e-9]
        if not cheaper:
            continue
        its = [ilookup[r[0]] for r in src.items]
        for v in cheaper:
            if time.monotonic() > t_end:
                break
            nb = _pack_exact(its, v, t_end, rng)
            if nb is None:
                continue
            cand = [bins[k].copy() for k in range(len(bins))]
            cand[bi] = nb
            if _cost(cand) < _cost(bins) - 1e-9:
                return cand, True
            break
    return bins, False

def op_retype_all(bins, ilookup, vehicles, t_end):
    """Sweep all bins and retype each to the cheapest feasible vehicle."""
    if not bins:
        return bins, False
    cur = [b.copy() for b in bins]
    improved = False
    for bi in sorted(range(len(cur)), key=lambda i: cur[i].cost, reverse=True):
        if time.monotonic() > t_end:
            break
        src = cur[bi]
        its = [ilookup[r[0]] for r in src.items]
        nb = _retype_partial_bin(src, its, vehicles, t_end)
        if nb is not src and nb.cost < src.cost - 1e-9:
            cur[bi] = nb
            improved = True
    if improved and _cost(cur) < _cost(bins) - 1e-9:
        return cur, True
    return bins, False

def lns(bins, ilookup, vehicles, t_end, rng=None, verbose=False, pool=None):
    """
    Adaptive Large Neighborhood Search (ALNS).
    op_elim is always tried first (deterministic improvement, cheap).
    The remaining operators are selected by roulette wheel based on recent success.
    Weights are periodically decayed toward uniform to keep exploration alive.
    """
    if rng is None: rng=random.Random()
    cur=list(bins); stag=0
    destroy_rate=0.33
    elim_failures=0

    # Mutable state for closures
    state = {'destroy_rate': destroy_rate}

    sec_ops = [
        ('RELOC',  lambda c: op_relocate(c, ilookup, vehicles, rng, t_end)),
        ('RETYPE', lambda c: op_retype(c,   ilookup, vehicles, rng, t_end)),
        ('RETALL', lambda c: op_retype_all(c, ilookup, vehicles, t_end)),
        ('SHAKE',  lambda c: op_shake(c,    ilookup, vehicles, rng, t_end)),
        ('MERGE3', lambda c: op_merge3(c,   ilookup, vehicles, rng, t_end)),
        ('EJECT',  lambda c: op_eject(c,    ilookup, vehicles, rng, t_end,
                                       destroy_rate=state['destroy_rate'])),
        ('SWAP',   lambda c: op_swap_pair(c, ilookup, vehicles, rng, t_end)),
        ('CONS2',  lambda c: op_consolidate_pair(c, ilookup, vehicles, rng, t_end)),
        ('RUIN',   lambda c: op_ruin_recreate(c, ilookup, vehicles, rng, t_end,
                                              ruin_frac=min(0.45, max(0.20,
                                                  state['destroy_rate'])))),
        # Stronger ruin variant — basin-escape kick larger than the normal RUIN.
        # ALNS roulette will use it sparingly unless it produces improvements.
        ('RUIN_STRONG', lambda c: op_ruin_recreate(c, ilookup, vehicles, rng, t_end,
                                                   ruin_frac=0.60)),
        ('WPAIR',  lambda c: op_weight_pair_repack(c, ilookup, vehicles, rng, t_end)),
        ('SPLIT',  lambda c: op_bin_split(c, ilookup, vehicles, rng, t_end)),
        ('REDIS',  lambda c: op_redistribute_then_retype(c, ilookup, vehicles, rng, t_end)),
    ]
    decay_factor = {'RELOC': 0.88, 'RETYPE': 0.92, 'RETALL': 0.94, 'SHAKE': 0.92,
                    'MERGE3': 0.90, 'EJECT': 0.95, 'SWAP': 0.93,
                    'CONS2': 0.92, 'RUIN': 0.95, 'RUIN_STRONG': 0.97,
                    'WPAIR': 0.92, 'SPLIT': 0.93, 'REDIS': 0.94}
    n_ops    = len(sec_ops)
    weights  = [1.0] * n_ops
    reaction = 0.40   # how aggressively we update weights on success
    decay    = 0.96   # periodic pull toward uniform for exploration
    iter_n   = 0
    # VNS-style escalation: when stuck, fire progressively stronger kicks
    # (Variable Neighborhood Search) before giving up on this LNS call.
    vns_levels = [0.30, 0.45, 0.60]  # ruin fractions for escalation
    vns_idx = 0
    best_cost_seen = _cost(cur)
    best_cur = [b.copy() for b in cur]

    while time.monotonic() < t_end:
        # Always try elim first.
        new, ok = op_elim(cur, ilookup, vehicles, t_end)
        if ok:
            cur = new; stag = 0
            state['destroy_rate'] = max(0.25, state['destroy_rate'] * 0.85)
            elim_failures = 0
            if pool: pool.add_solution(cur)
            cc = _cost(cur)
            if cc < best_cost_seen - 1e-9:
                best_cost_seen = cc
                best_cur = [b.copy() for b in cur]
                vns_idx = 0  # reset escalation on real improvement
            if verbose: print(f'    [ELIM]  bins={len(cur):3d}  cost={cc:.2f}')
            continue
        else:
            elim_failures += 1
            if elim_failures >= 5:
                state['destroy_rate'] = min(0.85, state['destroy_rate'] + 0.25)
                elim_failures = 0

        # Roulette-wheel pick over secondary operators.
        total = sum(weights)
        if total <= 1e-9:
            weights = [1.0] * n_ops; total = float(n_ops)
        threshold = rng.random() * total
        acc = 0.0
        choice = n_ops - 1
        for i, w in enumerate(weights):
            acc += w
            if threshold <= acc:
                choice = i
                break
        op_name, op_fn = sec_ops[choice]
        new, ok = op_fn(cur)
        if ok:
            cur = new; stag = 0
            state['destroy_rate'] = max(0.25,
                state['destroy_rate'] * decay_factor[op_name])
            if pool: pool.add_solution(cur)
            cc = _cost(cur)
            if cc < best_cost_seen - 1e-9:
                best_cost_seen = cc
                best_cur = [b.copy() for b in cur]
                vns_idx = 0
            if verbose: print(f'    [{op_name}] bins={len(cur):3d}  cost={cc:.2f}')
            # Reward: bias toward this operator.
            weights[choice] = weights[choice] * (1 - reaction) + reaction * 6.0
        else:
            # Mild penalty so a failing operator fades.
            weights[choice] = max(0.1, weights[choice] * 0.97)
            stag += 1
            # VNS escalation at stagnation milestones
            if stag in (60, 90) and vns_idx < len(vns_levels):
                if time.monotonic() > t_end:
                    break
                kicked, kok = op_ruin_recreate(
                    best_cur, ilookup, vehicles, rng, t_end,
                    ruin_frac=vns_levels[vns_idx]
                )
                vns_idx += 1
                if kok:
                    cur = kicked
                    if pool: pool.add_solution(cur)
                    if verbose:
                        print(f'    [VNS-K{vns_idx}] cost={_cost(cur):.2f}')
                    cc = _cost(cur)
                    if cc < best_cost_seen - 1e-9:
                        best_cost_seen = cc
                        best_cur = [b.copy() for b in cur]
                        vns_idx = 0
            if stag >= 150: break

        iter_n += 1
        if iter_n % 30 == 0:
            avg = sum(weights) / n_ops
            for i in range(n_ops):
                weights[i] = weights[i] * decay + avg * (1.0 - decay)

    # Always return the best seen, not just the final state.
    return best_cur if _cost(best_cur) < _cost(cur) - 1e-9 else cur

def post_optimize_bins(bins, ilookup, vehicles, t_end, pool=None):
    """
    Deterministic intensification pass for the incumbent.
    Applies only strict-improvement moves, so objective cannot worsen.
    Includes op_relocate and op_swap_pair for final polish.
    """
    cur = [b.copy() for b in bins]
    rng = random.Random(987654321)
    while time.monotonic() < t_end:
        imp = False
        for op in (
            lambda x: op_elim(x, ilookup, vehicles, t_end),
            lambda x: op_redistribute_then_retype(x, ilookup, vehicles, rng, t_end),
            lambda x: op_consolidate_pair(x, ilookup, vehicles, rng, t_end),
            lambda x: op_weight_pair_repack(x, ilookup, vehicles, rng, t_end),
            lambda x: op_relocate(x, ilookup, vehicles, rng, t_end),
            lambda x: op_retype_all(x, ilookup, vehicles, t_end),
            lambda x: op_retype(x, ilookup, vehicles, rng, t_end),
            lambda x: op_merge3(x, ilookup, vehicles, rng, t_end),
            lambda x: op_bin_split(x, ilookup, vehicles, rng, t_end),
            lambda x: op_swap_pair(x, ilookup, vehicles, rng, t_end),
            lambda x: op_eject(x, ilookup, vehicles, rng, t_end, destroy_rate=0.45),
        ):
            new, ok = op(cur)
            if ok and _cost(new) < _cost(cur) - 1e-9:
                cur = new
                imp = True
                if pool:
                    pool.add_solution(cur)
                break
            if time.monotonic() >= t_end:
                break
        if not imp:
            break
    return cur


# ══════════════════════════════════════════════════════════════════════════════
#  WEIGHT-AWARE / VOLUME-AWARE CONSTRUCTION
#  Targets weight-binding and volume-binding instances directly
#  by aggressively filling the most resource-efficient vehicle to its cap.
# ══════════════════════════════════════════════════════════════════════════════

def _retype_partial_bin(b, items_in_b, by_abs, t_end):
    """Try to downsize bin b to cheaper vehicle that still fits items.
    Returns new bin (or original if no downsize found)."""
    tw = sum(it['weight'] for it in items_in_b)
    tv = sum(it['value']  for it in items_in_b)
    for v in by_abs:
        if time.monotonic() > t_end: break
        if v['cost'] >= b.cost - 1e-9: break
        if v['max_weight'] < tw - 1e-9: continue
        if v['max_value']  < tv - 1e-9: continue
        # Heuristic: also need volume to fit at least
        if v['vol'] < sum(it['vol'] for it in items_in_b) - 1e-9: continue
        # Stable baseline: volume-first retype order.
        nb = Bin3D(v['type'], v['W'], v['D'], v['H'],
                   v['max_weight'], v['max_value'],
                   v['gravity'], v['cost'])
        ordered = sorted(items_in_b, key=lambda x: -x['vol'])
        ok = True
        for it in ordered:
            if not nb.try_add(it):
                ok = False
                break
        if ok:
            return nb

    return b


def _lpt_pack(items_sorted, vehicles, n_primary, primary, by_abs, t_end,
              squeeze=True):
    """LPT-style packing into n_primary primary bins + leftover tail bin.
    Heaviest items first, each goes to the bin with currently lowest weight
    that can fit it. Items that can't fit any primary bin go to leftover.

    With squeeze=True, after main LPT pass, try unplaced items in vol-asc
    order against ALL bins (small items can fill EP gaps left by big items).

    Leftovers placed in cheapest viable single tail vehicle.
    Returns list of bins (post-retype) or None on infeasibility."""
    if n_primary < 1:
        return None
    bins = [Bin3D(primary['type'], primary['W'], primary['D'], primary['H'],
                  primary['max_weight'], primary['max_value'],
                  primary['gravity'], primary['cost'])
            for _ in range(n_primary)]
    unplaced = []
    for k, it in enumerate(items_sorted):
        if time.monotonic() > t_end:
            unplaced.extend(items_sorted[k:])
            break
        # Try lightest bin first (LPT scheduling rule)
        order = sorted(range(len(bins)), key=lambda i: bins[i].weight)
        placed = False
        for j in order:
            if bins[j].try_add(it):
                placed = True; break
        if not placed:
            unplaced.append(it)

    # Squeeze pass: small leftovers may fit in EP gaps of existing bins.
    # Sort unplaced by volume asc — smallest items first.
    if squeeze and unplaced:
        unplaced.sort(key=lambda x: x['vol'])
        still_unplaced = []
        for it in unplaced:
            if time.monotonic() > t_end:
                still_unplaced.append(it); continue
            placed = False
            # Try bin with most volume slack first (best chance for small)
            order = sorted(range(len(bins)),
                           key=lambda i: -(bins[i].W * bins[i].D * bins[i].H
                                            - bins[i].vol_used))
            for j in order:
                if bins[j].try_add(it):
                    placed = True; break
            if not placed:
                still_unplaced.append(it)
        unplaced = still_unplaced

    bins = [b for b in bins if b.items]

    # Place leftovers in single cheapest viable tail vehicle
    if unplaced:
        unp_w = sum(it['weight'] for it in unplaced)
        unp_v = sum(it['value'] for it in unplaced)
        unp_vol = sum(it['vol'] for it in unplaced)
        tail_done = False
        for v in by_abs:
            if time.monotonic() > t_end: break
            if v['max_weight'] < unp_w - 1e-9: continue
            if v['max_value'] < unp_v - 1e-9: continue
            if v['vol'] < unp_vol - 1e-9: continue
            # Try multiple item orderings for tail packing
            for order_key in (
                lambda x: -x['vol'],
                lambda x: -x['maxdim'],
                lambda x: -x['base_area'],
            ):
                if time.monotonic() > t_end: break
                nb = Bin3D(v['type'], v['W'], v['D'], v['H'],
                           v['max_weight'], v['max_value'],
                           v['gravity'], v['cost'])
                ordered = sorted(unplaced, key=order_key)
                ok = True
                for it in ordered:
                    if not nb.try_add(it):
                        ok = False; break
                if ok:
                    bins.append(nb)
                    unplaced = []
                    tail_done = True
                    break
            if tail_done: break
        # Fallback: place each leftover individually
        if not tail_done:
            for it in unplaced:
                placed = False
                for b in bins:
                    if b.try_add(it):
                        placed = True; break
                if not placed:
                    nb = open_bin(it, by_abs)
                    if nb: bins.append(nb)
                    else: return None

    # Retype each bin to cheapest viable vehicle
    item_by_id = {it['id']: it for it in items_sorted}
    for i in range(len(bins)):
        if time.monotonic() > t_end: break
        b = bins[i]
        items_in_b = [item_by_id[rec[0]] for rec in b.items]
        bins[i] = _retype_partial_bin(b, items_in_b, by_abs, t_end)
    return bins


def construct_weight_packed(items_dicts, vehicles, t_end):
    """
    Weight-FFD construction with LPT distribution.

    Uses cheapest cost-per-weight vehicle as 'primary', tries
    (n_min - 1, n_min, n_min + 1) bins crossed with multiple item orderings,
    then retypes each bin to the cheapest viable vehicle.
    Returns the cheapest feasible candidate.
    """
    if not items_dicts:
        return []
    by_cpw = sorted(vehicles, key=lambda v: v['cost'] / max(v['max_weight'], 1.0))
    by_abs = sorted(vehicles, key=lambda v: v['cost'])
    primary = by_cpw[0]
    total_w = sum(it['weight'] for it in items_dicts)
    n_min = max(1, int(math.ceil(total_w / max(primary['max_weight'], 1.0))))

    # Try multiple item orderings — LPT phase order matters for 3D density.
    orderings = [
        sorted(items_dicts, key=lambda x: -x['weight']),
        sorted(items_dicts, key=lambda x: (-x['vol'], -x['maxdim'])),
        sorted(items_dicts, key=lambda x: (-x['maxdim'], -x['vol'])),
    ]

    candidates = []
    for n in (n_min - 1, n_min, n_min + 1):
        if time.monotonic() > t_end: break
        if n < 1: continue
        for items_sorted in orderings:
            if time.monotonic() > t_end: break
            try:
                bins = _lpt_pack(items_sorted, vehicles, n, primary, by_abs, t_end)
            except Exception:
                bins = None
            if bins:
                candidates.append(bins)

    if not candidates:
        return None
    return min(candidates, key=_cost)


def construct_volume_packed(items_dicts, vehicles, t_end):
    """
    Greedy volume-FFD: target volume-binding instances.
    Picks vehicle with min(cost / vol), packs items vol-desc to volume cap.
    """
    if not items_dicts:
        return []
    by_cpv = sorted(vehicles, key=lambda v: v['cost'] / max(v['vol'], 1.0))
    by_abs = sorted(vehicles, key=lambda v: v['cost'])

    primary = by_cpv[0]
    items_sorted = sorted(items_dicts, key=lambda x: (-x['vol'], -x['maxdim']))

    bins = []
    unplaced = list(items_sorted)
    while unplaced:
        if time.monotonic() > t_end:
            return None
        nb = Bin3D(primary['type'], primary['W'], primary['D'], primary['H'],
                   primary['max_weight'], primary['max_value'],
                   primary['gravity'], primary['cost'])
        new_unplaced = []
        for it in unplaced:
            if not nb.try_add(it):
                new_unplaced.append(it)
        if not nb.items:
            it = unplaced[0]
            nb_fb = open_bin(it, by_abs)
            if nb_fb is None:
                return None
            bins.append(nb_fb)
            unplaced = unplaced[1:]
            continue
        bins.append(nb)
        unplaced = new_unplaced

    item_by_id = {it['id']: it for it in items_dicts}
    for i in range(len(bins)):
        if time.monotonic() > t_end: break
        b = bins[i]
        items_in_b = [item_by_id[rec[0]] for rec in b.items]
        bins[i] = _retype_partial_bin(b, items_in_b, by_abs, t_end)
    return bins


def construct_weight_packed_diverse(items_dicts, vehicles, t_end, rng,
                                    n_primaries=3, n_perturbed=2):
    """
    Diversified weight-FFD constructor for column-pool seeding.

    construct_weight_packed only uses the single cheapest cost-per-weight
    vehicle as 'primary', so the column pool tends to be dominated by bin
    shapes from one vehicle type. This variant additionally:
      - tries the top-N cheapest cost-per-weight vehicles as primary, and
      - picks an extra primary using a perturbed cost-per-weight ordering
        (cost ± 20%); this is a pure SELECTION jitter — the picked vehicle
        dict is from the original list, so bins always carry true costs.

    Returns a list of complete bin-list candidates (each is one full cover
    of items_dicts under one specific primary). Caller feeds each to pool.
    """
    if not items_dicts or not vehicles:
        return []

    by_abs = sorted(vehicles, key=lambda v: v['cost'])
    total_w = sum(it['weight'] for it in items_dicts)
    orderings = [
        sorted(items_dicts, key=lambda x: -x['weight']),
        sorted(items_dicts, key=lambda x: (-x['vol'], -x['maxdim'])),
    ]

    by_cpw = sorted(vehicles, key=lambda v: v['cost'] / max(v['max_weight'], 1.0))
    primaries = list(by_cpw[:min(n_primaries, len(by_cpw))])

    seen_types = {p['type'] for p in primaries}
    for _ in range(n_perturbed):
        keys = [
            (v, v['cost'] * (1.0 + rng.uniform(-0.20, 0.20))
                / max(v['max_weight'], 1.0))
            for v in vehicles
        ]
        keys.sort(key=lambda x: x[1])
        cand = keys[0][0]
        if cand['type'] not in seen_types:
            primaries.append(cand)
            seen_types.add(cand['type'])

    candidates = []
    for primary in primaries:
        if time.monotonic() > t_end:
            break
        n_min = max(1, int(math.ceil(total_w / max(primary['max_weight'], 1.0))))
        for n in (n_min - 1, n_min, n_min + 1):
            if n < 1 or time.monotonic() > t_end:
                continue
            for items_sorted in orderings:
                if time.monotonic() > t_end:
                    break
                try:
                    bins = _lpt_pack(items_sorted, vehicles, n, primary,
                                     by_abs, t_end)
                except Exception:
                    bins = None
                if bins:
                    candidates.append(bins)
    return candidates


def op_redistribute_then_retype(bins, ilookup, vehicles, rng, t_end):
    """
    Move items from a 'victim' bin into other same-vtype bins (filling them
    toward weight cap), then retype victim to cheapest vehicle that still
    fits its remaining items.

    Targets the canonical pathology: N homogeneous bins at 80-90% weight
    utilization, when the LP-optimal would be (N-1) at 100% + 1 small bin.

    Strict-improvement only.
    """
    n = len(bins)
    if n < 2:
        return bins, False
    by_type = {}
    for i, b in enumerate(bins):
        by_type.setdefault(b.vtype, []).append(i)
    candidates = [(vt, idxs) for vt, idxs in by_type.items() if len(idxs) >= 2]
    if not candidates:
        return bins, False

    rng.shuffle(candidates)
    by_abs_cost = sorted(vehicles, key=lambda v: v['cost'])

    for vt, idxs in candidates:
        if time.monotonic() > t_end:
            break
        # Try multiple "victim" choices: the lightest 1..3 bins
        idxs_by_w = sorted(idxs, key=lambda i: bins[i].weight)
        n_victims_try = min(3, len(idxs_by_w) - 1)
        for vt_pick in range(n_victims_try):
            if time.monotonic() > t_end:
                break
            victim_idx = idxs_by_w[vt_pick]
            victim = bins[victim_idx]
            if not victim.items:
                continue
            other_idx_list = [i for i in idxs_by_w if i != victim_idx]
            recv_copies = [bins[i].copy() for i in other_idx_list]

            items_to_move = sorted(
                [(rec[0], ilookup[rec[0]]) for rec in victim.items],
                key=lambda p: -p[1]['weight']
            )
            moved_ids = set()
            for iid, idict in items_to_move:
                if time.monotonic() > t_end:
                    break
                # Place in receiver with most remaining weight first (best fit chance)
                order = sorted(
                    range(len(recv_copies)),
                    key=lambda j: -(recv_copies[j].max_weight - recv_copies[j].weight)
                )
                for j in order:
                    if recv_copies[j].try_add(idict):
                        moved_ids.add(iid)
                        break
            if not moved_ids:
                continue

            remaining_items = [ilookup[rec[0]] for rec in victim.items
                               if rec[0] not in moved_ids]

            if not remaining_items:
                # Victim fully emptied — drop it
                cand = []
                for k, b in enumerate(bins):
                    if k == victim_idx:
                        continue
                    if k in other_idx_list:
                        cand.append(recv_copies[other_idx_list.index(k)])
                    else:
                        cand.append(b.copy())
                if _cost(cand) < _cost(bins) - 1e-9:
                    return cand, True
                continue

            # Try retype victim to cheapest vehicle holding remaining_items
            retyped = _retype_partial_bin(victim, remaining_items,
                                          by_abs_cost, t_end)
            if retyped is victim or retyped.cost >= victim.cost - 1e-9:
                continue

            cand = []
            for k, b in enumerate(bins):
                if k == victim_idx:
                    cand.append(retyped)
                elif k in other_idx_list:
                    cand.append(recv_copies[other_idx_list.index(k)])
                else:
                    cand.append(b.copy())
            if _cost(cand) < _cost(bins) - 1e-9:
                return cand, True
    return bins, False


# ══════════════════════════════════════════════════════════════════════════════
#  GRASP construction
# ══════════════════════════════════════════════════════════════════════════════

def build_sequence(items, mode, rng):
    if mode == 'vol':
        return sorted(items, key=lambda x: (-x['vol'], -x['maxdim']))
    if mode == 'weight':
        return sorted(items, key=lambda x: (-x['weight'], -x['vol']))
    if mode == 'value':
        return sorted(items, key=lambda x: (-x['value'], -x['vol']))
    if mode == 'maxdim':
        return sorted(items, key=lambda x: (-x['maxdim'], -x['vol']))
    if mode == 'footprint':
        return sorted(items, key=lambda x: (-x['base_area'], -x['vol']))
    if mode == 'densw':
        return sorted(items, key=lambda x: (-x['density_w'], -x['vol']))
    if mode == 'densv':
        return sorted(items, key=lambda x: (-x['density_v'], -x['vol']))
    if mode == 'hard':
        return sorted(
            items,
            key=lambda x: (
                -x.get('hardness', 0.0),
                x.get('fit_count', 10**9),
                -x['maxdim'],
                -x['vol'],
            ),
        )
    if mode == 'mixed':
        a = 0.6 + rng.random()
        b = 0.4 + rng.random()
        c = 0.2 + rng.random()
        d = 0.2 + rng.random()
        e = 0.2 + rng.random()
        return sorted(
            items,
            key=lambda x: -(
                a * x['vol'] +
                b * x['base_area'] +
                c * x['weight'] +
                d * x['value'] +
                e * x['maxdim']
            ),
        )
    seq = list(items)
    rng.shuffle(seq)
    return seq

def grasp(items, vehicles, alpha, rng, t_end, best_fit=False, honor_order=False):
    scored = list(items) if honor_order else sorted(items, key=lambda x: -x['vol'])
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


def _filter_columns_lp(columns, all_item_ids, max_cols=700, lp_budget=8.,
                        must_keep_keys=None):
    """
    Reduce column pool to max_cols using a multi-step strategy:
      0. Mandatory inclusion of caller-supplied "must keep" columns
         (e.g. the incumbent's bins) — protects warm-start from being filtered out.
      1. LP Set-Cover relaxation (>= 1) to identify fractionally useful columns.
      2. Cheapest column per item — guarantees coverage feasibility.
      3. Fill to max_cols by cheapest cost-per-covered-item ratio.
    """
    if len(columns) <= max_cols:
        return columns

    n_items = len(all_item_ids)
    A, costs, _ = _build_coverage_matrix(columns, all_item_ids)

    keep = set()

    # Step 0: must-keep (incumbent columns) — never filter out the incumbent.
    if must_keep_keys:
        col_keys = [frozenset(rec[0] for rec in b.items) for b in columns]
        for j, k in enumerate(col_keys):
            if k in must_keep_keys:
                keep.add(j)

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
    for i in range(n_items):
        covering = A.getrow(i).indices
        if covering.size == 0:
            continue
        # Prefer cheapest covering column for this item.
        best_j = int(covering[np.argmin(costs[covering])])
        keep.add(int(best_j))

    # Step 3: Fill remainder by cost/n_items_covered ratio
    score  = costs / np.maximum(n_items_col, 1.)
    ranked = np.argsort(score)
    for j in ranked:
        if len(keep) >= max_cols:
            break
        keep.add(int(j))

    return [columns[j] for j in sorted(keep)]

def _build_cover_warm_start(A, costs):
    """Item-wise cheapest covering warm start."""
    warm = set()
    for i in range(A.shape[0]):
        cols = A.getrow(i).indices
        if cols.size == 0:
            continue
        warm.add(int(cols[np.argmin(costs[cols])]))
    return sorted(warm)

def _expr_sum_vars(xvars, cols):
    cols = list(cols)
    if not cols:
        return None
    expr = xvars[int(cols[0])]
    for j in cols[1:]:
        expr = expr + xvars[int(j)]
    return expr

def _solve_cover_highspy(
    scaled_costs, cover_rows, clique_rows, t_budget, gap,
    warm_cols=None, parallel_mode='on'
):
    if not _HAS_HIGHSPY:
        return None
    try:
        h = highspy.Highs()
        h.setOptionValue('output_flag', False)
        h.setOptionValue('presolve', 'on')
        h.setOptionValue('parallel', parallel_mode)
        h.setOptionValue('threads', 4 if parallel_mode == 'on' else 1)
        h.setOptionValue('time_limit', float(max(1.0, t_budget)))
        h.setOptionValue('mip_rel_gap', float(gap))
        # Disable symmetry detection: a past worker-thread crash inside
        # HighsSymmetryDetection::run was reproducible on large clique sets.
        try:
            h.setOptionValue('mip_detect_symmetry', False)
        except Exception:
            pass

        xvars = [h.addBinary(obj=float(c), name=f'x{j}') for j, c in enumerate(scaled_costs)]

        for cols in cover_rows:
            expr = _expr_sum_vars(xvars, cols)
            if expr is None:
                return None
            h.addConstr(expr >= 1)

        for cols in clique_rows:
            expr = _expr_sum_vars(xvars, cols)
            if expr is None:
                continue
            h.addConstr(expr <= 0)

        if warm_cols:
            idx = np.asarray(sorted(set(int(c) for c in warm_cols)), dtype=np.int32)
            vals = np.ones(len(idx), dtype=np.float64)
            h.setSolution(len(idx), idx, vals)

        h.solve()
        model_status = h.getModelStatus()
        good = {
            highspy.HighsModelStatus.kOptimal,
            highspy.HighsModelStatus.kTimeLimit,
            highspy.HighsModelStatus.kObjectiveBound,
            highspy.HighsModelStatus.kObjectiveTarget,
            highspy.HighsModelStatus.kSolutionLimit,
        }
        if model_status not in good:
            return None

        vals = np.asarray(h.allVariableValues(), dtype=np.float64)
        if vals.size != len(scaled_costs):
            return None
        return vals
    except Exception as e:
        print(f'  [highspy] error: {e}')
        return None

def _solve_cover_cpsat(
    scaled_costs, cover_rows, clique_rows, t_budget, gap,
    warm_cols=None, n_workers=4, seed=1
):
    if not _HAS_ORTOOLS or t_budget <= 0.5:
        return None
    try:
        m = cp_model.CpModel()
        n = len(scaled_costs)
        x = [m.NewBoolVar(f'x{j}') for j in range(n)]

        scale = 1_000_000
        int_costs = [max(1, int(round(float(c) * scale))) for c in scaled_costs]
        m.Minimize(sum(int_costs[j] * x[j] for j in range(n)))

        for cols in cover_rows:
            if not cols:
                return None
            m.Add(sum(x[int(j)] for j in cols) >= 1)

        # Columns in each clique row are forbidden together (sum <= 0).
        for cols in clique_rows:
            if cols:
                m.Add(sum(x[int(j)] for j in cols) <= 0)

        if warm_cols:
            for j in sorted(set(int(c) for c in warm_cols)):
                if 0 <= j < n:
                    m.AddHint(x[j], 1)

        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = float(max(0.5, t_budget))
        s.parameters.num_search_workers = int(max(1, n_workers))
        s.parameters.random_seed = int(seed)
        s.parameters.relative_gap_limit = float(max(1e-6, gap))
        s.parameters.log_search_progress = False

        status = s.Solve(m)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        return np.asarray([float(s.Value(v)) for v in x], dtype=np.float64)
    except Exception as e:
        print(f'  [cp-sat] error: {e}')
        return None


def _build_conflict_pairs(ilookup, vehicles, max_items=260, max_pairs=15000):
    """
    Build a limited set of pairwise incompatibilities for clique-style cuts.
    A pair is marked conflicting if it exceeds at least one absolute bin limit
    (volume, weight, value) even for the largest-capacity vehicle.
    """
    if not ilookup or not vehicles:
        return []

    max_vol = max(v['vol'] for v in vehicles)
    max_w   = max(v['max_weight'] for v in vehicles)
    max_v   = max(v['max_value'] for v in vehicles)

    cand = sorted(ilookup.values(), key=lambda x: (-x['vol'], -x['weight'], -x['value']))
    cand = cand[:min(max_items, len(cand))]

    pairs = []
    for i in range(len(cand)):
        a = cand[i]
        for j in range(i + 1, len(cand)):
            b = cand[j]
            if (a['vol'] + b['vol'] > max_vol + 1e-9 or
                a['weight'] + b['weight'] > max_w + 1e-9 or
                a['value'] + b['value'] > max_v + 1e-9):
                pairs.append((a['id'], b['id']))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs

def _rebuild_bin_subset(template_bin, records, ilookup):
    nb = Bin3D(template_bin.vtype, template_bin.W, template_bin.D, template_bin.H,
               template_bin.max_weight, template_bin.max_value,
               template_bin.gravity, template_bin.cost)
    for (iid, x, y, z, iw, id_, ih, orient) in records:
        it = ilookup[iid]
        nb.place(iid, x, y, z, iw, id_, ih, orient, it['weight'], it['value'])
    return nb

def _pack_assigned_items(template_bin, assigned_ids, ilookup):
    rec_map = {rec[0]: rec for rec in template_bin.items}
    ordered = sorted(
        assigned_ids,
        key=lambda iid: (rec_map.get(iid, (None, 0, 0, 0))[3], -ilookup[iid]['vol']),
    )
    nb = Bin3D(
        template_bin.vtype, template_bin.W, template_bin.D, template_bin.H,
        template_bin.max_weight, template_bin.max_value, template_bin.gravity, template_bin.cost
    )
    rem = []
    for iid in ordered:
        if not nb.try_add(ilookup[iid]):
            rem.append(iid)
    return nb, rem

def _cover_postprocess(selected_bins, all_item_ids, ilookup, vehicles):
    """
    Convert covering solution (A>=1) into a strict partition by keeping each
    item in one selected bin only (prefer cheaper bins, then lower z).
    """
    choice = {}
    for bi, b in enumerate(selected_bins):
        for rec in b.items:
            iid = rec[0]
            score = (b.cost, rec[3], rec[1] + rec[2], bi)
            prev = choice.get(iid)
            if prev is None or score < prev[0]:
                choice[iid] = (score, bi)

    for iid in all_item_ids:
        if iid not in choice:
            return None

    assigned = [[] for _ in selected_bins]
    for iid in all_item_ids:
        assigned[choice[iid][1]].append(iid)

    out = []
    leftovers = []
    for bi, ids in enumerate(assigned):
        if not ids:
            continue
        nb, rem = _pack_assigned_items(selected_bins[bi], ids, ilookup)
        if nb.items:
            out.append(nb)
        leftovers.extend(rem)

    # Repair residual items in feasible placements only.
    for iid in sorted(leftovers, key=lambda x: -ilookup[x]['vol']):
        item = ilookup[iid]
        placed = False
        for b in sorted(out, key=lambda bb: bb.rem_vol()):
            if b.try_add(item):
                placed = True
                break
        if not placed:
            nb = open_bin(item, vehicles)
            if nb is None:
                return None
            out.append(nb)

    covered = {rec[0] for b in out for rec in b.items}
    if covered != set(all_item_ids):
        return None
    return out

def solve_set_partition(columns, all_item_ids, t_budget=45.0, max_cols=900,
                        ilookup=None, vehicles=None, warm_keys=None,
                        highs_parallel='on'):
    """
    Solve set covering master (A >= 1), then post-process to exact partition.
    """
    if not columns:
        return None, float('inf')

    all_set = set(all_item_ids)

    # ── Filter to tractable size via LP ───────────────────────────────────
    lp_budget = min(8., t_budget * 0.18)
    must_keep = set(warm_keys) if warm_keys else None
    columns   = _filter_columns_lp(columns, all_item_ids,
                                   max_cols=max_cols, lp_budget=lp_budget,
                                   must_keep_keys=must_keep)

    # ── Check every item is covered by at least one column ────────────────
    A, costs, item_idx = _build_coverage_matrix(columns, all_item_ids)
    row_sums = np.array(A.sum(axis=1)).flatten()
    if row_sums.min() < 0.5:
        return None, float('inf')   # some item uncoverable in this pool

    n_items = len(all_item_ids)
    n_cols  = len(columns)

    constraints = [LinearConstraint(
        A,
        lb=np.ones(n_items),
        ub=np.full(n_items, np.inf),   # covering relaxation: A x >= 1
    )]
    bounds      = Bounds(lb=np.zeros(n_cols), ub=np.ones(n_cols))
    integrality = np.ones(n_cols)

    milp_budget = max(5., t_budget - lp_budget - 1.)
    # Tighter gap thresholds: MILP-final at 10s now reaches 5e-5 (was 1e-4),
    # MILP-1 at ~34s reaches 1e-5 (was 5e-5). Forces the solver to converge
    # closer to true optimum on the rich post-CG column pool.
    gap = 1e-4
    if milp_budget > 8.0:
        gap = 5e-5
    if milp_budget > 25.0:
        gap = 1e-5

    # Clique-style cuts on a limited pair set.
    conflict_pairs = _build_conflict_pairs(ilookup, vehicles)
    clique_rows_cols = []
    if conflict_pairs:
        involved = set()
        for a, b in conflict_pairs:
            if a in item_idx:
                involved.add(a)
            if b in item_idx:
                involved.add(b)
        item_cols = {
            iid: set(A.getrow(item_idx[iid]).indices.tolist())
            for iid in involved
        }
        cut_rows, cut_cols = [], []
        cut_idx = 0
        for (a, b) in conflict_pairs:
            cols_a = item_cols.get(a)
            cols_b = item_cols.get(b)
            if cols_a is None or cols_b is None:
                continue
            both = cols_a.intersection(cols_b)
            if not both:
                continue
            clique_rows_cols.append(sorted(both))
            for j in both:
                cut_rows.append(cut_idx)
                cut_cols.append(j)
            cut_idx += 1
        if cut_idx > 0:
            C = csc_matrix(
                (np.ones(len(cut_rows), dtype=np.float64), (cut_rows, cut_cols)),
                shape=(cut_idx, n_cols),
            )
            constraints.append(
                LinearConstraint(
                    C,
                    lb=np.full(cut_idx, -np.inf),
                    ub=np.zeros(cut_idx),
                )
            )

    try:
        max_c = max(float(np.max(costs)), 1.0)
        scaled_costs = costs / max_c
        warm_cols = set(_build_cover_warm_start(A, costs))
        if warm_keys:
            key_to_idx = {
                frozenset(rec[0] for rec in b.items): j
                for j, b in enumerate(columns)
            }
            for k in warm_keys:
                j = key_to_idx.get(k)
                if j is not None:
                    warm_cols.add(j)

        cover_rows = [A.getrow(i).indices.tolist() for i in range(n_items)]
        candidate_x = []

        use_cpsat = _HAS_ORTOOLS and n_cols <= 3200 and milp_budget >= 6.0
        cpsat_budget = min(6.0, milp_budget * 0.20) if use_cpsat else 0.0
        main_budget = max(2.0, milp_budget - cpsat_budget)

        x_main = None
        if _HAS_HIGHSPY:
            x_main = _solve_cover_highspy(
                scaled_costs, cover_rows, clique_rows_cols, main_budget, gap,
                warm_cols=sorted(warm_cols), parallel_mode=highs_parallel
            )

        if x_main is None:
            result = milp(
                scaled_costs,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={
                    'time_limit': main_budget,
                    'disp': False,
                    'mip_rel_gap': gap,
                    'presolve': True,
                }
            )
            if result.x is not None:
                x_main = result.x

        if x_main is not None:
            candidate_x.append(x_main)

        if use_cpsat and cpsat_budget > 1.0:
            x_cp = _solve_cover_cpsat(
                scaled_costs,
                cover_rows,
                clique_rows_cols,
                cpsat_budget,
                gap=max(gap, 5e-5),
                warm_cols=sorted(warm_cols),
                n_workers=4,
                seed=(len(all_item_ids) * 131 + n_cols * 17 + 7),
            )
            if x_cp is not None:
                candidate_x.append(x_cp)

        best_selected = None
        best_obj = float('inf')
        for x_vals in candidate_x:
            selected = [columns[j] for j in range(n_cols) if x_vals[j] > 0.5]
            if ilookup is not None:
                selected = _cover_postprocess(selected, all_item_ids, ilookup, vehicles)
                if selected is None:
                    continue
            covered = {iid for b in selected for (iid, *_) in b.items}
            if covered == all_set:
                obj = float(sum(b.cost for b in selected))
                if obj < best_obj - 1e-9:
                    best_obj = obj
                    best_selected = selected

        if best_selected is not None:
            return best_selected, best_obj

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


def generate_columns_pair_seeded(items, vehicles, n_cols, rng, t_end):
    """
    Pair-seeded column generation.

    Picks two compatible items (one large, one medium/small) and seeds them
    into a fresh bin, then greedily fills with remaining items in random order.
    Generates structurally diverse columns vs. single-seed: pairs that "fit
    naturally" lead to high-density packings the LNS pool may not contain.
    """
    out = []
    if len(items) < 2:
        return out
    sorted_items = sorted(items, key=lambda x: -x['vol'])
    n_top = min(len(sorted_items) // 3 + 1, 30)
    big_pool = sorted_items[:n_top]
    rest_pool = sorted_items[n_top:] if len(sorted_items) > n_top else sorted_items

    for _ in range(n_cols):
        if time.monotonic() > t_end:
            break
        if not big_pool or not rest_pool:
            break
        big = big_pool[rng.randrange(len(big_pool))]
        small = rest_pool[rng.randrange(len(rest_pool))]
        if big['id'] == small['id']:
            continue
        for v in vehicles:
            if time.monotonic() > t_end:
                break
            # Capacity precheck
            if big['weight'] + small['weight'] > v['max_weight'] + 1e-9:
                continue
            if big['value'] + small['value'] > v['max_value'] + 1e-9:
                continue
            if big['vol'] + small['vol'] > v['vol'] + 1e-9:
                continue
            if not item_fits(big, v) or not item_fits(small, v):
                continue
            b = Bin3D(v['type'], v['W'], v['D'], v['H'],
                      v['max_weight'], v['max_value'], v['gravity'], v['cost'])
            if not b.try_add(big):
                continue
            if not b.try_add(small):
                continue
            # Greedy fill — large items first then random
            placed_ids = {big['id'], small['id']}
            remaining = [it for it in items if it['id'] not in placed_ids]
            if rng.random() < 0.5:
                remaining.sort(key=lambda x: -x['vol'])
            else:
                rng.shuffle(remaining)
            for it in remaining:
                if not b.cap_ok(it['weight'], it['value']):
                    continue
                b.try_add(it)
            out.append(b)
            break
    return out

def path_relinking(incumbent, pool, ilookup, vehicles, t_end, rng):
    """
    Path Relinking from column pool to incumbent.

    For each candidate column C in the pool that is NOT in the incumbent:
      1. Identify the items in C.
      2. Remove those items from incumbent bins (compute "remainder bins").
      3. Add C as a new bin.
      4. Some incumbent bins may be empty or partially-emptied; reconstruct
         them by repacking remaining items.
      5. Accept if total cost strictly decreases.

    This swaps high-quality alternative bins (found by independent searches)
    into the incumbent, often unlocking improvements neither solution had alone.
    Only accepts strict improvements — never worsens cost.
    """
    if not incumbent or not pool:
        return incumbent, False
    cols = pool.get_columns()
    if not cols:
        return incumbent, False

    cur = [b.copy() for b in incumbent]
    cur_keys = {frozenset(rec[0] for rec in b.items) for b in cur}
    cur_cost = _cost(cur)

    # Rank candidate columns by efficiency: low cost-per-item-covered.
    # Skip columns already in incumbent and tiny ones (1-item columns can't
    # beat removing-and-resplitting from incumbent).
    candidates = []
    for c in cols:
        if len(c.items) < 2:
            continue
        key = frozenset(rec[0] for rec in c.items)
        if key in cur_keys:
            continue
        eff = c.cost / max(1, len(c.items))
        candidates.append((eff, c))
    candidates.sort(key=lambda x: x[0])
    candidates = [c for _, c in candidates[:80]]  # cap workload

    # Build item -> set of incumbent bin indices that contain it
    improved_any = False
    for cand in candidates:
        if time.monotonic() > t_end:
            break

        cand_ids = [rec[0] for rec in cand.items]
        cand_id_set = set(cand_ids)

        # Find bins touched by this candidate's items
        item_to_bin = {}
        for bi, b in enumerate(cur):
            for rec in b.items:
                if rec[0] in cand_id_set:
                    item_to_bin[rec[0]] = bi
        touched = sorted(set(item_to_bin.values()), reverse=True)
        if not touched:
            continue

        # Build trial: remove cand items from touched bins (rebuild those bins)
        trial = [b.copy() for b in cur]
        leftover_items = []
        new_bins_for_touched = []
        ok = True
        for bi in touched:
            old_b = trial[bi]
            keep_items = [ilookup[rec[0]] for rec in old_b.items
                          if rec[0] not in cand_id_set]
            if not keep_items:
                continue  # bin is fully emptied
            nb = Bin3D(old_b.vtype, old_b.W, old_b.D, old_b.H,
                       old_b.max_weight, old_b.max_value, old_b.gravity, old_b.cost)
            for it in sorted(keep_items, key=lambda x: -x['vol']):
                if not nb.try_add(it):
                    leftover_items.append(it)
            new_bins_for_touched.append((bi, nb))
        if not ok:
            continue

        # Build resulting bin list
        result = [b.copy() for k, b in enumerate(trial) if k not in touched]
        for _, nb in new_bins_for_touched:
            if nb.items:
                result.append(nb)
        result.append(cand.copy())

        # Place leftover items
        infeasible = False
        for it in sorted(leftover_items, key=lambda x: -x['vol']):
            if time.monotonic() > t_end:
                infeasible = True
                break
            placed = False
            for bb in sorted(result, key=lambda x: x.rem_vol()):
                if bb.try_add(it):
                    placed = True
                    break
            if not placed:
                nb2 = open_bin(it, vehicles)
                if nb2 is None:
                    infeasible = True
                    break
                result.append(nb2)
        if infeasible:
            continue

        new_cost = _cost(result)
        if new_cost < cur_cost - 1e-9:
            cur = result
            cur_cost = new_cost
            cur_keys = {frozenset(rec[0] for rec in b.items) for b in cur}
            improved_any = True
    return cur, improved_any


def mine_columns_from_incumbent(best_bins, ilookup, vehicle_cycle, pool, t_end, rng):
    """
    Intensification around incumbent:
    destroy a subset of bins, repack removed items, run short LNS,
    and inject resulting bins into the column pool.

    Improvements vs. baseline:
      - Wider destruction (3-8 bins, scaled by incumbent size).
      - Probes targeted at the most expensive bins (higher leverage).
      - Tries 2 vehicle orderings per destruction for diversity.
      - Pool snapshot before AND after the local LNS.
    """
    if not best_bins:
        return
    base = [b.copy() for b in best_bins]
    if len(base) < 2:
        return

    n_base = len(base)
    # Scale destruction range to incumbent: small instances → small k, large → bigger k.
    k_min = 3 if n_base >= 6 else 2
    k_max = min(n_base - 1, max(k_min + 2, n_base // 6 + 4, 8))

    iter_n = 0
    while time.monotonic() < t_end:
        iter_n += 1

        # Mix of random and "expensive-bin-targeted" destructions.
        target_expensive = (iter_n % 3 == 0)
        work = [b.copy() for b in base]
        k = rng.randint(k_min, k_max)

        if target_expensive:
            order = sorted(range(len(work)), key=lambda i: -work[i].cost)
            top = order[: max(k * 2, 6)]
            rng.shuffle(top)
            chosen = sorted(top[:k], reverse=True)
        else:
            chosen = sorted(rng.sample(range(len(work)), k), reverse=True)

        removed = []
        for idx in chosen:
            for rec in work[idx].items:
                removed.append(ilookup[rec[0]])
            del work[idx]

        # Try this destruction with up to 2 different vehicle orderings.
        n_vehs_try = 2 if rng.random() < 0.6 else 1
        veh_indices = rng.sample(range(len(vehicle_cycle)),
                                 min(n_vehs_try, len(vehicle_cycle)))

        for vi in veh_indices:
            if time.monotonic() > t_end:
                return
            vehs = vehicle_cycle[vi]
            trial = [b.copy() for b in work]
            ok = True
            for it in sorted(removed, key=lambda x: -x['vol']):
                if time.monotonic() > t_end:
                    return
                placed = False
                for b in sorted(trial, key=lambda bb: bb.rem_vol()):
                    if b.try_add(it):
                        placed = True
                        break
                if not placed:
                    nb = open_bin(it, vehs)
                    if nb:
                        trial.append(nb)
                        placed = True
                if not placed:
                    ok = False
                    break

            if not ok:
                continue

            # Snapshot raw repack first — already a valid column set.
            pool.add_solution(trial)

            # Short local LNS for refinement.
            local_end = min(time.monotonic() + 5.0, t_end)
            trial = lns(trial, ilookup, vehs, local_end, rng,
                        verbose=False, pool=pool)
            pool.add_solution(trial)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_items(df):
    out=[]
    for iid,row in df.iterrows():
        rots=[int(c) for c in str(row['allowedRotations'])]
        w,d,h=float(row['width']),float(row['depth']),float(row['height'])
        urots = unique_rots(w,d,h,rots)
        vol = w*d*h
        base_area = max(iw*id_ for (_,iw,id_,_) in urots)
        out.append({'id':iid,'w':w,'d':d,'h':h,
                    'weight':float(row['weight']),'value':float(row['value']),
                    'urots':urots,
                    'vol':vol,'maxdim':max(w,d,h),
                    'base_area':base_area,
                    'density_w':float(row['weight'])/max(vol,1e-9),
                    'density_v':float(row['value'])/max(vol,1e-9)})
    return out

def parse_vehicles(df):
    out=[]
    for vtype,row in df.iterrows():
        mv=row.get('maxValue',float('nan'))
        if mv is None or (isinstance(mv,float) and math.isnan(mv)): mv=float('inf')
        W = float(row['width'])
        D = float(row['depth'])
        H = float(row['height'])
        max_weight = float(row['maxWeight'])
        max_value = float(mv)
        cost = float(row['cost'])
        vol = W * D * H
        cpv = cost / max(vol, 1.0)
        wpc = cost / max(max_weight, 1.0)
        vpc = 0.0 if max_value >= 1e18 else cost / max(max_value, 1.0)
        out.append({'type':vtype,
                    'W':W,'D':D,'H':H,
                    'max_weight':max_weight,'max_value':max_value,
                    'cost':cost,'gravity':float(row['gravityStrength']),
                    'vol':vol,'cpv':cpv,'wpc':wpc,'vpc':vpc})
    return out

def vehicle_orderings(vehicles):
    return {
        'cost': sorted(vehicles, key=lambda v: (v['cost'], v['cpv'], -v['vol'])),
        'cpv': sorted(vehicles, key=lambda v: (v['cpv'], v['cost'], -v['vol'])),
        'capacity': sorted(vehicles, key=lambda v: (v['wpc'], v['vpc'], v['cost'])),
        'large': sorted(vehicles, key=lambda v: (-v['vol'], v['cost'])),
        'big_eff': sorted(vehicles, key=lambda v: (-(v['vol'] / max(v['cost'], 1e-9)), v['cost'])),
        'balanced': sorted(
            vehicles,
            key=lambda v: (0.55*v['cpv'] + 0.25*v['wpc'] + 0.20*v['vpc'], v['cost']),
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SOLVER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class solver_364130(AbstractSolver):
    """3-D Bin Packing: EP + GRASP + LNS + Column Generation + Set Partition MILP."""

    SOLVE_SECONDS  = int(os.getenv('SOLVER_364130_TIME_LIMIT', '600'))
    DETERMINISTIC  = os.getenv('SOLVER_364130_DETERMINISTIC', '0') != '0'
    N_THREADS      = 1 if DETERMINISTIC else min(4, max(1, (os.cpu_count() or 4)))
    VERBOSE        = os.getenv('SOLVER_364130_VERBOSE', '1') != '0'
    BASE_SEED      = int(os.getenv('SOLVER_364130_SEED', '15'))
    HIGHS_PARALLEL = os.getenv(
        'SOLVER_364130_HIGHS_PARALLEL',
        'off' if DETERMINISTIC else 'on',
    )
  

    def __init__(self, inst):
        super().__init__(inst)
        self.name = 'solver_364130'

    def solve(self):
        t0   = time.monotonic()
        tend = t0 + self.SOLVE_SECONDS
        log  = print if self.VERBOSE else (lambda *a, **k: None)

        log(f"\n{'═'*65}")
        log(f"  Dataset  :  {self.inst.name}")
        #log(f"  Seed     :  {self.BASE_SEED}  (deterministic={self.DETERMINISTIC})")
        log(f"  Threads  :  {self.N_THREADS}  (HiGHS parallel={self.HIGHS_PARALLEL})")
        log(f"{'═'*65}")

        items         = parse_items(self.inst.df_items)
        vehicles_all  = parse_vehicles(self.inst.df_vehicles)
        # Item hardness is computed from how many vehicle types can host the item
        # and how tight those fits are on key capacities; used only to drive
        # generic ordering heuristics, not dataset-specific tuning.
        for it in items:
            feasible_vs = [
                v for v in vehicles_all
                if _vehicle_accepts_item(it, v)
            ]
            it['fit_count'] = len(feasible_vs)
            if not feasible_vs:
                it['hardness'] = 1e9
                continue
            min_cost = min(v['cost'] for v in feasible_vs)
            best_w_ratio = min(v['max_weight'] / max(it['weight'], 1e-9) for v in feasible_vs)
            best_vol_ratio = min(v['vol'] / max(it['vol'], 1e-9) for v in feasible_vs)
            if it['value'] > 1e-9:
                best_val_ratio = min(v['max_value'] / max(it['value'], 1e-9) for v in feasible_vs)
            else:
                best_val_ratio = 10.0
            it['hardness'] = (
                4.0 / max(1, it['fit_count'])
                + 0.6 / max(1e-9, best_w_ratio)
                + 0.6 / max(1e-9, best_vol_ratio)
                + 0.2 / max(1e-9, best_val_ratio)
                + 0.02 * min_cost
            )
        v_orders      = vehicle_orderings(vehicles_all)
        vehicles      = v_orders['cost']
        mono_vehicle_lists = [[v] for v in v_orders['large'][:min(3, len(v_orders['large']))]]
        vehicle_cycle = [
            v_orders['cost'],
            v_orders['cpv'],
            v_orders['capacity'],
            v_orders['balanced'],
            v_orders['large'],
            v_orders['big_eff'],
            *mono_vehicle_lists,
        ]
        name_seed     = sum((i + 1) * ord(ch) for i, ch in enumerate(self.inst.name))
        seed0         = 1009 + name_seed + self.BASE_SEED * 100_003
        portfolio_seed0 = [seed0]
        if not self.DETERMINISTIC:
            for s in (2, 7, 19):
                if s != self.BASE_SEED:
                    portfolio_seed0.append(1009 + name_seed + s * 100_003)

        ilookup = {it['id']: it for it in items}
        all_ids = sorted(ilookup.keys())

        log(f"  Items    : {len(items)}")
        for v in vehicles:
            mv = f"{v['max_value']:.0f}" if v['max_value'] < 1e14 else "∞"
            log(f"  Vehicle  : {v['type']:14s}  {v['W']}×{v['D']}×{v['H']}"
                f"  maxW={v['max_weight']:.0f}  maxV={mv}"
                f"  cost={v['cost']:.2f}  grav={v['gravity']}%")

        # Shared state
        pool      = ColumnPool()
        best_bins = [None]
        best_cost = [float('inf')]
        lock      = threading.Lock()

        def update_best(bins, label=''):
            c = _cost(bins)
            with lock:
                if c < best_cost[0] - 1e-9:
                    best_cost[0] = c
                    best_bins[0] = [b.copy() for b in bins]
                    log(f'    ★ NEW BEST {label}: cost={c:.2f}  bins={len(bins)}')
                    return True
            return False

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 1 — Parallel construction
        # ─────────────────────────────────────────────────────────────────────
        log(f"\n  Phase 1  —  parallel construction")
        t_p1 = t0 + min(45.0, self.SOLVE_SECONDS * 0.10)

        by_vol       = build_sequence(items, 'vol', random.Random(seed0 + 101))
        by_weight    = build_sequence(items, 'weight', random.Random(seed0 + 102))
        by_maxdim    = build_sequence(items, 'maxdim', random.Random(seed0 + 103))
        by_value     = build_sequence(items, 'value', random.Random(seed0 + 104))
        by_footprint = build_sequence(items, 'footprint', random.Random(seed0 + 105))
        by_densw     = build_sequence(items, 'densw', random.Random(seed0 + 106))
        by_densv     = build_sequence(items, 'densv', random.Random(seed0 + 107))
        by_mixed     = build_sequence(items, 'mixed', random.Random(seed0 + 108))
        by_hard      = build_sequence(items, 'hard', random.Random(seed0 + 109))

        def _worker(label, seq, bf, alpha, seed, vehs):
            rng = random.Random(seed)
            bins, unp = grasp(seq, vehs, alpha, rng, t_p1, bf, honor_order=True)
            if unp:
                return label, None, float('inf')
            bins = lns(bins, ilookup, vehs, t_p1, rng, verbose=False, pool=pool)
            pool.add_solution(bins)
            return label, bins, _cost(bins)

        def _worker_resource_packed(label, mode, seed):
            """Run weight-FFD or volume-FFD construction (no LNS — fast).
            Adds bins to pool so MILP set-partition can pick them.
            Direct attack on weight-binding / volume-binding pathologies."""
            if mode == 'weight':
                bins = construct_weight_packed(items, vehicles, t_p1)
            elif mode == 'volume':
                bins = construct_volume_packed(items, vehicles, t_p1)
            else:
                return label, None, float('inf')
            if bins is None:
                return label, None, float('inf')
            pool.add_solution(bins)
            return label, bins, _cost(bins)

        configs = [
            ('vol-cost',      by_vol,    False, 0.00, seed0 + 1, v_orders['cost']),
            ('vol-cpv-bf',    by_vol,    True,  0.00, seed0 + 2, v_orders['cpv']),
            ('w-cap',         by_weight, False, 0.00, seed0 + 3, v_orders['capacity']),
            ('maxd-bal',      by_maxdim, False, 0.00, seed0 + 4, v_orders['balanced']),
            ('val-cpv',       by_value,  False, 0.00, seed0 + 5, v_orders['cpv']),
            ('val-cost-bf',   by_value,  True,  0.00, seed0 + 6, v_orders['cost']),
            ('rnd-cost',      items,     False, 0.14, seed0 + 7, v_orders['cost']),
            ('rnd-bal-bf',    items,     True,  0.14, seed0 + 8, v_orders['balanced']),
            ('large-vol-bf',  by_vol,    True,  0.00, seed0 + 9, v_orders['large']),
            ('bigeff-maxd',   by_maxdim, True,  0.00, seed0 + 10, v_orders['big_eff']),
            ('fp-bal',        by_footprint, True, 0.00, seed0 + 11, v_orders['balanced']),
            ('densw-cap',     by_densw, True, 0.00, seed0 + 12, v_orders['capacity']),
            # Extra-diverse starters: cover orderings/vehicle priorities not in the
            # original 12-config portfolio, so different starting basins are reached.
            ('densv-bigeff',  by_densv, False, 0.00, seed0 + 13, v_orders['big_eff']),
            ('mixed-cpv-bf',  by_mixed, True,  0.05, seed0 + 14, v_orders['cpv']),
            ('rnd-large',     items,    True,  0.20, seed0 + 15, v_orders['large']),
            ('fp-cost-bf',    by_footprint, True, 0.00, seed0 + 16, v_orders['cost']),
            ('hard-cost',     by_hard,  True,  0.00, seed0 + 17, v_orders['cost']),
            ('hard-cap',      by_hard,  True,  0.04, seed0 + 18, v_orders['capacity']),
        ]
        for j, mono in enumerate(mono_vehicle_lists):
            configs.append((f"mono-{mono[0]['type']}", by_vol, True, 0.06, seed0 + 20 + j, mono))

        # Resource-packed constructions: direct attack on weight/volume-binding
        # instances. Run in same thread pool as GRASP workers.
        resource_configs = [
            ('w-pack',  'weight', seed0 + 50),
            ('v-pack',  'volume', seed0 + 51),
        ]

        with ThreadPoolExecutor(max_workers=self.N_THREADS) as ex:
            # Submit resource-packed workers FIRST — they're fast (deterministic
            # constructions, no LNS) and their bins go to the column pool.
            fs = {}
            for (rlabel, rmode, rseed) in resource_configs:
                fs[ex.submit(_worker_resource_packed, rlabel, rmode, rseed)] = rlabel
            for (label, seq, bf, alpha, seed, vehs) in configs:
                fs[ex.submit(_worker, label, seq, bf, alpha, seed, vehs)] = label
            for f in as_completed(fs):
                try:
                    label, bins, _ = f.result()
                    if bins:
                        update_best(bins, f'p1-{label}')
                except Exception as e:
                    log(f'    p1 worker error: {e}')

        if best_bins[0] is None:
            log('  WARN: emergency fallback — 1 item per bin')
            fb = []
            for it in items:
                b = open_bin(it, vehicles)
                if b:
                    fb.append(b)
                    pool.add_bin(b)
            best_bins[0] = fb
            best_cost[0] = _cost(fb)

        log(f"\n  After Phase 1: cost={best_cost[0]:.2f}  bins={len(best_bins[0])}"
            f"  columns={pool.size()}")

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 2 — LNS + GRASP restarts
        # ─────────────────────────────────────────────────────────────────────
        log(f"\n  Phase 2  —  GRASP restarts + LNS  (column harvesting)")
        t_p2      = tend - 75.0
        restarts  = [0]

        def _deep_lns():
            with lock:
                snap = [b.copy() for b in best_bins[0]]
            rng = random.Random(seed0 + 42)
            improved = lns(snap, ilookup, vehicles, t_p2, rng,
                           verbose=self.VERBOSE, pool=pool)
            update_best(improved, 'lns-deep')

        def _restart_worker(seed_base):
            rng = random.Random(seed_base)
            modes = ['vol', 'weight', 'value', 'maxdim', 'footprint', 'densw', 'densv', 'mixed', 'shuffle']
            if self.DETERMINISTIC:
                n_restarts = max(40, min(220, len(items) // 6))
                for _ in range(n_restarts):
                    if time.monotonic() >= t_p2:
                        break
                    slot = max(2.0, min(7.0, (t_p2 - time.monotonic()) / max(1, n_restarts)))
                    alpha = rng.uniform(0.05, 0.35)
                    bf    = rng.random() < 0.5
                    vehs  = vehicle_cycle[rng.randrange(len(vehicle_cycle))]
                    mode = modes[rng.randrange(len(modes))]
                    seq = build_sequence(items, mode, rng)
                    local_end = min(time.monotonic() + slot, t_p2)
                    bins, unp = grasp(seq, vehs, alpha, rng, local_end, bf, honor_order=True)
                    with lock:
                        restarts[0] += 1
                    if unp:
                        continue
                    pool.add_solution(bins)
                    with lock:
                        thresh = best_cost[0] * 1.10
                    if _cost(bins) <= thresh:
                        lns_end = min(time.monotonic() + slot * 0.65, t_p2)
                        bins = lns(bins, ilookup, vehs, lns_end, rng, verbose=False, pool=pool)
                    update_best(bins, 'restart')
                return

            while time.monotonic() < t_p2:
                alpha = rng.uniform(0.05, 0.35)
                bf    = rng.random() < 0.5
                vehs  = vehicle_cycle[rng.randrange(len(vehicle_cycle))]
                mode = modes[rng.randrange(len(modes))]
                seq = build_sequence(items, mode, rng)
                bins, unp = grasp(seq, vehs, alpha, rng, t_p2, bf, honor_order=True)
                with lock:
                    restarts[0] += 1
                if unp:
                    continue
                pool.add_solution(bins)
                with lock:
                    thresh = best_cost[0] * 1.10
                if _cost(bins) <= thresh:
                    tl   = min(time.monotonic() + 35.0, t_p2)
                    bins = lns(bins, ilookup, vehs, tl, rng, verbose=False, pool=pool)
                update_best(bins, 'restart')

        with ThreadPoolExecutor(max_workers=self.N_THREADS) as ex:
            futures = [ex.submit(_deep_lns)]
            for k in range(self.N_THREADS - 1):
                root = portfolio_seed0[k % len(portfolio_seed0)]
                futures.append(ex.submit(_restart_worker, root + 3000 + k * 1337))
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    log(f'  Phase 2 error: {e}')

        log(f"\n  After Phase 2: cost={best_cost[0]:.2f}  bins={len(best_bins[0])}"
            f"  columns={pool.size()}  restarts={restarts[0]}")

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 3 — Targeted column generation + Set Partition MILP
        # ─────────────────────────────────────────────────────────────────────
        log(f"\n  Phase 3  —  Set Partition ILP  ({int(tend - time.monotonic())} s left)")

        rng_cg = random.Random(seed0 + 5555)
        # Pre-seed: weight-FFD and volume-FFD constructions before MILP. Phase 1's
        # _worker_resource_packed runs the same builders but on a contended ~45s
        # budget shared with 12 GRASP workers, so it may not exhaust all
        # primary/ordering combinations; this pass gets a dedicated 10s and adds
        # any extra columns the pool didn't already have.
        t_seed_end = min(time.monotonic() + 10.0, tend - 55.0)
        if t_seed_end > time.monotonic() + 1.0:
            try:
                wb = construct_weight_packed(items, vehicles, t_seed_end)
                if wb:
                    pool.add_solution(wb)
                    log(f"  Pre-seed weight-packed: {len(wb)} bins, "
                        f"cost={_cost(wb):.2f}")
            except Exception as e:
                log(f"  weight-packed seed failed: {e}")
            try:
                vb = construct_volume_packed(items, vehicles, t_seed_end)
                if vb:
                    pool.add_solution(vb)
                    log(f"  Pre-seed volume-packed: {len(vb)} bins, "
                        f"cost={_cost(vb):.2f}")
            except Exception as e:
                log(f"  volume-packed seed failed: {e}")

        # Diversified pre-seed: weight-FFD with top-3 CPW primaries plus a
        # cost-perturbed primary pick. The standard pre-seed only uses the
        # cheapest CPW vehicle, so without this the column pool is biased
        # toward bin shapes from one vehicle type.
        t_div_end = min(time.monotonic() + 3.0, tend - 52.0)
        if t_div_end > time.monotonic() + 0.5:
            try:
                div_cands = construct_weight_packed_diverse(
                    items, vehicles, t_div_end, rng_cg
                )
                for div_bins in div_cands:
                    pool.add_solution(div_bins)
                if div_cands:
                    log(f"  Diverse pre-seed: {len(div_cands)} candidate "
                        f"covers added")
            except Exception as e:
                log(f"  diverse pre-seed failed: {e}")


        # Split CG budget: ~70% item-centric, ~30% pair-seeded for diversity.
        t_cg_total = min(time.monotonic() + 22.0, tend - 50.0)
        t_cg_pair_split = time.monotonic() + max(2.0, (t_cg_total - time.monotonic()) * 0.30)

        # Pair-seeded columns first (cheap, structurally diverse)
        for vi in range(min(3, len(vehicle_cycle))):
            if time.monotonic() > t_cg_pair_split:
                break
            n_pair = max(20, len(items) // 6)
            for b in generate_columns_pair_seeded(
                items, vehicle_cycle[vi], n_pair, rng_cg, t_cg_pair_split
            ):
                pool.add_bin(b)

        # Item-centric columns on the remaining budget
        for i, item in enumerate(sorted(items, key=lambda x: -x['vol'])):
            if time.monotonic() > t_cg_total:
                break
            vehs = vehicle_cycle[i % len(vehicle_cycle)]
            for b in generate_columns_for_item(item, items, vehs, 6, rng_cg, t_cg_total):
                pool.add_bin(b)

        cols = pool.get_columns()
        log(f"  Column pool size: {len(cols)}")

        milp_budget = min(34.0, tend - time.monotonic() - 24.0)
        milp_cols   = min(3000, max(1200, len(all_ids) // 2 + 800))
        milp_improved = False
        if milp_budget > 5.0 and cols:
            log(f"  Running MILP  (budget={milp_budget:.0f}s, cols={milp_cols}) ...")
            with lock:
                warm_keys = [frozenset(rec[0] for rec in b.items) for b in (best_bins[0] or [])]
            selected, milp_cost = solve_set_partition(
                cols, all_ids, milp_budget, max_cols=milp_cols,
                ilookup=ilookup, vehicles=vehicles_all, warm_keys=warm_keys,
                highs_parallel=self.HIGHS_PARALLEL
            )
            if selected is not None:
                log(f"  MILP solution: cost={milp_cost:.2f}  bins={len(selected)}")
                milp_improved = update_best(selected, 'milp') or milp_improved
            else:
                log('  MILP: no feasible solution found in column pool')
        else:
            log('  MILP skipped (budget too small or no columns)')

        # Extra intensification: mine columns near incumbent, then rerun MILP.
        t_mine = min(time.monotonic() + 28.0, tend - 22.0)
        if t_mine > time.monotonic() + 2.0 and best_bins[0]:
            with lock:
                incumbent = [b.copy() for b in best_bins[0]]
            mine_columns_from_incumbent(
                incumbent, ilookup, vehicle_cycle, pool, t_mine, random.Random(seed0 + 7777)
            )
            cols2 = pool.get_columns()
            log(f"  Column pool after intensification: {len(cols2)}")
            milp_budget2 = min(12.0, tend - time.monotonic() - 10.0)
            milp_cols2 = min(3200, milp_cols + 500)
            run_milp2 = milp_improved or (len(cols2) >= len(cols) + 120)
            if milp_budget2 > 4.0 and cols2 and run_milp2:
                log(f"  Running MILP-2 (budget={milp_budget2:.0f}s, cols={milp_cols2}) ...")
                with lock:
                    warm_keys2 = [frozenset(rec[0] for rec in b.items) for b in (best_bins[0] or [])]
                selected2, milp_cost2 = solve_set_partition(
                    cols2, all_ids, milp_budget2, max_cols=milp_cols2,
                    ilookup=ilookup, vehicles=vehicles_all, warm_keys=warm_keys2,
                    highs_parallel=self.HIGHS_PARALLEL
                )
                if selected2 is not None:
                    log(f"  MILP-2 solution: cost={milp_cost2:.2f}  bins={len(selected2)}")
                    update_best(selected2, 'milp-2')
            elif milp_budget2 > 4.0 and cols2:
                log("  MILP-2 skipped (not promising after MILP-1)")

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 4 — Final polish
        # ─────────────────────────────────────────────────────────────────────
        remaining = int(tend - time.monotonic())
        log(f"\n  Phase 4  —  final LNS polish  ({remaining} s left)")

        # Path Relinking: substitute alternative columns from the pool into
        # the incumbent. Often unlocks improvements neither GRASP+LNS nor
        # the MILP found on their own.
        if remaining > 25 and best_bins[0]:
            with lock:
                snap_pr = [b.copy() for b in best_bins[0]]
            pr_end = min(time.monotonic() + 8.0, tend - 18.0)
            pr_result, pr_imp = path_relinking(
                snap_pr, pool, ilookup, vehicles, pr_end,
                random.Random(seed0 + 31313)
            )
            if pr_imp:
                update_best(pr_result, 'path-relink')

        with lock:
            snap = [b.copy() for b in best_bins[0]]
        final = lns(snap, ilookup, vehicles, tend - 16.0, random.Random(seed0 + 9999),
                    verbose=self.VERBOSE, pool=pool)
        update_best(final, 'final-lns-cost')

        if time.monotonic() < tend - 10.0:
            with lock:
                snap = [b.copy() for b in best_bins[0]]
            final = lns(snap, ilookup, v_orders['balanced'], tend - 10.0, random.Random(seed0 + 1117),
                        verbose=False, pool=pool)
            update_best(final, 'final-lns-balanced')

        # Multi-seed late intensification: short bursts on different vehicle orders.
        seed = seed0 + 2221
        late_modes = ['vol', 'weight', 'value', 'maxdim', 'footprint', 'densw', 'densv', 'mixed', 'shuffle']
        if self.DETERMINISTIC:
            n_late = 24
            for i in range(n_late):
                if time.monotonic() >= tend - 2.0:
                    break
                rem = tend - time.monotonic() - 2.0
                if rem <= 0:
                    break
                slot = max(1.2, min(4.5, rem / max(1, n_late - i)))
                rng_local = random.Random(seed)
                vehs = vehicle_cycle[seed % len(vehicle_cycle)]
                burst_end = min(time.monotonic() + slot, tend - 2.0)
                if seed % 2 == 0:
                    with lock:
                        snap = [b.copy() for b in best_bins[0]]
                    cand = lns(snap, ilookup, vehs, burst_end, rng_local,
                               verbose=False, pool=pool)
                    update_best(cand, f'lateL-{seed % 1000}')
                else:
                    mode = late_modes[seed % len(late_modes)]
                    seq = build_sequence(items, mode, rng_local)
                    alpha = rng_local.uniform(0.02, 0.22)
                    bf = rng_local.random() < 0.6
                    cand, unp = grasp(seq, vehs, alpha, rng_local, burst_end, bf, honor_order=True)
                    if not unp:
                        cand = lns(cand, ilookup, vehs, burst_end, rng_local,
                                   verbose=False, pool=pool)
                        pool.add_solution(cand)
                        update_best(cand, f'lateR-{seed % 1000}')
                seed += 97
        else:
            # Reserve a small tail for the bumped MILP-final and post-opt.
            while time.monotonic() < tend - 7.0:
                rng_local = random.Random(seed)
                vehs = vehicle_cycle[seed % len(vehicle_cycle)]
                burst_end = min(time.monotonic() + 4.5, tend - 7.0)
                if seed % 2 == 0:
                    with lock:
                        snap = [b.copy() for b in best_bins[0]]
                    cand = lns(snap, ilookup, vehs, burst_end, rng_local,
                               verbose=False, pool=pool)
                    update_best(cand, f'lateL-{seed % 1000}')
                else:
                    mode = late_modes[seed % len(late_modes)]
                    seq = build_sequence(items, mode, rng_local)
                    alpha = rng_local.uniform(0.02, 0.22)
                    bf = rng_local.random() < 0.6
                    cand, unp = grasp(seq, vehs, alpha, rng_local, burst_end, bf, honor_order=True)
                    if not unp:
                        cand = lns(cand, ilookup, vehs, burst_end, rng_local,
                                   verbose=False, pool=pool)
                        pool.add_solution(cand)
                        update_best(cand, f'lateR-{seed % 1000}')
                seed += 97

        # Second Path Relinking round — pool has grown a lot in late bursts.
        rem = tend - time.monotonic()
        if rem > 9.0 and best_bins[0]:
            with lock:
                snap_pr2 = [b.copy() for b in best_bins[0]]
            pr2_end = min(time.monotonic() + 4.0, tend - 5.0)
            pr2_result, pr2_imp = path_relinking(
                snap_pr2, pool, ilookup, vehicles, pr2_end,
                random.Random(seed0 + 42424)
            )
            if pr2_imp:
                update_best(pr2_result, 'path-relink-2')

        # Deterministic incumbent polishing (cannot worsen objective).
        # Trimmed budget so the bumped MILP-final actually gets the bigger
        # window it was bumped to (otherwise post-opt's cap-at-tend-3 forces
        # MILP-final to start with rem~3 and use only ~2s).
        rem = tend - time.monotonic()
        if rem > 4.0:
            with lock:
                snap = [b.copy() for b in best_bins[0]]
            polish_end = min(tend - 6.0, time.monotonic() + min(6.0, rem - 6.0))
            polished = post_optimize_bins(snap, ilookup, v_orders['cost'], polish_end, pool=pool)
            update_best(polished, 'post-opt')

        # Last exact re-optimization over full column pool — given a longer
        # budget and larger column cap to fully exploit the pool. Reaches
        # combinations that the time-constrained MILP-1/MILP-2 missed.
        rem = tend - time.monotonic()
        if rem > 2.5:
            cols3 = pool.get_columns()
            if cols3:
                budget3 = min(10.0, rem - 0.8)
                cols_cap3 = min(3500, max(1500, len(all_ids) + 700))
                with lock:
                    warm_keys3 = [frozenset(rec[0] for rec in b.items) for b in (best_bins[0] or [])]
                selected3, milp_cost3 = solve_set_partition(
                    cols3, all_ids, budget3, max_cols=cols_cap3,
                    ilookup=ilookup, vehicles=vehicles_all, warm_keys=warm_keys3,
                    highs_parallel=self.HIGHS_PARALLEL
                )
                if selected3 is not None:
                    update_best(selected3, 'milp-final')

        # ─────────────────────────────────────────────────────────────────────
        # Build output with final safety repair
        # ─────────────────────────────────────────────────────────────────────
        with lock:
            result = [b.copy() for b in (best_bins[0] or [])]
            total  = best_cost[0]
        elapsed = time.monotonic() - t0

        if not result:
            result = []
            for it in items:
                b = open_bin(it, vehicles)
                if b:
                    result.append(b)
            total = _cost(result)

        rows = []
        seen = set()
        idx_v = 0
        for b in result:
            added = False
            for (iid, x, y, z, iw, id_, ih, orient) in b.items:
                if iid in seen:
                    continue
                rows.append({
                    'type_vehicle': b.vtype,
                    'idx_vehicle': idx_v,
                    'id_item': iid,
                    'x_origin': round(x, 9),
                    'y_origin': round(y, 9),
                    'z_origin': round(z, 9),
                    'orient': int(orient),
                })
                seen.add(iid)
                added = True
            if added:
                idx_v += 1

        missing = [iid for iid in all_ids if iid not in seen]
        if missing:
            log(f"  Repair step: {len(missing)} missing items")
            for iid in missing:
                nb = open_bin(ilookup[iid], vehicles)
                if nb is None:
                    continue
                for (iid2, x, y, z, iw, id_, ih, orient) in nb.items:
                    rows.append({
                        'type_vehicle': nb.vtype,
                        'idx_vehicle': idx_v,
                        'id_item': iid2,
                        'x_origin': round(x, 9),
                        'y_origin': round(y, 9),
                        'z_origin': round(z, 9),
                        'orient': int(orient),
                    })
                    seen.add(iid2)
                idx_v += 1

        log(f"\n{'═'*65}")
        log(f"  FINAL  cost={total:.2f}  bins={idx_v}  time={elapsed:.1f}s")
        log(f"{'═'*65}\n")

        self.sol = {
            'type_vehicle': [r['type_vehicle'] for r in rows],
            'idx_vehicle': [r['idx_vehicle'] for r in rows],
            'id_item': [r['id_item'] for r in rows],
            'x_origin': [r['x_origin'] for r in rows],
            'y_origin': [r['y_origin'] for r in rows],
            'z_origin': [r['z_origin'] for r in rows],
            'orient': [r['orient'] for r in rows],
        }
        self.write_solution_to_file()
        log(f"  → results/sol_{self.inst.name}_{self.name}.csv")
        return result, total
