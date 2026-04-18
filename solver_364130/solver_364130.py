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

def lns(bins, ilookup, vehicles, t_end, rng=None, verbose=False, pool=None):
    if rng is None: rng=random.Random()
    cur=list(bins); stag=0
    destroy_rate=0.33
    elim_failures=0
    while time.monotonic()<t_end:
        imp=False
        new,ok=op_elim(cur,ilookup,vehicles,t_end)
        if ok:
            cur=new; stag=0; imp=True
            destroy_rate=max(0.25, destroy_rate*0.85)
            elim_failures=0
            if pool: pool.add_solution(cur)
            if verbose: print(f'    [ELIM]  bins={len(cur):3d}  cost={_cost(cur):.2f}')
            continue
        else:
            elim_failures += 1
            if elim_failures >= 5:
                destroy_rate=min(0.85, destroy_rate + 0.25)
                elim_failures = 0
        new,ok=op_retype(cur,ilookup,vehicles,rng,t_end)
        if ok:
            cur=new; stag=0; imp=True
            destroy_rate=max(0.25, destroy_rate*0.92)
            if pool: pool.add_solution(cur)
            if verbose: print(f'    [RETYPE] bins={len(cur):3d}  cost={_cost(cur):.2f}')
        new,ok=op_shake(cur,ilookup,vehicles,rng,t_end)
        if ok:
            cur=new; stag=0; imp=True
            destroy_rate=max(0.25, destroy_rate*0.92)
            if pool: pool.add_solution(cur)
            if verbose: print(f'    [SHAKE] bins={len(cur):3d}  cost={_cost(cur):.2f}')
        new,ok=op_merge3(cur,ilookup,vehicles,rng,t_end)
        if ok:
            cur=new; stag=0; imp=True
            destroy_rate=max(0.25, destroy_rate*0.90)
            if pool: pool.add_solution(cur)
            if verbose: print(f'    [MERGE3] bins={len(cur):3d}  cost={_cost(cur):.2f}')
        new,ok=op_eject(cur,ilookup,vehicles,rng,t_end,destroy_rate=destroy_rate)
        if ok:
            cur=new; stag=0; imp=True
            destroy_rate=max(0.25, destroy_rate*0.95)
            if pool: pool.add_solution(cur)
            if verbose: print(f'    [EJECT] bins={len(cur):3d}  cost={_cost(cur):.2f}')
        if not imp:
            stag+=1
            if stag>=40: break
    return cur

def post_optimize_bins(bins, ilookup, vehicles, t_end, pool=None):
    """
    Deterministic intensification pass for the incumbent.
    Applies only strict-improvement moves, so objective cannot worsen.
    """
    cur = [b.copy() for b in bins]
    rng = random.Random(987654321)
    while time.monotonic() < t_end:
        imp = False
        for op in (
            lambda x: op_elim(x, ilookup, vehicles, t_end),
            lambda x: op_retype(x, ilookup, vehicles, rng, t_end),
            lambda x: op_merge3(x, ilookup, vehicles, rng, t_end),
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
    A, costs, _ = _build_coverage_matrix(columns, all_item_ids)

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
        h.setOptionValue('time_limit', float(max(1.0, t_budget)))
        h.setOptionValue('mip_rel_gap', float(gap))

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
    columns   = _filter_columns_lp(columns, all_item_ids,
                                   max_cols=max_cols, lp_budget=lp_budget)

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
    gap = 1e-4
    if milp_budget > 30.0:
        gap = 5e-5
    if milp_budget > 42.0:
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
                n_workers=min(8, max(1, os.cpu_count() or 4)),
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

def mine_columns_from_incumbent(best_bins, ilookup, vehicle_cycle, pool, t_end, rng):
    """
    Intensification around incumbent:
    destroy a subset of bins, repack removed items, run short LNS,
    and inject resulting bins into the column pool.
    """
    if not best_bins:
        return
    base = [b.copy() for b in best_bins]
    if len(base) < 2:
        return

    while time.monotonic() < t_end:
        work = [b.copy() for b in base]
        k = min(len(work) - 1, rng.randint(2, 5))
        if k <= 0:
            break

        removed = []
        for idx in sorted(rng.sample(range(len(work)), k), reverse=True):
            for rec in work[idx].items:
                removed.append(ilookup[rec[0]])
            del work[idx]

        vehs = vehicle_cycle[rng.randrange(len(vehicle_cycle))]
        ok = True
        for it in sorted(removed, key=lambda x: -x['vol']):
            if time.monotonic() > t_end:
                return
            placed = False
            for b in sorted(work, key=lambda bb: bb.rem_vol()):
                if b.try_add(it):
                    placed = True
                    break
            if not placed:
                nb = open_bin(it, vehs)
                if nb:
                    work.append(nb)
                    placed = True
            if not placed:
                ok = False
                break

        if not ok:
            continue

        local_end = min(time.monotonic() + 6.0, t_end)
        work = lns(work, ilookup, vehs, local_end, rng, verbose=False, pool=pool)
        pool.add_solution(work)


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
    """
    3-D Bin Packing: EP + GRASP + LNS + Column Generation + Set Partition MILP.
    """

    """CHANGE WITH:
        
        SOLVE_SECONDS  = 545
        DETERMINISTIC  = True
        N_THREADS      = 1
        VERBOSE        = False
        BASE_SEED      = 15
        HIGHS_PARALLEL = 'off'
        for deterministic behavior (same solution every run, useful for debugging and local testing).
        
        or with 
        SOLVE_SECONDS  = int(os.getenv('SOLVER_364130_TIME_LIMIT', '590'))
        DETERMINISTIC  = os.getenv('SOLVER_364130_DETERMINISTIC', '0') != '0'
        N_THREADS      = 1 if DETERMINISTIC else min(4, max(1, (os.cpu_count() or 4)))
        VERBOSE        = os.getenv('SOLVER_364130_VERBOSE', '1') != '0'
        BASE_SEED      = int(os.getenv('SOLVER_364130_SEED', '15'))
        HIGHS_PARALLEL = os.getenv(
            'SOLVER_364130_HIGHS_PARALLEL',
            'off' if DETERMINISTIC else 'on',
        )
    
    for non-deterministic behavior (potentially better solutions, useful for final submission)."""

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
        log(f"  Seed     :  {self.BASE_SEED}  (deterministic={self.DETERMINISTIC})")
        log(f"  Threads  :  {self.N_THREADS}  (HiGHS parallel={self.HIGHS_PARALLEL})")
        log(f"{'═'*65}")

        items         = parse_items(self.inst.df_items)
        vehicles_all  = parse_vehicles(self.inst.df_vehicles)
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

        def _worker(label, seq, bf, alpha, seed, vehs):
            rng = random.Random(seed)
            bins, unp = grasp(seq, vehs, alpha, rng, t_p1, bf, honor_order=False)
            if unp:
                return label, None, float('inf')
            bins = lns(bins, ilookup, vehs, t_p1, rng, verbose=False, pool=pool)
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
        ]
        for j, mono in enumerate(mono_vehicle_lists):
            configs.append((f"mono-{mono[0]['type']}", by_vol, True, 0.06, seed0 + 20 + j, mono))

        with ThreadPoolExecutor(max_workers=self.N_THREADS) as ex:
            fs = {
                ex.submit(_worker, label, seq, bf, alpha, seed, vehs): label
                for (label, seq, bf, alpha, seed, vehs) in configs
            }
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
                    bins, unp = grasp(seq, vehs, alpha, rng, local_end, bf, honor_order=False)
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
                bins, unp = grasp(seq, vehs, alpha, rng, t_p2, bf, honor_order=False)
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
        t_cg   = min(time.monotonic() + 22.0, tend - 50.0)
        for i, item in enumerate(sorted(items, key=lambda x: -x['vol'])):
            if time.monotonic() > t_cg:
                break
            vehs = vehicle_cycle[i % len(vehicle_cycle)]
            for b in generate_columns_for_item(item, items, vehs, 6, rng_cg, t_cg):
                pool.add_bin(b)

        cols = pool.get_columns()
        log(f"  Column pool size: {len(cols)}")

        milp_budget = min(34.0, tend - time.monotonic() - 24.0)
        milp_cols   = min(2200, max(1000, len(all_ids) // 2 + 500))
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
            milp_cols2 = min(2400, milp_cols + 400)
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
                    cand, unp = grasp(seq, vehs, alpha, rng_local, burst_end, bf, honor_order=False)
                    if not unp:
                        cand = lns(cand, ilookup, vehs, burst_end, rng_local,
                                   verbose=False, pool=pool)
                        pool.add_solution(cand)
                        update_best(cand, f'lateR-{seed % 1000}')
                seed += 97
        else:
            while time.monotonic() < tend - 2.0:
                rng_local = random.Random(seed)
                vehs = vehicle_cycle[seed % len(vehicle_cycle)]
                burst_end = min(time.monotonic() + 4.5, tend - 2.0)
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
                    cand, unp = grasp(seq, vehs, alpha, rng_local, burst_end, bf, honor_order=False)
                    if not unp:
                        cand = lns(cand, ilookup, vehs, burst_end, rng_local,
                                   verbose=False, pool=pool)
                        pool.add_solution(cand)
                        update_best(cand, f'lateR-{seed % 1000}')
                seed += 97

        # Deterministic incumbent polishing (cannot worsen objective).
        rem = tend - time.monotonic()
        if rem > 6.0:
            with lock:
                snap = [b.copy() for b in best_bins[0]]
            polish_end = min(tend - 3.0, time.monotonic() + min(12.0, rem - 3.0))
            polished = post_optimize_bins(snap, ilookup, v_orders['cost'], polish_end, pool=pool)
            update_best(polished, 'post-opt')

        # Last tiny exact re-optimization over full column pool.
        rem = tend - time.monotonic()
        if rem > 2.5:
            cols3 = pool.get_columns()
            if cols3:
                budget3 = min(6.0, rem - 0.8)
                cols_cap3 = min(2600, max(1200, len(all_ids) // 2 + 700))
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
