#!/usr/bin/env python3
"""
Batch evaluator for solver_364130_354977_356856_359530 solutions.

For each requested dataset it:
  - re-runs the EXACT feasibility logic of the official results_checker.py
    (overlap / bounds / weight / value / gravity / all-items-placed),
  - reports the objective cost,
  - reports a VALID continuous lower bound (LB) and the gap to it.

Usage (from anywhere):
    python3 tools/eval_solutions.py            # default A..J
    python3 tools/eval_solutions.py A-J
    python3 tools/eval_solutions.py K-Z
    python3 tools/eval_solutions.py 0-9

LB is a lower bound on the OPTIMAL cost (any feasible cost >= LB). It is loose
(it assumes perfect resource fill), so a positive gap is normal; use it to
compare datasets and to track progress, not as 'distance to optimum'.
"""
import os, sys, math

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import pandas as pd
from instances import Instance

SOLVER = 'solver_364130_354977_356856_359530'


def expand(args):
    if not args:
        return [f'Dataset{c}' for c in 'ABCDEFGHIJ']
    out = []
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for a in args:
        a = a.strip()
        if '-' in a and len(a) == 3:           # range like A-J or 0-9
            lo, hi = a[0].upper(), a[2].upper()
            i, j = chars.index(lo), chars.index(hi)
            out += [f'Dataset{c}' for c in chars[i:j + 1]]
        elif a.startswith('Dataset'):
            out.append(a)
        else:
            out.append(f'Dataset{a.upper()}')
    return out


def get_dims(item, orient):
    w, d, h = item["width"], item["depth"], item["height"]
    return [(w, d, h), (d, w, h), (h, d, w),
            (d, h, w), (w, h, d), (h, w, d)][orient]


def overlap_1d(a1, a2, b1, b2):
    return max(a1, b1) < min(a2, b2)


def boxes_overlap(a, b):
    return (overlap_1d(a["x1"], a["x2"], b["x1"], b["x2"]) and
            overlap_1d(a["y1"], a["y2"], b["y1"], b["y2"]) and
            overlap_1d(a["z1"], a["z2"], b["z1"], b["z2"]))


def lower_bound(items, veh):
    items = items.copy(); veh = veh.copy()
    items['vol'] = items['width'] * items['depth'] * items['height']
    veh['vol'] = veh['width'] * veh['depth'] * veh['height']
    tw, tvol, tval = items['weight'].sum(), items['vol'].sum(), items['value'].sum()
    lb_w = tw * (veh['cost'] / veh['maxWeight']).min()
    lb_vol = tvol * (veh['cost'] / veh['vol']).min()
    mv = veh['maxValue']
    if mv.notna().all() and (mv < 1e14).all() and tval > 0:
        lb_val = tval * (veh['cost'] / veh['maxValue']).min()
    else:
        lb_val = 0.0
    return max(lb_w, lb_vol, lb_val)


def check(ds):
    inst = Instance(ds)
    items, veh = inst.df_items, inst.df_vehicles
    path = os.path.join('results', f'sol_{ds}_{SOLVER}.csv')
    if not os.path.exists(path):
        return None
    sol = pd.read_csv(path)
    idict, vdict = items.to_dict("index"), veh.to_dict("index")
    total_cost = 0.0
    feasible = True
    placed = set()
    reasons = []
    # ── New official-checker format rules (June 2026 version) ──
    req = {"type_vehicle", "idx_vehicle", "id_item",
           "x_origin", "y_origin", "z_origin", "orient"}
    if req - set(sol.columns):
        reasons.append("MISSING COLUMNS"); feasible = False
    vidxs = sorted(sol["idx_vehicle"].dropna().unique())
    if vidxs != list(range(len(vidxs))):
        reasons.append("NON-CONSECUTIVE IDX"); feasible = False
    for vidx, g in sol.groupby("idx_vehicle"):
        vt = g.iloc[0]['type_vehicle']
        if g["type_vehicle"].nunique(dropna=False) != 1:
            reasons.append(f"MIXED TYPES v{vidx}"); feasible = False
            continue
        if vt not in vdict:
            reasons.append(f"UNKNOWN VEHICLE v{vidx}"); feasible = False
            continue
        v = vdict[vt]
        total_cost += v["cost"]
        boxes = []; tw = 0.0; tval = 0.0; okv = True
        for _, row in g.iterrows():
            iid = row["id_item"]
            if iid in placed:
                reasons.append(f"DUPLICATE {iid}"); okv = False
            placed.add(iid)
            if iid not in idict:
                reasons.append(f"UNKNOWN ITEM {iid}"); okv = False
                continue
            it = idict[iid]
            orient = int(row["orient"])
            if orient < 0 or orient > 5:
                reasons.append(f"INVALID ORIENT {iid}"); okv = False
                continue
            if str(orient) not in str(it["allowedRotations"]):
                reasons.append(f"FORBIDDEN ROT {iid} o={orient}"); okv = False
            w, d, h = get_dims(it, orient)
            x, y, z = row["x_origin"], row["y_origin"], row["z_origin"]
            box = {"id": iid, "x1": x, "y1": y, "z1": z,
                   "x2": x + d, "y2": y + w, "z2": z + h, "ba": w * d}
            if box["x1"] < 0 or box["y1"] < 0 or box["z1"] < 0 or \
                    box["x2"] > v["depth"] or box["y2"] > v["width"] or box["z2"] > v["height"]:
                reasons.append(f"OOB v{vidx} {iid}"); okv = False
            for o in boxes:
                if boxes_overlap(box, o):
                    reasons.append(f"OVERLAP v{vidx} {iid}/{o['id']}"); okv = False
            boxes.append(box); tw += it["weight"]; tval += it["value"]
        if tw > v["maxWeight"]:
            reasons.append(f"WEIGHT v{vidx}"); okv = False
        if tval > v["maxValue"]:
            reasons.append(f"VALUE v{vidx}"); okv = False
        grav = v["gravityStrength"]
        for i, box in enumerate(boxes):
            if box["z1"] == 0:
                continue
            sup = 0.0
            for j, o in enumerate(boxes):
                if i == j:
                    continue
                if abs(o["z2"] - box["z1"]) < 1e-6:
                    dx = max(0, min(box["x2"], o["x2"]) - max(box["x1"], o["x1"]))
                    dy = max(0, min(box["y2"], o["y2"]) - max(box["y1"], o["y1"]))
                    sup += dx * dy
            if sup < box["ba"] * (grav / 100.0):
                reasons.append(f"GRAVITY v{vidx} {box['id']}"); okv = False
        feasible = feasible and okv
    missing = set(items.index) - placed
    if missing:
        reasons.append(f"MISSING {len(missing)}"); feasible = False
    lb = lower_bound(items, veh)
    return {"ds": ds, "bins": sol['idx_vehicle'].nunique(), "cost": total_cost,
            "feasible": feasible, "LB": lb,
            "gap%": 100 * (total_cost - lb) / lb if lb > 0 else float('nan'),
            "note": "" if feasible else "; ".join(reasons[:3])}


def main():
    dsets = expand(sys.argv[1:])
    print(f"{'Dataset':<10}{'bins':>6}{'cost':>14}{'LB':>14}{'gap%':>8}  feasible  notes")
    print("-" * 92)
    tcost = tlb = 0.0; allok = True; n = 0
    for ds in dsets:
        r = check(ds)
        if r is None:
            print(f"{ds:<10}{'(no solution file)':>40}")
            continue
        n += 1; tcost += r['cost']; tlb += r['LB']
        allok = allok and r['feasible']
        flag = "OK" if r['feasible'] else "INFEAS"
        print(f"{r['ds']:<10}{r['bins']:>6}{r['cost']:>14,.2f}{r['LB']:>14,.2f}"
              f"{r['gap%']:>8.1f}  {flag:<8} {r['note']}")
    print("-" * 92)
    g = 100 * (tcost - tlb) / tlb if tlb > 0 else float('nan')
    print(f"{'TOTAL':<10}{'':>6}{tcost:>14,.2f}{tlb:>14,.2f}{g:>8.1f}  "
          f"{'ALL OK' if allok else 'SOME INFEASIBLE'}  ({n} datasets)")


if __name__ == '__main__':
    main()
