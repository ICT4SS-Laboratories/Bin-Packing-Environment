"""
Tighter lower bound: per-item compatibility-aware LP.

The naive LB uses min(cost/capacity) across ALL vehicles, ignoring that
heavier items can't fit in smaller vehicles. We improve by:

1. Group items by their *minimum compatible vehicle class* (smallest
   vehicle they can fit in, weight+dim+value-wise).
2. For each group, the LB cost is (group_total_weight) *
   min(cost/maxWeight across vehicles compatible with the group's
   weight ceiling).
3. Sum LBs per group.

Also produces a more rigorous LP that respects compatibility.
"""
import math
import pandas as pd
from instances import Instance
from scipy.optimize import linprog


def get_compatible_vehicles(item_row, vehicles_df):
    """List of vehicle types that can host this item (weight, value, fits)."""
    w, d, h = item_row['width'], item_row['depth'], item_row['height']
    rotations = str(item_row['allowedRotations'])
    rot_dims = {
        '0': (w, d, h), '1': (d, w, h), '2': (h, d, w),
        '3': (d, h, w), '4': (w, h, d), '5': (h, w, d),
    }
    compat = []
    for vt, v in vehicles_df.iterrows():
        if item_row['weight'] > v['maxWeight'] + 1e-9:
            continue
        mv = v.get('maxValue')
        if pd.notna(mv) and mv > 0 and mv < 1e15:
            if item_row['value'] > mv + 1e-9:
                continue
        # Dim fit
        ok = False
        for r in rotations:
            if r in rot_dims:
                iw, id_, ih = rot_dims[r]
                if iw <= v['width'] + 1e-9 and id_ <= v['depth'] + 1e-9 and ih <= v['height'] + 1e-9:
                    ok = True
                    break
        if ok:
            compat.append(vt)
    return compat


def tight_lb(name):
    inst = Instance(name)
    items = inst.df_items
    vehicles = inst.df_vehicles
    veh = vehicles.copy()
    veh['vol'] = veh['width'] * veh['depth'] * veh['height']
    veh['cpw'] = veh['cost'] / veh['maxWeight']
    veh['cpv'] = veh['cost'] / veh['vol']

    # Per-item: min cpw across compatible vehicles
    n = len(items)
    item_compat = []
    item_min_cpw = []
    item_min_cpv = []
    for _, item in items.iterrows():
        compat = get_compatible_vehicles(item, vehicles)
        if not compat:
            item_compat.append([])
            item_min_cpw.append(float('inf'))
            item_min_cpv.append(float('inf'))
            continue
        sub = veh.loc[compat]
        item_compat.append(compat)
        item_min_cpw.append(sub['cpw'].min())
        item_min_cpv.append(sub['cpv'].min())

    # LB: sum over items of weight * min compatible cpw
    LB_compat_weight = sum(it_w * cpw for it_w, cpw in
                            zip(items['weight'].values, item_min_cpw)
                            if cpw < float('inf'))
    # Volume version
    item_vols = (items['width'] * items['depth'] * items['height']).values
    LB_compat_vol = sum(v * cpv for v, cpv in zip(item_vols, item_min_cpv)
                        if cpv < float('inf'))

    # Also: per-item min cost / capacity GIVEN the item must use a single bin
    # Each item needs at least cost(min_compat_v) / max_items_in_min_compat_v.
    # Approx max_items by weight: floor(maxW / item_w) — but we don't know per-bin.
    # Instead: for each item, min over compatible v of (cost_v / max_items_in_v_by_weight).
    # max_items_in_v_by_weight = maxW_v / item_w  (continuous)
    item_lb = []
    for it_w, compat in zip(items['weight'].values, item_compat):
        if not compat:
            item_lb.append(0)
            continue
        sub = veh.loc[compat]
        # cost per item if packing at full weight
        cpis = []
        for _, v in sub.iterrows():
            if it_w > 0:
                items_per_v = v['maxWeight'] / it_w
            else:
                items_per_v = 1e9
            cpis.append(v['cost'] / max(items_per_v, 1.0))
        item_lb.append(min(cpis))
    LB_per_item_cost_share = sum(item_lb)

    # Compare with naive (using all vehicles)
    LB_naive_w = items['weight'].sum() * veh['cpw'].min()
    LB_naive_v = item_vols.sum() * veh['cpv'].min()

    return {
        'n_items': n,
        'LB_naive_weight': LB_naive_w,
        'LB_naive_volume': LB_naive_v,
        'LB_compat_weight': LB_compat_weight,
        'LB_compat_volume': LB_compat_vol,
        'LB_per_item_share': LB_per_item_cost_share,
        'LB_tight_master': max(LB_naive_w, LB_naive_v, LB_compat_weight,
                                LB_compat_vol, LB_per_item_cost_share),
    }


def get_solution_cost(name, solver, vehicles):
    import os
    path = os.path.join('results', f'sol_{name}_{solver}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    vt = df.groupby('idx_vehicle')['type_vehicle'].first()
    return float(vt.map(vehicles['cost'].to_dict()).sum())


if __name__ == '__main__':
    datasets = ['DatasetA', 'DatasetB', 'DatasetC', 'DatasetD', 'DatasetE',
                'DatasetF', 'DatasetG', 'DatasetH', 'DatasetI', 'DatasetJ']

    print(f"\n{'='*120}")
    print(f"{'Dataset':<10} {'#Items':>7} "
          f"{'LB_naiveW':>11} {'LB_compatW':>11} {'LB_share':>11} "
          f"{'LB_tight':>11} {'Ours':>11} {'gap_tight%':>11}")
    print(f"{'='*120}")

    sum_lb = sum_ours = 0
    for ds in datasets:
        try:
            inst = Instance(ds)
            r = tight_lb(ds)
            ours = get_solution_cost(ds, 'solver_364130', inst.df_vehicles)
        except Exception as e:
            print(f"{ds:<10} skipped ({e})")
            continue
        gap = (ours - r['LB_tight_master']) / r['LB_tight_master'] * 100 if ours else 0
        if ours:
            sum_lb += r['LB_tight_master']
            sum_ours += ours
        print(f"{ds:<10} {r['n_items']:>7} "
              f"{r['LB_naive_weight']:>11.0f} {r['LB_compat_weight']:>11.0f} "
              f"{r['LB_per_item_share']:>11.0f} {r['LB_tight_master']:>11.0f} "
              f"{(ours or 0):>11.0f} {gap:>10.1f}%")
    print(f"{'='*120}")
    if sum_lb > 0:
        print(f"{'TOTAL':<10} {'':>7} {'':>11} {'':>11} {'':>11} "
              f"{sum_lb:>11.0f} {sum_ours:>11.0f} "
              f"{(sum_ours - sum_lb) / sum_lb * 100:>10.1f}%")
