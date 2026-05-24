"""
Lower-bound analysis for 3D BPP with multiple vehicle types.

Computes several valid lower bounds for each dataset and compares
against the current solver result + competitor result (when known).

Lower bounds:
  LB_volume  = total_item_volume * min(cost/volume across vehicles)
               -> tightest possible volume "rental price"
  LB_weight  = total_item_weight * min(cost/max_weight across vehicles)
  LB_value   = total_item_value  * min(cost/max_value  across vehicles)
  LB_n_bins  = ceil(total_volume / max_vehicle_volume) * cheapest_cost
               -> at least this many bins are needed; combined with cheapest cost
  LB_LP      = LP relaxation of an aggregated multi-vehicle "knapsack" model
               (continuous # of vehicles of each type, must cover totals)

The LB the solver must beat is max(all LBs).
"""
import os
import math
import pandas as pd

from instances import Instance


def parse_dataset(name):
    inst = Instance(name)
    items = inst.df_items
    vehicles = inst.df_vehicles
    return items, vehicles


def item_fits_in_vehicle(item_row, v_row):
    """Check if item can fit (in any orientation) in vehicle."""
    w, d, h = item_row['width'], item_row['depth'], item_row['height']
    W, D, H = v_row['width'], v_row['depth'], v_row['height']
    rotations = str(item_row['allowedRotations'])
    rot_dims = {
        '0': (w, d, h),
        '1': (d, w, h),
        '2': (h, d, w),
        '3': (d, h, w),
        '4': (w, h, d),
        '5': (h, w, d),
    }
    for r in rotations:
        if r in rot_dims:
            iw, id_, ih = rot_dims[r]
            if iw <= W + 1e-9 and id_ <= D + 1e-9 and ih <= H + 1e-9:
                return True
    return False


def compute_bounds(items, vehicles):
    total_vol = (items['width'] * items['depth'] * items['height']).sum()
    total_wt = items['weight'].sum()
    total_val = items['value'].sum()

    # vehicle metrics
    veh = vehicles.copy()
    veh['vol'] = veh['width'] * veh['depth'] * veh['height']
    veh['cost_per_vol'] = veh['cost'] / veh['vol']
    veh['cost_per_wt'] = veh['cost'] / veh['maxWeight']
    # maxValue may be NaN/None for "infinite"
    has_val_cap = veh['maxValue'].notna() & (veh['maxValue'] > 0) & (veh['maxValue'] < 1e15)
    veh.loc[has_val_cap, 'cost_per_val'] = veh.loc[has_val_cap, 'cost'] / veh.loc[has_val_cap, 'maxValue']
    veh.loc[~has_val_cap, 'cost_per_val'] = float('nan')

    # Per-item: find min-cost vehicle that can host it alone (item must fit dimensionally + weight + value)
    per_item_min_cost = []
    for _, item in items.iterrows():
        cheapest = float('inf')
        for _, v in veh.iterrows():
            if item['weight'] > v['maxWeight'] + 1e-9:
                continue
            if pd.notna(v['maxValue']) and v['maxValue'] > 0 and v['maxValue'] < 1e15:
                if item['value'] > v['maxValue'] + 1e-9:
                    continue
            if not item_fits_in_vehicle(item, v):
                continue
            if v['cost'] < cheapest:
                cheapest = v['cost']
        per_item_min_cost.append(cheapest)

    # LB_volume (continuous): total volume packed at minimum cost-per-volume rate
    LB_vol = total_vol * veh['cost_per_vol'].min()
    LB_wt = total_wt * veh['cost_per_wt'].min()
    if veh['cost_per_val'].notna().any():
        LB_val = total_val * veh['cost_per_val'].dropna().min()
    else:
        LB_val = 0.0

    # LB_n_bins: at least this many vehicles needed by volume
    n_bins_vol = math.ceil(total_vol / veh['vol'].max())
    # at least this many by weight
    n_bins_wt = math.ceil(total_wt / veh['maxWeight'].max())
    # If we need at least n_bins, and each costs at least cheapest_cost...
    cheapest_cost = veh['cost'].min()
    LB_n_bins = max(n_bins_vol, n_bins_wt) * cheapest_cost

    # LB_per_item: each item needs to be in a vehicle. The min-cost-per-item gives a partial LB
    # but it's NOT a valid LB by itself. However, max over items of min-cost-per-item IS a LB
    # (you must use at least one bin that costs at least that much for the most expensive item).
    LB_max_item = max(c for c in per_item_min_cost if c < float('inf'))

    # LP relaxation of multi-vehicle aggregated model:
    #   minimize  sum_t  cost_t * n_t
    #   s.t. sum_t  vol_t * n_t  >= total_vol
    #        sum_t  wt_t  * n_t  >= total_wt
    #        sum_t  val_t * n_t  >= total_val (if val capped)
    #        n_t >= 0  (continuous LP relaxation)
    # This is a small LP — we can solve it.
    try:
        from scipy.optimize import linprog
        n_v = len(veh)
        c = veh['cost'].values
        A_ub = []
        b_ub = []
        # -vol * n <= -total_vol
        A_ub.append((-veh['vol'].values).tolist())
        b_ub.append(-total_vol)
        # -maxWeight * n <= -total_wt
        A_ub.append((-veh['maxWeight'].values).tolist())
        b_ub.append(-total_wt)
        # value (if applicable)
        if veh['cost_per_val'].notna().any():
            mv = veh['maxValue'].values.astype(float).copy()
            mv[~has_val_cap.values] = 1e18  # infinite cap
            A_ub.append((-mv).tolist())
            b_ub.append(-total_val)
        bounds = [(0, None)] * n_v
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        LB_LP = res.fun if res.success else 0.0
    except Exception:
        LB_LP = 0.0

    LB_master = max(LB_vol, LB_wt, LB_val, LB_n_bins, LB_max_item, LB_LP)

    return {
        'n_items': len(items),
        'n_vehicles': len(vehicles),
        'total_vol': total_vol,
        'total_wt': total_wt,
        'total_val': total_val,
        'LB_volume': LB_vol,
        'LB_weight': LB_wt,
        'LB_value': LB_val,
        'LB_n_bins': LB_n_bins,
        'LB_max_item': LB_max_item,
        'LB_LP': LB_LP,
        'LB_master': LB_master,
    }


def get_solution_cost(dataset_name, solver_name, vehicles):
    """Read solution file and compute total cost."""
    path = os.path.join('results', f'sol_{dataset_name}_{solver_name}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # group by idx_vehicle, get one row per vehicle
    vt = df.groupby('idx_vehicle')['type_vehicle'].first()
    vehicle_costs = vehicles['cost'].to_dict()
    return float(vt.map(vehicle_costs).sum())


if __name__ == '__main__':
    datasets = ['DatasetA', 'DatasetB', 'DatasetC', 'DatasetD', 'DatasetE',
                'DatasetF', 'DatasetG', 'DatasetH', 'DatasetI', 'DatasetJ']
    solver = 'solver_364130'
    competitor = 'solver_354977'

    print(f"\n{'='*120}")
    print(f"{'Dataset':<10} {'#Items':>7} {'LB_vol':>11} {'LB_wt':>11} "
          f"{'LB_LP':>11} {'LB*':>11} {'Ours':>11} {'Comp':>11} "
          f"{'gap_LB%':>8} {'gap_C%':>8}")
    print(f"{'='*120}")

    rows = []
    sum_LB = sum_ours = sum_comp = 0.0
    for ds in datasets:
        try:
            items, vehicles = parse_dataset(ds)
        except Exception as e:
            print(f"{ds:<10} skipped ({e})")
            continue
        bounds = compute_bounds(items, vehicles)
        ours = get_solution_cost(ds, solver, vehicles)
        comp = get_solution_cost(ds, competitor, vehicles)

        LB = bounds['LB_master']
        gap_LB = (ours - LB) / LB * 100 if (ours and LB > 0) else float('nan')
        gap_C = (ours - comp) / comp * 100 if (ours and comp) else float('nan')

        if ours: sum_ours += ours
        if comp: sum_comp += comp
        if LB:   sum_LB   += LB

        print(f"{ds:<10} {bounds['n_items']:>7} "
              f"{bounds['LB_volume']:>11.0f} {bounds['LB_weight']:>11.0f} "
              f"{bounds['LB_LP']:>11.0f} {LB:>11.0f} "
              f"{(ours or 0):>11.0f} {(comp or 0):>11.0f} "
              f"{gap_LB:>8.1f} {gap_C:>8.1f}")

        rows.append({
            'dataset': ds, 'n_items': bounds['n_items'],
            'LB_volume': bounds['LB_volume'],
            'LB_weight': bounds['LB_weight'],
            'LB_value': bounds['LB_value'],
            'LB_n_bins': bounds['LB_n_bins'],
            'LB_max_item': bounds['LB_max_item'],
            'LB_LP': bounds['LB_LP'],
            'LB_master': LB,
            'ours': ours, 'comp': comp,
            'gap_LB_pct': gap_LB, 'gap_comp_pct': gap_C,
        })

    print(f"{'='*120}")
    if sum_LB > 0:
        print(f"{'TOTAL':<10} {'':>7} {'':>11} {'':>11} {'':>11} "
              f"{sum_LB:>11.0f} {sum_ours:>11.0f} {sum_comp:>11.0f} "
              f"{(sum_ours - sum_LB) / sum_LB * 100:>8.1f} "
              f"{(sum_ours - sum_comp) / sum_comp * 100:>8.1f}")
    print(f"{'='*120}\n")

    # Save detailed CSV
    pd.DataFrame(rows).to_csv('lower_bounds_analysis.csv', index=False)
    print("→ Detailed results saved to lower_bounds_analysis.csv")
