"""
Why are DatasetH, G, F so far from LB?
Look at item dimensions, vehicle dimensions, weight/volume ratios,
and our current solution structure.
"""
import os
import pandas as pd
from instances import Instance


def look_at(name):
    inst = Instance(name)
    items = inst.df_items
    vehicles = inst.df_vehicles

    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")

    print(f"\n  Items ({len(items)}):")
    print(f"    width  : min={items['width'].min():.1f}  max={items['width'].max():.1f}  mean={items['width'].mean():.1f}")
    print(f"    depth  : min={items['depth'].min():.1f}  max={items['depth'].max():.1f}  mean={items['depth'].mean():.1f}")
    print(f"    height : min={items['height'].min():.1f}  max={items['height'].max():.1f}  mean={items['height'].mean():.1f}")
    print(f"    weight : min={items['weight'].min():.1f}  max={items['weight'].max():.1f}  mean={items['weight'].mean():.1f}  total={items['weight'].sum():.1f}")
    print(f"    value  : min={items['value'].min():.1f}  max={items['value'].max():.1f}  mean={items['value'].mean():.1f}  total={items['value'].sum():.1f}")
    vol = items['width'] * items['depth'] * items['height']
    print(f"    volume : total={vol.sum():.0f}  mean={vol.mean():.0f}")

    print(f"\n  Vehicles ({len(vehicles)}):")
    for vt, v in vehicles.iterrows():
        vol_v = v['width'] * v['depth'] * v['height']
        mv = v.get('maxValue', 'inf')
        if pd.notna(mv) and mv > 1e15:
            mv = 'inf'
        print(f"    {vt:14s}  {v['width']:.0f}x{v['depth']:.0f}x{v['height']:.0f} (vol={vol_v:.0f})  "
              f"maxW={v['maxWeight']:.0f}  maxV={mv}  cost={v['cost']:.2f}  grav={v['gravityStrength']}%")

    # Compare totals to vehicle capacities
    total_vol = vol.sum()
    total_wt = items['weight'].sum()
    total_val = items['value'].sum()
    print(f"\n  Aggregates:")
    print(f"    sum(item vol)    = {total_vol:.0f}")
    print(f"    sum(item weight) = {total_wt:.1f}")
    print(f"    sum(item value)  = {total_val:.1f}")

    max_vol = (vehicles['width'] * vehicles['depth'] * vehicles['height']).max()
    max_wt = vehicles['maxWeight'].max()
    print(f"    max veh vol      = {max_vol:.0f}  -> need >= {total_vol/max_vol:.1f} bins (vol)")
    print(f"    max veh maxWt    = {max_wt:.1f}  -> need >= {total_wt/max_wt:.1f} bins (weight)")

    # Read current solution
    path = os.path.join('results', f'sol_{name}_solver_364130.csv')
    if os.path.exists(path):
        sol = pd.read_csv(path)
        bins_used = sol.groupby('idx_vehicle').agg(
            type_vehicle=('type_vehicle', 'first'),
            n_items=('id_item', 'count'),
        )
        n_bins = len(bins_used)
        # Compute cost
        cost_map = vehicles['cost'].to_dict()
        total_cost = bins_used['type_vehicle'].map(cost_map).sum()
        print(f"\n  Our solution:")
        print(f"    bins used       = {n_bins}")
        print(f"    total cost      = {total_cost:.2f}")
        print(f"    by type:")
        type_count = bins_used['type_vehicle'].value_counts()
        for vt, cnt in type_count.items():
            cost = cost_map.get(vt, 0) * cnt
            print(f"      {vt:14s} x {cnt:4d} = {cost:.2f}")

        # How many items per bin on average
        items_per_bin = bins_used['n_items'].describe()
        print(f"    items/bin       : min={items_per_bin['min']:.0f}  mean={items_per_bin['mean']:.1f}  "
              f"max={items_per_bin['max']:.0f}")

    print()


if __name__ == '__main__':
    for ds in ['DatasetF', 'DatasetG', 'DatasetH']:
        look_at(ds)
