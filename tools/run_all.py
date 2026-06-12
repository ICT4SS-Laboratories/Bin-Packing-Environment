#!/usr/bin/env python3
"""
Convenience runner: solve a list of datasets back-to-back with solver_364130_354977_356856_359530,
without touching the graded main.py. Each dataset uses the full time budget
(env SOLVER_364130_TIME_LIMIT, default 600s) and writes results/sol_*.csv.

ALWAYS-OVERWRITE: the CSV on disk is ALWAYS the output of the latest run —
this runner never restores a previous file. After each dataset it only
PRINTS a comparison against the previous cost so regressions are visible.

SEED ROTATION: each invocation uses a different solver BASE_SEED (derived
from the current time). The solver's deep-LNS trajectory is seed-driven, so
re-running with the same seed reproduces the same local optimum; rotating
the seed is what makes repeated runs actually explore different basins.
Set SOLVER_364130_SEED explicitly to pin a seed instead.

Usage (from project root):
    python3 tools/run_all.py A-J
    python3 tools/run_all.py K-Z
    python3 tools/run_all.py D G H
    SOLVER_364130_TIME_LIMIT=120 python3 tools/run_all.py A B   # faster smoke
"""
if __name__ == '__main__':
    # macOS segfault fix (same as main.py): set spawn before heavy imports.
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    import os, sys, time, shutil
    HERE = os.path.dirname(os.path.abspath(__file__))
    PROJ = os.path.dirname(HERE)
    sys.path.insert(0, PROJ)
    os.chdir(PROJ)

    import pandas as pd
    from instances import Instance
    from solver_364130_354977_356856_359530 import solver_364130_354977_356856_359530

    SOLVER = 'solver_364130_354977_356856_359530'

    # Seed rotation (unless the user pinned one via env).
    if 'SOLVER_364130_SEED' not in os.environ:
        seed = (int(time.time()) // 7) % 100000 + 17
        solver_364130_354977_356856_359530.BASE_SEED = seed
        print(f"Seed rotation: BASE_SEED={seed} "
              f"(pin with SOLVER_364130_SEED=<n>)")

    def get_dims(it, o):
        w, d, h = it['width'], it['depth'], it['height']
        return [(w, d, h), (d, w, h), (h, d, w),
                (d, h, w), (w, h, d), (h, w, d)][o]

    def cost_if_feasible(ds):
        """Objective cost of the on-disk CSV if feasible (exact checker), else
        None. Never raises: a transient FS error here (observed: TimeoutError
        from read_csv under system load) must not kill the whole batch."""
        try:
            return _cost_if_feasible_inner(ds)
        except Exception as e:
            print(f"  (comparison skipped: {type(e).__name__}: {e})")
            return None

    def _cost_if_feasible_inner(ds):
        inst = Instance(ds)
        items, veh = inst.df_items, inst.df_vehicles
        idict, vdict = items.to_dict('index'), veh.to_dict('index')
        path = os.path.join('results', f'sol_{ds}_{SOLVER}.csv')
        if not os.path.exists(path):
            return None
        try:
            sol = pd.read_csv(path)
        except Exception:
            return None
        # New official-checker format rules (June 2026 version).
        vidxs = sorted(sol['idx_vehicle'].dropna().unique())
        if vidxs != list(range(len(vidxs))):
            return None
        total = 0.0
        placed = set()
        for vidx, g in sol.groupby('idx_vehicle'):
            if g['type_vehicle'].nunique(dropna=False) != 1:
                return None
            v = vdict[g.iloc[0]['type_vehicle']]
            total += v['cost']
            boxes = []; tw = tval = 0.0
            for _, r in g.iterrows():
                iid = r['id_item']
                if iid in placed:
                    return None
                placed.add(iid); it = idict[iid]
                o = int(r['orient'])
                if o < 0 or o > 5 or str(o) not in str(it['allowedRotations']):
                    return None
                w, d, h = get_dims(it, o)
                x, y, z = r['x_origin'], r['y_origin'], r['z_origin']
                if x < 0 or y < 0 or z < 0:
                    return None
                b = (x, y, z, x + d, y + w, z + h, w * d)
                if b[3] > v['depth'] or b[4] > v['width'] or b[5] > v['height']:
                    return None
                for o in boxes:
                    if (max(b[0], o[0]) < min(b[3], o[3]) and
                            max(b[1], o[1]) < min(b[4], o[4]) and
                            max(b[2], o[2]) < min(b[5], o[5])):
                        return None
                boxes.append(b); tw += it['weight']; tval += it['value']
            if tw > v['maxWeight']:
                return None
            if pd.notna(v['maxValue']) and tval > v['maxValue']:
                return None
            grav = v['gravityStrength']
            for i, b in enumerate(boxes):
                if b[2] == 0:
                    continue
                sup = 0.0
                for j, o in enumerate(boxes):
                    if i == j:
                        continue
                    if abs(o[5] - b[2]) < 1e-6:
                        dx = max(0, min(b[3], o[3]) - max(b[0], o[0]))
                        dy = max(0, min(b[4], o[4]) - max(b[1], o[1]))
                        sup += dx * dy
                if sup < b[6] * (grav / 100.0):
                    return None
        if set(items.index) - placed:
            return None
        return total

    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    args = sys.argv[1:] or ['K-Z']
    dsets = []
    for a in args:
        a = a.strip()
        if '-' in a and len(a) == 3:
            i, j = chars.index(a[0].upper()), chars.index(a[2].upper())
            dsets += [f'Dataset{c}' for c in chars[i:j + 1]]
        elif a.startswith('Dataset'):
            dsets.append(a)
        else:
            dsets.append(f'Dataset{a.upper()}')

    print(f"Running {len(dsets)} dataset(s): {', '.join(dsets)}")
    for ds in dsets:
        prev_cost = cost_if_feasible(ds)

        t0 = time.monotonic()
        inst = Instance(ds)
        solver = solver_364130_354977_356856_359530(inst)
        print(f"\n>>> {ds}: items={len(inst.df_items)} vehicles={len(inst.df_vehicles)}")
        solver.solve()
        print(f">>> {ds} done in {time.monotonic() - t0:.0f}s")

        # Informative comparison only — the new CSV ALWAYS stays on disk.
        new_cost = cost_if_feasible(ds)
        new_str = f"{new_cost:,.2f}" if new_cost is not None else "INFEASIBLE"
        if prev_cost is not None and new_cost is not None:
            delta = new_cost - prev_cost
            tag = ("improved" if delta < -1e-9
                   else ("worse" if delta > 1e-9 else "same"))
            print(f">>> {ds}: {new_str}  (previous {prev_cost:,.2f}, {tag})")
        else:
            print(f">>> {ds}: {new_str}")
    print("\nAll done. Verify with:  python tools/eval_solutions.py "
          + ' '.join(args))
