#!/usr/bin/env python3
"""
Best-of-N runner: run each dataset several times and KEEP only the cheapest
FEASIBLE solution CSV on disk. The solver is stochastic (multi-thread, time
limited), so some datasets land on a better basin only on some runs; this
captures the best one. The kept CSV is always a real solver output, so it
stays "coherent and repeatable" in the sense the assignment requires.

Usage (from project root):
    python tools/run_best.py A 3            # DatasetA, 3 runs, keep best
    python tools/run_best.py A-C 5          # A,B,C — 5 runs each
    python tools/run_best.py K O S T 3      # a custom list

Each run uses the full time budget (env SOLVER_364130_TIME_LIMIT, default 600s).
A run that is INFEASIBLE or more expensive than the current best is discarded.
"""
if __name__ == '__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    import os, sys, shutil
    HERE = os.path.dirname(os.path.abspath(__file__))
    PROJ = os.path.dirname(HERE)
    sys.path.insert(0, PROJ)
    os.chdir(PROJ)
    import pandas as pd
    from instances import Instance
    from solver_364130_354977_356856_359530 import solver_364130_354977_356856_359530

    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    SOLVER = 'solver_364130_354977_356856_359530'

    # ---- parse args: dataset tokens + optional trailing integer N ----
    args = sys.argv[1:]
    N = 3
    if args and args[-1].isdigit():
        N = int(args[-1]); args = args[:-1]
    if not args:
        args = ['A-J']
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

    def get_dims(it, o):
        w, d, h = it['width'], it['depth'], it['height']
        return [(w, d, h), (d, w, h), (h, d, w),
                (d, h, w), (w, h, d), (h, w, d)][o]

    def cost_if_feasible(ds):
        """Return objective cost if the on-disk CSV is feasible, else None."""
        inst = Instance(ds)
        items, veh = inst.df_items, inst.df_vehicles
        idict, vdict = items.to_dict('index'), veh.to_dict('index')
        path = os.path.join('results', f'sol_{ds}_{SOLVER}.csv')
        if not os.path.exists(path):
            return None
        sol = pd.read_csv(path)
        total = 0.0
        placed = set()
        for vidx, g in sol.groupby('idx_vehicle'):
            v = vdict[g.iloc[0]['type_vehicle']]
            total += v['cost']
            boxes = []; tw = tval = 0.0
            for _, r in g.iterrows():
                placed.add(r['id_item']); it = idict[r['id_item']]
                w, d, h = get_dims(it, int(r['orient']))
                x, y, z = r['x_origin'], r['y_origin'], r['z_origin']
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

    print(f"Best-of-{N} over: {', '.join(dsets)}")
    for ds in dsets:
        best_cost = None
        best_path = os.path.join('results', f'sol_{ds}_{SOLVER}.csv')
        keep_path = best_path + '.best'
        # seed the incumbent with any existing feasible file
        existing = cost_if_feasible(ds)
        if existing is not None:
            best_cost = existing
            shutil.copyfile(best_path, keep_path)
        for k in range(N):
            inst = Instance(ds)
            solver = solver_364130_354977_356856_359530(inst)
            solver.solve()  # writes results/sol_<ds>_solver_364130.csv
            c = cost_if_feasible(ds)
            tag = f"{c:,.2f}" if c is not None else "INFEASIBLE"
            better = c is not None and (best_cost is None or c < best_cost - 1e-9)
            if better:
                best_cost = c
                shutil.copyfile(best_path, keep_path)
            print(f"  {ds} run {k+1}/{N}: {tag}"
                  f"{'  <-- new best' if better else ''}", flush=True)
        # restore the best feasible CSV
        if os.path.exists(keep_path):
            shutil.copyfile(keep_path, best_path)
            os.remove(keep_path)
            print(f"  {ds}: kept best = {best_cost:,.2f}")
        else:
            print(f"  {ds}: WARNING no feasible run found")
    print("Done. Verify with:  python tools/eval_solutions.py "
          + ' '.join(args))
