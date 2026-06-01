#!/usr/bin/env python3
"""
Convenience runner: solve a list of datasets back-to-back with solver_364130_354977_356856_359530,
without touching the graded main.py. Each dataset uses the full time budget
(env SOLVER_364130_TIME_LIMIT, default 600s) and writes results/sol_*.csv.

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

    import os, sys, time
    HERE = os.path.dirname(os.path.abspath(__file__))
    PROJ = os.path.dirname(HERE)
    sys.path.insert(0, PROJ)
    os.chdir(PROJ)

    from instances import Instance
    from solver_364130_354977_356856_359530 import solver_364130_354977_356856_359530

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
        t0 = time.monotonic()
        inst = Instance(ds)
        solver = solver_364130_354977_356856_359530(inst)
        print(f"\n>>> {ds}: items={len(inst.df_items)} vehicles={len(inst.df_vehicles)}")
        solver.solve()
        print(f">>> {ds} done in {time.monotonic() - t0:.0f}s")
    print("\nAll done. Verify with:  python tools/eval_solutions.py "
          + ' '.join(args))
