if __name__ == '__main__':
    # Fix for macOS segfault with OR-Tools and multiprocessing
    # MUST set spawn method before any imports
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Already set
    
    from instances import Instance
    from solver_354977 import solver_354977
    from solver_356856 import solver_356856
    from solver_364130 import solver_364130

    dataset_name = 'DatasetH'

    inst = Instance(dataset_name)

    # solver = solver_354977(inst)
    solver = solver_364130(inst)

    print(f"Starting solver on {dataset_name}...")
    print(f"Items: {len(inst.df_items)}, Vehicle types: {len(inst.df_vehicles)}")
    print("This will take approximately 8-10 minutes. Please wait...")
    
    solver.solve()
    
    print("Solver completed! Check results/ directory for output.")