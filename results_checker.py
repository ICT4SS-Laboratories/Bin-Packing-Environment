import os
import pandas as pd
from instances import Instance

if __name__ == '__main__':
    dataset_name = 'DatasetJ'
    solver_name = 'solver_364130_354977_356856_359530'
    inst = Instance(dataset_name)

    # -------------------------
    # Load data
    # -------------------------
    items = inst.df_items
    vehicles = inst.df_vehicles
    solution = pd.read_csv(os.path.join('results', f'sol_{dataset_name}_{solver_name}.csv'))

    # -------------------------
    # Helpers
    # -------------------------
    def get_dims(item, orient):
        w, d, h = item["width"], item["depth"], item["height"]

        rotations = [
            (w, d, h),
            (d, w, h),
            (h, d, w),
            (d, h, w),
            (w, h, d),
            (h, w, d),
        ]
        return rotations[orient]


    def overlap_1d(a_min, a_max, b_min, b_max):
        return max(a_min, b_min) < min(a_max, b_max)


    def boxes_overlap(a, b):
        return (
            overlap_1d(a["x1"], a["x2"], b["x1"], b["x2"]) and
            overlap_1d(a["y1"], a["y2"], b["y1"], b["y2"]) and
            overlap_1d(a["z1"], a["z2"], b["z1"], b["z2"])
        )


    # -------------------------
    # Prepare lookups
    # -------------------------
    items_dict = items.to_dict("index")
    vehicles_dict = vehicles.to_dict("index")

    groups = solution.groupby("idx_vehicle")

    total_cost = 0
    feasible = True
    placed_item_ids = set()

    print("\n================ SOLUTION CHECK ================\n")

    for vidx, group in groups:
        vehicle_type = group.iloc[0]['type_vehicle']
        vehicle = vehicles_dict[vehicle_type]

        vehicle_ok = True
        placed_boxes = []
        total_weight = 0
        total_value = 0

        total_cost += vehicle["cost"]

        for _, row in group.iterrows():
            item_id = row["id_item"]
            placed_item_ids.add(item_id)
            item = items_dict[item_id]

            orient = int(row["orient"])
            w, d, h = get_dims(item, orient)

            x, y, z = row["x_origin"], row["y_origin"], row["z_origin"]

            box = {
                "id": item_id,
                "x1": x, "y1": y, "z1": z,
                "x2": x + d, "y2": y + w, "z2": z + h,
                "base_area": w * d
            }

            # -------------------------
            # Bounds
            # -------------------------
            if box["x2"] > vehicle["depth"] or \
            box["y2"] > vehicle["width"] or \
            box["z2"] > vehicle["height"]:
                print(f"\nVehicle {vidx} ({vehicle_type}):")
                print(f"OUT OF BOUNDS: {item_id}")
                vehicle_ok = False

            # -------------------------
            # Overlap
            # -------------------------
            for other in placed_boxes:
                if boxes_overlap(box, other):
                    print(f"\nVehicle {vidx} ({vehicle_type})")
                    print(f"OVERLAP: {item_id} with {other['id']}")
                    vehicle_ok = False

            placed_boxes.append(box)

            total_weight += item["weight"]
            total_value += item["value"]

        # -------------------------
        # Weight
        # -------------------------
        if total_weight > vehicle["maxWeight"]:
            print(f"\nVehicle {vidx} ({vehicle_type})")
            print(f" WEIGHT: {total_weight:.2f} > {vehicle['maxWeight']}")
            vehicle_ok = False

        # -------------------------
        # Value
        # -------------------------
        if total_value > vehicle["maxValue"]:
            print(f"\nVehicle {vidx} ({vehicle_type})")
            print(f"VALUE: {total_value:.2f} > {vehicle['maxValue']}")
            vehicle_ok = False

        # -------------------------
        # Gravity
        # -------------------------
        gravity = vehicle["gravityStrength"]

        for i, box in enumerate(placed_boxes):
            if box["z1"] == 0:
                continue

            support_area = 0

            for j, other in enumerate(placed_boxes):
                if i == j:
                    continue

                if abs(other["z2"] - box["z1"]) < 1e-6:
                    dx = max(0, min(box["x2"], other["x2"]) - max(box["x1"], other["x1"]))
                    dy = max(0, min(box["y2"], other["y2"]) - max(box["y1"], other["y1"]))
                    support_area += dx * dy

            required = box["base_area"] * (gravity / 100.0)

            if support_area < required:
                print(f"\nVehicle {vidx} ({vehicle_type})")
                print(f"GRAVITY: {box['id']} supported {support_area:.1f} < {required:.1f}")
                vehicle_ok = False

        if not vehicle_ok:
            feasible = False
    
    # -------------------------
    # Check that all items are placed
    # -------------------------
    all_items = set(items.index)
    missing_items = all_items - placed_item_ids

    if missing_items:
        print(f"MISSING ITEMS: {', '.join(sorted(missing_items))}")
        feasible = False

    # -------------------------
    # Final summary
    # -------------------------
    print("=============== SUMMARY ===============")
    print(f"Total cost: {total_cost:.2f}")

    if feasible:
        print("✅ FEASIBLE solution")
    else:
        print("❌ INFEASIBLE solution")