import pandas as pd
import plotly.graph_objects as go
import numpy as np
import random
import os
from instances import Instance

if __name__ == '__main__':
    
    vehicle_to_visualize = 0

    dataset_name = 'DatasetC'
    solver_name = 'solver_364130'
    inst = Instance(dataset_name)

    # -------------------------
    # Load data
    # -------------------------
    items = inst.df_items
    vehicles = inst.df_vehicles
    solution = pd.read_csv(os.path.join('results', f'sol_{dataset_name}_{solver_name}.csv'))

    # ---------------------------
    # ROTATIONS
    # ---------------------------
    def apply_rotation(w, d, h, rot):
        if rot == 0:
            return w, d, h
        elif rot == 1:
            return d, w, h
        elif rot == 2:
            return h, d, w
        elif rot == 3:
            return d, h, w
        elif rot == 4:
            return w, h, d
        elif rot == 5:
            return h, w, d
        else:
            raise ValueError("Invalid rotation")
    # ---------------------------
    # DRAW BOX (ITEM)
    # ---------------------------
    def create_box(x, y, z, dx, dy, dz, color, name):
        vertices = np.array([
            [x, y, z],
            [x+dx, y, z],
            [x+dx, y+dy, z],
            [x, y+dy, z],
            [x, y, z+dz],
            [x+dx, y, z+dz],
            [x+dx, y+dy, z+dz],
            [x, y+dy, z+dz]
        ])

        faces = [
            [0,1,2], [0,2,3],
            [4,5,6], [4,6,7],
            [0,1,5], [0,5,4],
            [2,3,7], [2,7,6],
            [1,2,6], [1,6,5],
            [0,3,7], [0,7,4]
        ]

        i, j, k = zip(*faces)

        return go.Mesh3d(
            x=vertices[:,0],
            y=vertices[:,1],
            z=vertices[:,2],
            i=i, j=j, k=k,
            color=color,
            opacity=0.5,
            name=name
        )

    # ---------------------------
    # DRAW CONTAINER (WIREFRAME)
    # ---------------------------
    def create_container_wireframe(width, depth, height):
        # note: x=depth, y=width, z=height

        x = [0, depth, depth, 0, 0, 0, depth, depth, 0]
        y = [0, 0, width, width, 0, 0, 0, width, width]
        z = [0, 0, 0, 0, 0, height, height, height, height]

        edges = [
            (0,1),(1,2),(2,3),(3,0),  # bottom
            (4,5),(5,6),(6,7),(7,4),  # top
            (0,4),(1,5),(2,6),(3,7)   # verticals
        ]

        traces = []
        for e in edges:
            traces.append(go.Scatter3d(
                x=[x[e[0]], x[e[1]]],
                y=[y[e[0]], y[e[1]]],
                z=[z[e[0]], z[e[1]]],
                mode='lines',
                line=dict(color='black', width=3),
                showlegend=False
            ))
        return traces

    # ---------------------------
    # MAIN FUNCTION
    # ---------------------------
    def visualize_solution(df_items, df_vehicles, df_solution, idx_vehicle):
        fig = go.Figure()

        # filter only selected vehicle
        df_sol = df_solution[df_solution["idx_vehicle"] == idx_vehicle]

        if df_sol.empty:
            print(f"No items in vehicle {idx_vehicle}")
            return

        # get vehicle type
        vehicle_type = df_sol.iloc[0]["type_vehicle"]
        vehicle = df_vehicles.loc[vehicle_type]

        cont_width = vehicle["width"]
        cont_depth = vehicle["depth"]
        cont_height = vehicle["height"]

        # draw container wireframe
        for trace in create_container_wireframe(cont_width, cont_depth, cont_height):
            fig.add_trace(trace)

        colors = {}

        for _, row in df_sol.iterrows():
            item_id = row["id_item"]
            item = df_items[df_items.index == item_id].iloc[0]

            w, d, h = item["width"], item["depth"], item["height"]
            rot = int(row["orient"])

            w, d, h = apply_rotation(w, d, h, rot)

            x = row["x_origin"]
            y = row["y_origin"]
            z = row["z_origin"]

            if item_id not in colors:
                colors[item_id] = "rgb({},{},{})".format(
                    random.randint(50,200),
                    random.randint(50,200),
                    random.randint(50,200)
                )

            fig.add_trace(
                create_box(x, y, z, d, w, h, colors[item_id], item_id)
            )

        # layout
        fig.update_layout(
            scene=dict(
                xaxis_title="Depth (X)",
                yaxis_title="Width (Y)",
                zaxis_title="Height (Z)",
                aspectmode='data'
            ),
            title=f"Vehicle {idx_vehicle} Visualization",
            showlegend=True
        )

        fig.show()
    
    visualize_solution(items, vehicles, solution, vehicle_to_visualize)