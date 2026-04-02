from instances import Instance
from solver_354977 import solver_354977

if __name__ == '__main__':

    dataset_name = 'DatasetA'

    inst = Instance(dataset_name)

    solver = solver_354977(inst)

    solver.solve()