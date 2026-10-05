This is a repository for a project of the course "Operational research" in which we had to manage a bin packing problem in 3d space, using the algorithms that we studied in the lectures.


# Problem Description

We have a set of heterogeneous containers, each of which is characterized by:

- *type*: a string that describes the type (unique identifier);
- *width*, *depth*, *height*: dimensions of the container;
- *maxWeight*: the maximum weight that a single container can support;
- *cost*: the cost of use;
- *maxValue*: maximum value that a container can contain (if not specified, it is assumed *maxValue* = +∞);
- *gravityStrength*: minimum percentage of the base surface of the object that must be supported (in contact) by underlying surfaces (other objects or the bottom of the container) for the position to be valid.

The objects instead are characterized by:

- *id*: to identify the single object;
- *width*, *depth*, *height*, *weight*: physical characteristics of the object;
- *value*: value of the object;
- *allowedRotations*: string containing the codes of the allowed rotations (e.g. "01" indicates that rotations 0 and 1 are allowed); the options are:
  - 0: No rotation;
  - 1: Rotation around the z axis of 90°;
  - 2: Rotation around the x axis of 90°;
  - 3: Rotation around the x axis of 90°, then around the z axis of 90°;
  - 4: Rotation around the y axis of 90°;
  - 5: Rotation around the z axis of 90°, then around the x axis of 90°.

The rotations are applied with respect to the axes of the reference system of the container and determine a permutation of the dimensions of the object (width, depth, height).

(**N.B.**: the x axis corresponds to the length dimension (depth), the y axis to the width, and the z axis to the height.)

## Objective

Given a set of containers and objects, determine:

- which containers to use;
- how to place the objects in the containers;

minimizing the total cost of the containers used, respecting all physical and logistical constraints.

It is possible to use an unlimited number of containers for each type.  
The total cost is given by the sum of the costs of the containers used (independently of their degree of filling).

The solution must, for each object, define:

- *type_vehicle*: type of the vehicle that contains the object;
- *idx_vehicle*: index (consecutive) of the vehicle that contains the object;
- *id_item*: id of the object;
- *x_origin*: minimum x coordinate of the object;
- *y_origin*: minimum y coordinate of the object;
- *z_origin*: minimum z coordinate of the object;
- *orient*: orientation (rotation) of the object.

The reference system has origin in the lower front left corner of the container.  
*x_origin*, *y_origin*, and *z_origin* therefore represent the coordinates of lower front left corner of the item.

## Constraints

- Each object must be assigned to exactly one container;
- Objects cannot intersect: the occupied volumes must be disjoint;
- Objects must remain within the limits of the container;
- The total weight of the objects in a container cannot exceed *maxWeight*;
- The total value cannot exceed *maxValue* (if defined);
- Rotations must respect *allowedRotations*;
- The stability constraint must respect *gravityStrength*;
- It is allowed to place objects on top of each other, provided that stability and weight constraints are respected.

# Dataset Example

## Containers

| type    | width | depth | height | maxWeight | cost | maxValue | gravityStrength |
|--------|------|------|--------|-----------|------|----------|----------------|
| PalletA | 1.2 | 0.8 | 1.65 | 1500 | 10 | 1000 | 0 |
| PalletB | 1.2 | 0.8 | 0.8  | 1500 | 10 |  | 50 |

## Objects

| id     | width | depth | height | weight | value | allowedRotations |
|--------|------|------|--------|--------|-------|------------------|
| item-1 | 1.2  | 0.8  | 0.6  | 10 | 5  | 012345 |
| item-2 | 1.2  | 0.8  | 0.6  | 10 | 20 | 0 |
| item-3 | 1.2  | 0.8  | 0.6  | 10 | 7  | 01 |
| item-4 | 0.2  | 0.5  | 0.8  | 1  | 14 | 012345 |
| item-5 | 0.26 | 0.46 | 0.81 | 10 | 1  | 014 |

# Solution Example

| type_vehicle | idx_vehicle | id_item | x_origin | y_origin | z_origin | orient |
|-------------|------------|--------|----------|----------|----------|--------|
| PalletA | 0 | item-1 | 0 | 0 | 0 | 0 |
| PalletA | 0 | item-2 | 0 | 0 | 0.8 | 0 |
| PalletB | 1 | item-3 | 0 | 0 | 0 | 0 |
| PalletB | 1 | item-4 | 0.8 | 0 | 0 | 3 |
| PalletB | 2 | item-5 | 0 | 0 | 0 | 1 |

## Rules

- **Time limit**: 10 minutes  
- **Maximum number of threads**: 4  
- **Deadline**: 23/06/2026 23:59  
- after the deadline, all codes will be executed on a group of 10 hidden instances (not available among the shared ones).  
- **Day for questions and clarifications**: 30/06/2026  

# Task

Using python as programming language and any type of open source code, develop a heuristic to solve the problem. Subsequently, it is necessary to use the heuristic on all the provided instances producing, for each one, a solution.

Finally, it is necessary to upload, in the "Elaborati" section of the Portale della Didattica, a compressed folder containing the code of the heuristic and the solutions obtained for all the provided instances.

## Technical Requirements

The project must be carried out in groups of 1-4 people.

A single member of the group must, within the deadline, upload a single compressed folder in the "Elaborati" section of the Portale della Didattica with the name:

`assignment_XXXXXX_YYYYYY_ZZZZZZ.zip`

where `XXXXXX`, `YYYYYY` and `ZZZZZZ` are the student ID numbers of the group members. For example, if a group is composed of two people with IDs 123456 and 654321, the folder must be called `assignment_123456_654321.zip`, (the order of the IDs is indifferent) while if the student with ID 999999 decides to carry out the project individually, the folder must be called `assignment_999999.zip`.

Inside the folder there must be:

- a solution file  
  `sol_DatasetN_XXXXXX_YYYYYY_ZZZZZZ.csv`  
  for each `DatasetN`;

- the solver  
  `solver_XXXXXX_YYYYYY_ZZZZZZ.py`;

- the file `__init__.py` in order to be able to import the solver in the main.

- the file `requirements.txt` containing all the dependencies necessary for the execution of the solver (the file must allow the creation of a working virtual environment through the command `pip install -r requirements.txt`);

- (optionally) additional scripts used by the solver.

**N.B.**: the solver must be executable without modifications in a standard Python environment (version 3.x), using exclusively the libraries specified in the file `requirements.txt`. Furthermore, it is allowed to use exclusively open-source libraries, freely installable via `pip` and without the need for proprietary or commercial licenses.

Inside the solver there must be a class with the same name as the file that contains it:

`class solver_XXXXXX_YYYYYY_ZZZZZZ():`

In this class it is necessary to implement the method `solve()`, which must not take anything as input (the necessary data are present in `self.inst`) and must not return anything as output, but must take care of creating the solution file `sol_DatasetN_XXXXXX_YYYYYY_ZZZZZZ.csv` (for `DatasetN`).

(Optionally, but strongly recommended, include in the compressed folder the file `abstract_solver.py` - provided in the example folder `solver_000000/` - and make your solver inherit from the class `AbstractSolver()` contained in it. This allows access to the method `write_solution_to_file()` to correctly create the solution file.)

An example of solver (`solver_000000`) is present in the project. The files that must be included in the compressed folder are all the elements contained in `solver_000000/` together with those (for now absent) contained in `results/`.

To test the correct functioning of the solver:

- in `main.py` replace line 2  
  `from solver_000000 import solver_000000`  
  with  
  `from solver_XXXXXX_YYYYYY_ZZZZZZ import solver_XXXXXX_YYYYYY_ZZZZZZ`  

- and line 10  
  `solver = solver_000000(inst)`  
  with  
  `solver = solver_XXXXXX_YYYYYY_ZZZZZZ(inst)`

- execute `main.py`

- execute `results_checker.py` to verify that the solution found by the solver is feasible and to visualize its total cost.

**N.B.**: The submitted solvers will be tested for evaluation exactly as described in the previous section, therefore with the exact same tools provided during development. For this reason, if `main.py` and `results_checker.py`, in the provided version, produce an error (due to wrong file name, wrong class name, wrong output format or for any other reason) when they are executed, the project will automatically be assigned 0 points.

# Ranking

A ranking will be created for each of the provided instances, based on the cost of the objective function of the (feasible) provided solution (lower cost = higher position in ranking). Each group will receive points equal to the position in the ranking (first position = 1 point, second position = 2 points, etc.). The overall ranking will be created by summing, for each group, the points obtained on all instances and ordering the groups by lower total score. Top 10 positions will be ranked again by adding the scores of a new set of instances. Moreover, by sampling, some heuristics will be tested to verify that the provided solutions are coherent and repeatable.

## FAQ

- Q1 ......?
