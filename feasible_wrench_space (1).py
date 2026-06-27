import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
import time

# ==========================================
# 1. parameter configuration
# ==========================================
L = 1.0             # moment arm
max_thrust = 5.0    # max_thrust
sigma = 0.015       # drag moment coefficient
mass = 1.0          # drone mass
gravity = 9.81      # gravity
num_tilt_divs = 12  # number of discrete tilt directions for each rotor

def rot_z(gamma_rad):
    c = np.cos(gamma_rad)
    s = np.sin(gamma_rad)
    return np.array([
        [ c, -s,  0],
        [ s,  c,  0],
        [ 0,  0,  1]
    ])

# rotor position from CoM
positions = np.array([
    [ L,  0,  0],
    [-L,  0,  0],
    [ 0,  L,  0],
    [ 0, -L,  0]
])

# rotor orientations from CoM
orientations = [
    rot_z(0),
    rot_z(np.pi),
    rot_z(np.pi/2),
    rot_z(-np.pi/2)
]

rotor_directions = [-1, -1, 1, 1]


# ==========================================
# 2. V expressions of rotor
# ==========================================
def skew(p):
    return np.array([
        [ 0,    -p[2],  p[1]],
        [ p[2],  0,    -p[0]],
        [-p[1],  p[0],  0   ]
    ])

def get_rotor_wrenches(position, orientation, rotor_direction, max_thrust, sigma, num_tilt_divs=8):
    thrust_wrench_units = []

    # thrust wrench units in rotor frame
    for i in range(num_tilt_divs):
        phi = 2 * np.pi * i / num_tilt_divs
        axis = np.array([0, -np.sin(phi), np.cos(phi)]) # +x is 1DoF gimbal axis
        thrust_wrench_unit = np.concatenate([axis, sigma * rotor_direction * axis])
        thrust_wrench_units.append(thrust_wrench_unit)

    # Adjoint matrix
    Ad = np.block([[orientation, np.zeros((3, 3))], [skew(position) @ orientation, orientation]])

    # wrench in CoM frame
    wrenches = []
    for thrust_wrench_unit in thrust_wrench_units:
        wrench = max_thrust * Ad @ thrust_wrench_unit
        wrenches.append(wrench)

    return np.array(wrenches)


# ==========================================
# 3. calculate minkowski sum of V expression
# ==========================================
print("calculate minkowski sum")
wrenches_all_rotors = []
for p, R, d in zip(positions, orientations, rotor_directions):
    wrenches_all_rotors.append(get_rotor_wrenches(p, R, d, max_thrust, sigma, num_tilt_divs=num_tilt_divs))

all_combinations = itertools.product(*wrenches_all_rotors)
W_total = np.sum(list(all_combinations), axis=1)

print(f"total wrench space vertices: {len(W_total)} points")


# ==========================================
# 4. Shift all vertices by gravity wrench
# ==========================================
gravity_wrench = np.array([0, 0, mass * gravity, 0, 0, 0])
W_offset = W_total - gravity_wrench


# ==========================================
# 5. Convex hull and halfspace intersection to extract vertices of the projection in moment space
# ==========================================
print("calculating 6D convex hull")
start = time.time()
hull_6d = ConvexHull(W_offset)
end = time.time()
print(f"6D convex hull calculated in {end - start:.2f} seconds")

A_6d = hull_6d.equations[:, :-1]
b_6d = hull_6d.equations[:, -1]

# moment
A_3d = A_6d[:, 3:6]

# 3D halfspace representation (A_3d, b_6d)
halfspaces = np.hstack((A_3d, b_6d[:, np.newaxis]))

# calculate an interior point for halfspace intersection
interior_point = np.array([0.0, 0.0, 0.0])
print(f"assume {interior_point} is an interior point for halfspace intersection. Please check if it is valid.")
try:
    hs = HalfspaceIntersection(halfspaces, interior_point)
    M_vertices = hs.intersections
    print(f"extracted vertices in 3D intersections: {len(M_vertices)} points")
except Exception as e:
    print("do not have a valid interior point for halfspace intersection. Please check the configuration.")
    raise e

hull_3d = ConvexHull(M_vertices)


# ==========================================
# 6. Visualization of the projection in moment space
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# surface
ax.plot_trisurf(hull_3d.points[:, 0], hull_3d.points[:, 1], hull_3d.points[:, 2],
                triangles=hull_3d.simplices, alpha=0.5, color='lime', edgecolor='k', linewidth=0.5)

# vertices
ax.scatter(hull_3d.points[:, 0], hull_3d.points[:, 1], hull_3d.points[:, 2], s=10)

ax.set_title("Feasible Torque Space")
ax.set_xlabel("Torque Mx [Nm]")
ax.set_ylabel("Torque My [Nm]")
ax.set_zlabel("Torque Mz [Nm]")

# set scale
max_range = np.array([hull_3d.points[:,0].max()-hull_3d.points[:,0].min(),
                      hull_3d.points[:,1].max()-hull_3d.points[:,1].min(),
                      hull_3d.points[:,2].max()-hull_3d.points[:,2].min()]).max() / 2.0
mid_x = (hull_3d.points[:,0].max()+hull_3d.points[:,0].min()) * 0.5
mid_y = (hull_3d.points[:,1].max()+hull_3d.points[:,1].min()) * 0.5
mid_z = (hull_3d.points[:,2].max()+hull_3d.points[:,2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()
