from numpy import pi, zeros
from tqdm import tqdm

from action import ActionControllerLazy
from simulation import Sailplane6DOF
from plotter import SolViewer

deg = pi / 180
bounds = [(-20 * deg, 20 * deg), (-15 * deg, 15 * deg), (-15 * deg, 15 * deg), (0, 1)]
max_rate = [30 * deg, 20 * deg, 20 * deg, 0.5]
dt = 0.5
num = 20

ctrl = ActionControllerLazy(bounds, max_rate, dt)

steps = 30
disc = (3, 3, 3, 3)
seed = 42

# print("Total trajectories:", ctrl.num_trajectories(steps, disc))

# # Get first trajectory
# traj0 = ctrl.trajectory_by_index(0, steps, disc)

# # Iterate lazily over first 5 trajectories
# for i, traj in zip(range(5), ctrl.enumerate_lazy(steps, disc)):
#     print(f"Trajectory {i} shape:", traj.shape)

ctrl_trajs = ctrl.sample_trajectories(steps, num, seed=42)

plane = Sailplane6DOF()

# x0 = initial 6-DOF state
# [pN, pE, pD, u, v, w, p, q, r, q0, q1, q2, q3]
x0 = zeros(13)
x0[9] = 1.0  # initial quaternion identity

sols = []


for c_traj in tqdm(ctrl_trajs):
    c_fun = ctrl.trajectory_to_ctrl_timefun(c_traj)

    t_span = (0.0, dt * (c_traj.shape[0] - 1))  # total simulation time

    sol = plane.integrate(x0, t_span, ctrl_timefun=c_fun)

    sols.append(sol)

var_names = ['pN','pE','pD','u','v','w','p','q','r','q0','q1','q2','q3']
print(len(sols))
viewer = SolViewer(sols, var_names=var_names)
fig = viewer.phase_plot('pN', 'u', title="north-east Positions")

fig.savefig("Figure.png")
