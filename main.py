from numpy import pi
from action import ActionControllerLazy

deg = pi / 180
bounds = [(-20 * deg, 20 * deg), (-15 * deg, 15 *deg), (-15*deg,15*deg), (0,1)]
max_rate = [30*deg, 20*deg, 20*deg, 0.5]
dt = 0.5

ctrl = ActionControllerLazy(bounds, max_rate, dt)

steps = 3
disc = (3,3,3,3)

print("Total trajectories:", ctrl.num_trajectories(steps, disc))

# Get first trajectory
traj0 = ctrl.trajectory_by_index(0, steps, disc)

# Iterate lazily over first 5 trajectories
for i, traj in zip(range(5), ctrl.enumerate_lazy(steps, disc)):
    print(f"Trajectory {i} shape:", traj.shape)