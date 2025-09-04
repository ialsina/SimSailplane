"""
action_controller.py

Wrapper class to manage pilot actions (roll, pitch, yaw, airbrake) and generate trajectories.
Works with Sailplane6DOF or similar dynamics models.

Controls: [delta_a, delta_e, delta_r, airbrake]
Units: radians for control surfaces, 0..1 for airbrake
"""

import numpy as np
import itertools
from typing import List, Dict, Callable


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def control_dict_from_vec(v: np.ndarray) -> Dict:
    return {
        "delta_a": float(v[0]),
        "delta_e": float(v[1]),
        "delta_r": float(v[2]),
        "airbrake": float(v[3]),
    }


class ActionController:
    def __init__(
        self,
        bounds: List[tuple],
        max_rate: List[float],
        dt: float,
        initial: np.ndarray = None,
    ):
        """
        Args:
          bounds: [(a_min,a_max), (e_min,e_max), (r_min,r_max), (ab_min,ab_max)]
          max_rate: [rate_a, rate_e, rate_r, rate_ab] maximum rate (units/s)
          dt: timestep (s) for trajectory discretization
          initial: initial control vector (4,), defaults to [0,0,0,0]
        """
        self.bounds = bounds
        self.max_rate = np.array(max_rate)
        self.dt = dt
        self.step_max = self.max_rate * self.dt
        if initial is None:
            self.initial = np.zeros(4)
        else:
            self.initial = initial.copy()

    # -----------------
    # trajectory generation
    # -----------------
    def enumerate_bfs(
        self,
        steps: int,
        discretization_per_control=(3, 3, 3, 3),
        max_paths: int = 1000,
    ) -> List[np.ndarray]:
        """Enumerate all possible trajectories within limits (breadth-first)."""
        increments_per_control = []
        for i, n in enumerate(discretization_per_control):
            if n <= 1:
                increments = np.array([0.0])
            else:
                increments = np.linspace(-self.step_max[i], self.step_max[i], n)
            increments_per_control.append(increments)

        step_increments = np.array(
            list(itertools.product(*increments_per_control))
        )  # (branch,4)

        # queue of partial trajectories
        queue = [[self.initial.copy()]]
        results = []

        while queue:
            traj = queue.pop(0)
            if len(traj) - 1 == steps:
                results.append(np.vstack(traj))
                if len(results) >= max_paths:
                    break
                continue

            cur = traj[-1]
            for inc in step_increments:
                nxt = cur + inc
                for j, (lo, hi) in enumerate(self.bounds):
                    nxt[j] = clamp(nxt[j], lo, hi)
                if np.any(np.abs(nxt - cur) > self.step_max + 1e-12):
                    continue
                queue.append(traj + [nxt.copy()])

        return results

    def generate_primitives(self, steps: int) -> List[np.ndarray]:
        """Generate ramp-up, ramp-down, hold primitives for each control."""
        trajectories = []

        def control_prims(val, lo, hi, step):
            arrs = []
            # hold
            arrs.append(np.full(steps + 1, val))
            # ramp up
            up = [val]
            for _ in range(steps):
                up.append(clamp(up[-1] + step, lo, hi))
            arrs.append(np.array(up))
            # ramp down
            dn = [val]
            for _ in range(steps):
                dn.append(clamp(dn[-1] - step, lo, hi))
            arrs.append(np.array(dn))
            return arrs

        prims = []
        for i in range(4):
            lo, hi = self.bounds[i]
            prims.append(control_prims(self.initial[i], lo, hi, self.step_max[i]))

        for pa in prims[0]:
            for pe in prims[1]:
                for pr in prims[2]:
                    for pab in prims[3]:
                        trajectories.append(np.vstack([pa, pe, pr, pab]).T)

        return trajectories

    def sample_random(self, steps: int, n_samples: int = 100, seed=None) -> List[np.ndarray]:
        """Random sampling of action trajectories within bounds and derivative limits."""
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(n_samples):
            traj = np.zeros((steps + 1, 4))
            traj[0] = self.initial.copy()
            for k in range(1, steps + 1):
                inc = rng.uniform(-self.step_max, self.step_max)
                nxt = traj[k - 1] + inc
                for j, (lo, hi) in enumerate(self.bounds):
                    nxt[j] = clamp(nxt[j], lo, hi)
                traj[k] = nxt
            samples.append(traj)
        return samples

    # -----------------
    # utility for simulation
    # -----------------
    def trajectory_to_ctrl_timefun(self, traj: np.ndarray, t0: float = 0.0) -> Callable:
        """Convert trajectory (N x 4) to ctrl_timefun(t)."""
        times = t0 + np.arange(traj.shape[0]) * self.dt

        def ctrl_timefun(t):
            if t <= times[0]:
                idx = 0
            elif t >= times[-1]:
                idx = len(times) - 1
            else:
                idx = int((t - times[0]) // self.dt)
                idx = clamp(idx, 0, len(times) - 1)
            return control_dict_from_vec(traj[idx])

        return ctrl_timefun

    def run_on_model(
        self,
        model,
        x0,
        steps: int,
        trajectories: List[np.ndarray],
        wind_timefun=lambda t: np.zeros(3),
        max_runs=20,
    ):
        """Run trajectories on a sailplane model."""
        results = []
        for i, traj in enumerate(trajectories):
            if i >= max_runs:
                break
            T = self.dt * (steps)
            ctrl_fun = self.trajectory_to_ctrl_timefun(traj)
            sol = model.integrate(x0, (0, T), ctrl_timefun=ctrl_fun, wind_timefun=wind_timefun)
            results.append((traj, sol))
        return results


if __name__ == "__main__":
    deg = np.pi / 180.0
    bounds = [(-20 * deg, 20 * deg), (-15 * deg, 15 * deg), (-15 * deg, 15 * deg), (0, 1)]
    max_rate = [30 * deg, 20 * deg, 20 * deg, 0.5]  # per second
    dt = 0.5
    ctrl = ActionController(bounds, max_rate, dt)

    steps = 5
    # Enumerate
    enum_trajs = ctrl.enumerate_bfs(steps, max_paths=50)
    print("Enumerated:", len(enum_trajs))

    # Primitives
    prim_trajs = ctrl.generate_primitives(steps)
    print("Primitives:", len(prim_trajs))

    # Random
    rnd_trajs = ctrl.sample_random(steps, n_samples=10, seed=42)
    print("Random:", len(rnd_trajs))

