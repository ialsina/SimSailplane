"""
action_controller.py

Wrapper class to manage pilot actions (roll, pitch, yaw, airbrake) and generate trajectories.
Works with Sailplane6DOF or similar dynamics models.

Controls: [delta_a, delta_e, delta_r, airbrake]
Units: radians for control surfaces, 0..1 for airbrake
"""

import numpy as np
import itertools
from typing import List, Dict, Callable, Iterator


import numpy as np
import itertools
from typing import List, Dict, Callable, Iterator


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def control_dict_from_vec(v: np.ndarray) -> Dict:
    return {
        "delta_a": float(v[0]),
        "delta_e": float(v[1]),
        "delta_r": float(v[2]),
        "airbrake": float(v[3]),
    }


# ------------------------
# Base class (shared logic)
# ------------------------
class BaseActionController:
    def __init__(self, bounds, max_rate, dt, initial=None):
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
            self.initial = np.array(initial, dtype=float)

    # ---- increment utilities ----
    def step_increments(self, discretization_per_control):
        """Precompute possible increments for each timestep."""
        increments_per_control = []
        for i, n in enumerate(discretization_per_control):
            if n <= 1:
                increments = np.array([0.0])
            else:
                increments = np.linspace(-self.step_max[i], self.step_max[i], n)
            increments_per_control.append(increments)
        return list(itertools.product(*increments_per_control))

    # ---- trajectory count ----
    def num_trajectories(self, steps: int, discretization_per_control) -> int:
        branch = np.prod(discretization_per_control)
        return branch**steps

    # ---- trajectory → time function ----
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


# ------------------------
# Eager controller
# ------------------------
class ActionController(BaseActionController):
    def enumerate_bfs(
        self,
        steps: int,
        discretization_per_control=(3, 3, 3, 3),
        max_paths: int = 1000,
    ) -> List[np.ndarray]:
        """Eagerly build all trajectories (up to max_paths)."""
        step_increments = self.step_increments(discretization_per_control)

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
                nxt = cur + np.array(inc)
                for j, (lo, hi) in enumerate(self.bounds):
                    nxt[j] = np.clip(nxt[j], lo, hi)
                if np.any(np.abs(nxt - cur) > self.step_max + 1e-12):
                    continue
                queue.append(traj + [nxt.copy()])

        return results

    def generate_primitives(self, steps: int) -> List[np.ndarray]:
        """Generate hold, ramp-up, ramp-down primitives."""
        def control_prims(val, lo, hi, step):
            arrs = []
            arrs.append(np.full(steps + 1, val))  # hold
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

        trajectories = []
        for pa in prims[0]:
            for pe in prims[1]:
                for pr in prims[2]:
                    for pab in prims[3]:
                        trajectories.append(np.vstack([pa, pe, pr, pab]).T)

        return trajectories


# ------------------------
# Lazy controller
# ------------------------
class ActionControllerLazy(BaseActionController):
    def enumerate_lazy(
        self,
        steps: int,
        discretization_per_control=(3, 3, 3, 3),
    ) -> Iterator[np.ndarray]:
        """Yield trajectories lazily (one at a time)."""
        step_increments = self.step_increments(discretization_per_control)

        def recurse(traj, depth):
            if depth == steps:
                yield np.vstack(traj)
                return
            cur = traj[-1]
            for inc in step_increments:
                nxt = cur + np.array(inc)
                for j, (lo, hi) in enumerate(self.bounds):
                    nxt[j] = np.clip(nxt[j], lo, hi)
                if np.any(np.abs(nxt - cur) > self.step_max + 1e-12):
                    continue
                yield from recurse(traj + [nxt.copy()], depth + 1)

        yield from recurse([self.initial.copy()], 0)

    def trajectory_by_index(
        self,
        index: int,
        steps: int,
        discretization_per_control=(3, 3, 3, 3),
    ) -> np.ndarray:
        """Compute nth trajectory directly (random access)."""
        step_increments = self.step_increments(discretization_per_control)
        branch = len(step_increments)

        digits = []
        for _ in range(steps):
            digits.append(index % branch)
            index //= branch
        if index > 0:
            raise IndexError("Index exceeds total number of trajectories")

        traj = [self.initial.copy()]
        cur = self.initial.copy()
        for d in digits:
            inc = np.array(step_increments[d])
            nxt = cur + inc
            for j, (lo, hi) in enumerate(self.bounds):
                nxt[j] = np.clip(nxt[j], lo, hi)
            traj.append(nxt.copy())
            cur = nxt
        return np.vstack(traj)
