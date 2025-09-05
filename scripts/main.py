from pathlib import Path

from numpy import array
from numpy.typing import NDArray
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.integrate

from simsailplane import (
    ActionControllerLazy,
    Sailplane6DOF,
    SolViewer,
    SimArguments,
    PlaneArguments,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

# Unified argument parser combines both simulation and aircraft parameters
INTERACTIVE = True

sols = []


def simulate(
    ctrl: ActionControllerLazy,
    c_traj: NDArray,
    x0: NDArray,
    plane: Sailplane6DOF,
    sim_args: SimArguments,
):

    c_fun = ctrl.trajectory_to_ctrl_timefun(c_traj)

    t_span = (0.0, sim_args.dt * (c_traj.shape[0] - 1))  # total simulation time

    sol = plane.integrate(x0, t_span, ctrl_timefun=c_fun)

    return sol


def plot(sols, interactive: bool = False):

    var_names = ["pN", "pE", "pD", "u", "v", "w", "p", "q", "r", "q0", "q1", "q2", "q3"]
    viewer = SolViewer(sols, var_names=var_names)

    if interactive:
        plt.ion()

    fig = viewer.phase_plot(
        (
            ("pN", "pE", "pD"),
            lambda x, y, z: ((x - 0) ** 2 + (y - 0) ** 2 + (z - 0) ** 2) ** (1 / 2),
            "Dist to (0, 0, 0)",
        ),
        (
            ("u", "v", "w"),
            lambda x, y, z: ((x - 0) ** 2 + (y - 0) ** 2 + (z - 0) ** 2) ** (1 / 2),
            "Speed",
        ),
        title="Position Speed",
    )

    fig.savefig(OUTPUT_DIR / "Figure1.png")

    fig = viewer.phase_plot(
        "pN",
        "pE",
        "pD",
        color=(
            ("u", "v", "w"),
            lambda x, y, z: ((x - 0) ** 2 + (y - 0) ** 2 + (z - 0) ** 2) ** (1 / 2),
            "Speed",
        ),
        title="Position Speed",
    )

    fig.savefig(OUTPUT_DIR / "Figure2.png")
    if interactive:
        fig.show()

    fig = viewer.phase_plot(
        "pN",
        (
            ("u", "v", "w"),
            lambda x, y, z: ((x - 0) ** 2 + (y - 0) ** 2 + (z - 0) ** 2) ** (1 / 2),
            "Speed",
        ),
        "pD",
        color="time",
        title="Position Speed Time",
    )

    fig.savefig(OUTPUT_DIR / "Figure3.png")
    if interactive:
        fig.show()


def main():

    global sols

    # Create a unified argument parser that combines both SimArguments and PlaneArguments
    sim_parser = SimArguments.get_argument_parser()
    parser = PlaneArguments.get_argument_parser(sim_parser)

    # Parse all arguments at once
    args = parser.parse_args()

    # Create the argument objects
    sim_args = SimArguments.from_arguments(args)
    plane_args = PlaneArguments.from_arguments(args)

    # Create the sailplane with custom parameters
    plane_params = plane_args.to_plane_params()
    plane = Sailplane6DOF(**plane_params)

    # Create action controller
    ctrl = ActionControllerLazy(
        sim_args.bounds, sim_args.max_rate, sim_args.dt, sim_args.initial
    )

    # Generate control trajectories
    ctrl_trajs = ctrl.sample_trajectories(
        sim_args.steps, sim_args.num, seed=sim_args.seed
    )

    # x0 = initial 6-DOF state
    # [pN, pE, pD, u, v, w, p, q, r, q0, q1, q2, q3]
    x0 = array(sim_args.initial_state)

    sols = []

    for c_traj in tqdm(ctrl_trajs):
        sol = simulate(ctrl, c_traj, x0, plane, sim_args)
        sols.append(sol)

    plot(sols, interactive=INTERACTIVE)


if __name__ == "__main__":
    main()
