from datetime import datetime
import json
from pathlib import Path

from numpy import array
from numpy.typing import NDArray
from tqdm import tqdm
import matplotlib.pyplot as plt

from simsailplane import (
    ActionControllerLazy,
    Sailplane6DOF,
    SolViewer,
    ActionViewer,
    ActionSolViewer,
    SimArguments,
    PlaneArguments,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

# Unified argument parser combines both simulation and aircraft parameters
INTERACTIVE = 1

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


def plot(sols, control_trajs, output_dir: Path, interactive: bool = False):

    var_names = ["pN", "pE", "pD", "u", "v", "w", "p", "q", "r", "q0", "q1", "q2", "q3"]
    control_names = ["delta_a", "delta_e", "delta_r", "airbrake"]

    # Create combined viewer
    combined_viewer = ActionSolViewer(
        sols, control_trajs, var_names=var_names, control_names=control_names
    )

    # Original phase plots
    viewer = SolViewer(sols, var_names=var_names)

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

    fig.savefig(output_dir / "Figure1_PositionSpeed.png")

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

    fig.savefig(output_dir / "Figure2_PositionSpeed.png")
    if interactive == 1:
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

    fig.savefig(output_dir / "Figure3_PositionSpeedTime.png")
    if interactive > 0:
        fig.show()

    # New combined plots
    fig = combined_viewer.plot_state_and_controls(
        state_vars=["pN", "pE"],
        control_vars=["delta_a", "delta_e"],
        title="State and Control Evolution",
    )
    fig.savefig(output_dir / "Figure4_StateAndControls.png")
    if interactive > 1:
        fig.show()

    fig = combined_viewer.plot_combined_evolution(
        title="Combined State and Control Evolution"
    )
    fig.savefig(output_dir / "Figure5_CombinedEvolution.png")
    if interactive > 1:
        fig.show()

    fig = combined_viewer.plot_control_effectiveness(
        title="Control Effectiveness Analysis"
    )
    fig.savefig(output_dir / "Figure6_ControlEffectiveness.png")
    if interactive > 1:
        fig.show()

    # Control-only plots
    action_viewer = ActionViewer(control_trajs, control_names=control_names)
    fig = action_viewer.plot_control_evolution(title="Control Surface Evolution")
    fig.savefig(output_dir / "Figure7_ControlEvolution.png")
    if interactive > 1:
        fig.show()

    # Dual-axis plot: all state variables vs all control variables
    fig = combined_viewer.plot_dual_axis(
        title="All State and Control Variables vs Time",
        show_all_trajectories=False,  # Show only first trajectory for clarity
    )
    fig.savefig(output_dir / "Figure8_DualAxis.png")
    if interactive > 1:
        fig.show()


def main():

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

    output_dir = OUTPUT_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=False, exist_ok=True)
    args_json = {k: v for k, v in args.__dict__.items()}

    with open(output_dir / "args.json", "w") as f:
        json.dump(args_json, f)

    plot(sols, ctrl_trajs, output_dir, interactive=INTERACTIVE)


if __name__ == "__main__":
    main()
