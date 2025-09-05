import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


class SolViewer:
    """
    Viewer for one or more solve_ivp solutions.

    Allows plotting 2D or 3D representations of selected state variables over time.
    """

    def __init__(self, sols, var_names=None, labels=None):
        """
        Args:
            sols: a single scipy.integrate.OdeResult or a list of them
            var_names: list of names of state variables (optional, for labeling)
            labels: list of labels for each solution (optional)
        """
        if not isinstance(sols, list):
            sols = [sols]
        self.sols = sols
        self.labels = labels or [f"sol_{i}" for i in range(len(sols))]

        # Extract state shape info
        self.N_state = sols[0].y.shape[0]
        self.var_names = var_names or [f"x{i}" for i in range(self.N_state)]

    def plot_against_time(
        self, vars_to_plot=None, *, title=None, figsize=(8, 6), ax=None
    ):
        """
        Plot selected variables against time.

        Args:
            vars_to_plot: list of variable indices or names
            title: plot title
            figsize: figure size
        """
        if vars_to_plot is None:
            vars_to_plot = [0]
        indices = []
        for v in vars_to_plot:
            if isinstance(v, int):
                indices.append(v)
            elif isinstance(v, str):
                if v in self.var_names:
                    indices.append(self.var_names.index(v))
                else:
                    raise ValueError(f"Variable name {v} not found.")
            else:
                raise TypeError("vars_to_plot must be int or str")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ret_fig = True
        else:
            fig = ax.get_figure()
            ret_fig = False
        for sol, label in zip(self.sols, self.labels):
            y = sol.y.T
            t = sol.t
            for idx in indices:
                ax.plot(t, y[:, idx], label=f"{self.var_names[idx]} ({label})")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Value")
        if title:
            ax.set_title(title)
        ax.grid(True)
        ax.legend()

        if ret_fig:
            return fig
        return ax

    def phase_plot(
        self,
        x_var,
        y_var,
        z_var=None,
        *,
        color=None,
        figsize=(8, 6),
        title=None,
    ):
        """
        Plot a phase diagram of two or three variables, with optional coloring.

        Args:
            x_var: variable index/name or tuple (vars_tuple, func, optional_label)
            y_var: variable index/name or tuple (vars_tuple, func, optional_label)
            z_var: optional variable/index/name or tuple (vars_tuple, func, optional_label)
            color: None, "time", variable index/name, or tuple (vars_tuple, func, optional_label)
            figsize: figure size
            title: optional plot title
        """

        def eval_var(var, y, t=None):
            """Evaluate variable, supporting index/name, tuple (vars, func), or 'time'."""
            label = None
            if var is None:
                return None, None
            if isinstance(var, str) and var.lower() == "time":
                if t is None:
                    raise ValueError("Time array t must be provided to color by time.")
                return t, "Time [s]"
            if isinstance(var, tuple):
                if len(var) == 2:
                    vars_list, func = var
                    label = str(var)
                elif len(var) == 3:
                    vars_list, func, label = var
                indices = [
                    self.var_names.index(v) if isinstance(v, str) else v
                    for v in vars_list
                ]
                return func(*[y[:, idx] for idx in indices]), label
            else:
                idx = self.var_names.index(var) if isinstance(var, str) else var
                return y[:, idx], str(var)

        is_3d = z_var is not None

        if not is_3d:
            fig, ax = plt.subplots(figsize=figsize)
            for sol, label in zip(self.sols, self.labels):
                y = sol.y.T
                t = sol.t
                x_vals, x_label = eval_var(x_var, y, t)
                y_vals, y_label = eval_var(y_var, y, t)
                c_vals, c_label = eval_var(color, y, t)

                if c_vals is not None:
                    sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap="viridis", s=10)
                else:
                    ax.plot(x_vals, y_vals, label=label)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if title:
                ax.set_title(title)
            ax.grid(True)
            if color is None:
                ax.legend()
            if c_vals is not None:
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label(c_label)

        else:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            for sol, label in zip(self.sols, self.labels):
                y = sol.y.T
                t = sol.t
                x_vals, x_label = eval_var(x_var, y, t)
                y_vals, y_label = eval_var(y_var, y, t)
                z_vals, z_label = eval_var(z_var, y, t)
                c_vals, c_label = eval_var(color, y, t)

                if c_vals is not None:
                    sc = ax.scatter(
                        x_vals, y_vals, z_vals, c=c_vals, cmap="viridis", s=10
                    )
                else:
                    ax.plot(x_vals, y_vals, z_vals, label=label)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            if title:
                ax.set_title(title)
            ax.grid(True)
            if color is None:
                ax.legend()
            if c_vals is not None:
                cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
                cbar.set_label(c_label)

        return fig


class ActionViewer:
    """
    Viewer for control trajectories.

    Allows plotting control surface deflections and airbrake deployment over time.
    """

    def __init__(self, control_trajs, control_names=None, labels=None):
        """
        Args:
            control_trajs: a single control trajectory (N x 4) or a list of them
            control_names: list of names of control variables (optional)
            labels: list of labels for each trajectory (optional)
        """
        if not isinstance(control_trajs, list):
            control_trajs = [control_trajs]
        self.control_trajs = control_trajs
        self.labels = labels or [f"traj_{i}" for i in range(len(control_trajs))]

        # Control variable names
        self.control_names = control_names or [
            "delta_a",
            "delta_e",
            "delta_r",
            "airbrake",
        ]

        # Extract time info from first trajectory
        self.N_steps = control_trajs[0].shape[0]
        self.dt = 0.5  # Default timestep, should be passed from simulation

    def set_timestep(self, dt):
        """Set the timestep for time axis generation."""
        self.dt = dt

    def get_time_axis(self):
        """Get time axis for plotting."""
        return np.arange(self.N_steps) * self.dt

    def plot_against_time(
        self, controls_to_plot=None, *, title=None, figsize=(12, 8), ax=None
    ):
        """
        Plot selected control variables against time.

        Args:
            controls_to_plot: list of control indices or names
            title: plot title
            figsize: figure size
            ax: matplotlib axis (optional)
        """
        if controls_to_plot is None:
            controls_to_plot = list(range(len(self.control_names)))

        indices = []
        for c in controls_to_plot:
            if isinstance(c, int):
                indices.append(c)
            elif isinstance(c, str):
                if c in self.control_names:
                    indices.append(self.control_names.index(c))
                else:
                    raise ValueError(f"Control name {c} not found.")
            else:
                raise TypeError("controls_to_plot must be int or str")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        t = self.get_time_axis()

        for traj, label in zip(self.control_trajs, self.labels):
            for idx in indices:
                ax.plot(
                    t,
                    traj[:, idx],
                    label=f"{self.control_names[idx]} ({label})",
                    linewidth=2,
                )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Control Value")
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        return fig

    def plot_control_evolution(self, *, title="Control Evolution", figsize=(12, 10)):
        """
        Plot all control variables in subplots.

        Args:
            title: plot title
            figsize: figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        t = self.get_time_axis()

        for i, (control_name, ax) in enumerate(zip(self.control_names, axes)):
            for traj, label in zip(self.control_trajs, self.labels):
                ax.plot(t, traj[:, i], label=label, linewidth=2)

            ax.set_xlabel("Time [s]")
            ax.set_ylabel(f"{control_name}")
            ax.set_title(f"{control_name.replace('_', ' ').title()}")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.suptitle(title)
        plt.tight_layout()

        return fig


class ActionSolViewer:
    """
    Combined viewer for both state variables and control trajectories.

    Allows plotting state variables and controls together in synchronized plots.
    """

    def __init__(
        self, sols, control_trajs, var_names=None, control_names=None, labels=None
    ):
        """
        Args:
            sols: a single scipy.integrate.OdeResult or a list of them
            control_trajs: a single control trajectory (N x 4) or a list of them
            var_names: list of names of state variables (optional)
            control_names: list of names of control variables (optional)
            labels: list of labels for each solution (optional)
        """
        self.sol_viewer = SolViewer(sols, var_names, labels)
        self.action_viewer = ActionViewer(control_trajs, control_names, labels)

        # Synchronize timestep
        if hasattr(sols[0], "t") and len(sols[0].t) > 1:
            dt = sols[0].t[1] - sols[0].t[0]
            self.action_viewer.set_timestep(dt)

    def plot_state_and_controls(
        self,
        state_vars=None,
        control_vars=None,
        *,
        title="State and Control Evolution",
        figsize=(15, 10),
    ):
        """
        Plot state variables and controls in a 2x2 subplot layout.

        Args:
            state_vars: list of state variables to plot
            control_vars: list of control variables to plot
            title: plot title
            figsize: figure size
        """
        if state_vars is None:
            state_vars = ["pN", "pE", "pD"]  # Position
        if control_vars is None:
            control_vars = ["delta_a", "delta_e", "delta_r", "airbrake"]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot state variables
        for i, var in enumerate(state_vars[:2]):  # Plot first 2 state vars
            ax = axes[0, i]
            self.sol_viewer.plot_against_time([var], ax=ax, title=f"{var} vs Time")

        # Plot control variables
        for i, var in enumerate(control_vars[:2]):  # Plot first 2 control vars
            ax = axes[1, i]
            self.action_viewer.plot_against_time([var], ax=ax, title=f"{var} vs Time")

        plt.suptitle(title)
        plt.tight_layout()

        return fig

    def plot_combined_evolution(
        self, *, title="Combined State and Control Evolution", figsize=(16, 12)
    ):
        """
        Plot all state and control variables in a comprehensive layout.

        Args:
            title: plot title
            figsize: figure size
        """
        fig = plt.figure(figsize=figsize)

        # Create a grid layout: 3 rows, 2 columns
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

        # Position plots
        ax_pos = fig.add_subplot(gs[0, :])
        ax_vel = fig.add_subplot(gs[1, 0])
        ax_att = fig.add_subplot(gs[1, 1])
        ax_ctrl = fig.add_subplot(gs[2, :])

        # Plot position (3D trajectory)
        # Note: phase_plot doesn't support ax parameter, so we'll skip this for now
        # self.sol_viewer.phase_plot("pN", "pE", "pD", ax=ax_pos, title="3D Position Trajectory")

        # Plot velocity
        self.sol_viewer.plot_against_time(
            ["u", "v", "w"], ax=ax_vel, title="Velocity Components"
        )

        # Plot attitude (quaternion)
        self.sol_viewer.plot_against_time(
            ["q0", "q1", "q2", "q3"], ax=ax_att, title="Quaternion Attitude"
        )

        # Plot controls
        self.action_viewer.plot_against_time(
            ax=ax_ctrl, title="Control Surface Deflections"
        )

        plt.suptitle(title, fontsize=16)

        return fig

    def plot_control_effectiveness(
        self,
        state_vars=None,
        control_vars=None,
        *,
        title="Control Effectiveness Analysis",
        figsize=(14, 10),
    ):
        """
        Plot how controls affect state variables over time.

        Args:
            state_vars: list of state variables to analyze
            control_vars: list of control variables to analyze
            title: plot title
            figsize: figure size
        """
        if state_vars is None:
            state_vars = ["pN", "pE", "pD", "u", "v", "w"]
        if control_vars is None:
            control_vars = ["delta_a", "delta_e", "delta_r", "airbrake"]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        # Plot each state variable
        for i, var in enumerate(state_vars[:6]):  # Plot up to 6 state vars
            ax = axes[i]
            self.sol_viewer.plot_against_time([var], ax=ax, title=f"{var} Evolution")

        # Add control overlay to each subplot
        t_controls = self.action_viewer.get_time_axis()
        for i, var in enumerate(state_vars[:6]):
            ax = axes[i]
            ax2 = ax.twinx()

            # Plot controls on secondary y-axis
            for j, traj in enumerate(self.action_viewer.control_trajs):
                if j == 0:  # Only plot first trajectory to avoid clutter
                    for k, ctrl_var in enumerate(control_vars):
                        ax2.plot(
                            t_controls,
                            traj[:, k],
                            "--",
                            alpha=0.7,
                            label=f"{ctrl_var}",
                            linewidth=1,
                        )

            ax2.set_ylabel("Control Value", fontsize=8)
            ax2.tick_params(axis="y", labelsize=6)
            if i == 0:  # Only show legend on first subplot
                ax2.legend(fontsize=6, loc="upper right")

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        return fig

    def plot_dual_axis(
        self,
        *,
        title="State and Control Variables vs Time",
        figsize=(14, 10),
        show_all_trajectories=True,
    ):
        """
        Plot all state variables on top axis and all control variables on bottom axis.

        Args:
            title: plot title
            figsize: figure size
            show_all_trajectories: if False, only show the first trajectory to reduce clutter
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot state variables on top axis
        sols_to_plot = (
            self.sol_viewer.sols if show_all_trajectories else [self.sol_viewer.sols[0]]
        )
        labels_to_plot = (
            self.sol_viewer.labels
            if show_all_trajectories
            else [self.sol_viewer.labels[0]]
        )

        for sol, label in zip(sols_to_plot, labels_to_plot):
            y = sol.y.T
            t = sol.t
            for i, var_name in enumerate(self.sol_viewer.var_names):
                ax1.plot(
                    t, y[:, i], label=f"{var_name} ({label})", linewidth=1.5, alpha=0.8
                )

        ax1.set_ylabel("State Variables", fontsize=12)
        ax1.set_title("State Variables vs Time", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        # Plot control variables on bottom axis
        t_controls = self.action_viewer.get_time_axis()
        trajs_to_plot = (
            self.action_viewer.control_trajs
            if show_all_trajectories
            else [self.action_viewer.control_trajs[0]]
        )
        ctrl_labels_to_plot = (
            self.action_viewer.labels
            if show_all_trajectories
            else [self.action_viewer.labels[0]]
        )

        for traj, label in zip(trajs_to_plot, ctrl_labels_to_plot):
            for i, ctrl_name in enumerate(self.action_viewer.control_names):
                ax2.plot(
                    t_controls,
                    traj[:, i],
                    label=f"{ctrl_name} ({label})",
                    linewidth=1.5,
                    alpha=0.8,
                    linestyle="--",
                )

        ax2.set_xlabel("Time [s]", fontsize=12)
        ax2.set_ylabel("Control Variables", fontsize=12)
        ax2.set_title("Control Variables vs Time", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        return fig
