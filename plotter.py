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


    def plot_against_time(self, vars_to_plot=None, title=None, figsize=(8,6)):
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

        fig, ax = plt.subplots(figsize=figsize)
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
        return fig

    def phase_plot(self, x_var, y_var, z_var=None, *, time_color=False, figsize=(8,6), title=None):
        """
        Plot a phase diagram of two or three variables, optionally derived via a function.

        Args:
            x_var: variable index/name or a tuple (vars_tuple, func) to compute x-values
            y_var: variable index/name or a tuple (vars_tuple, func) to compute y-values
            z_var: optional variable/index/name or tuple (vars_tuple, func) for 3D plot
            time_color: if True, color points by time along the trajectory
            figsize: figure size
            title: optional plot title
        """

        def eval_var(var, y):
            """Evaluate variable, supporting index/name or tuple (vars, func)."""
            if isinstance(var, tuple):
                vars_list, func = var
                indices = [self.var_names.index(v) if isinstance(v, str) else v for v in vars_list]
                return func(*[y[:, idx] for idx in indices])
            else:
                idx = self.var_names.index(var) if isinstance(var, str) else var
                return y[:, idx]

        # Determine 2D vs 3D
        is_3d = z_var is not None

        if not is_3d:
            fig, ax = plt.subplots(figsize=figsize)
            if time_color:
                for sol in self.sols:
                    y = sol.y.T
                    t = sol.t
                    x_vals = eval_var(x_var, y)
                    y_vals = eval_var(y_var, y)
                    sc = ax.scatter(x_vals, y_vals, c=t, cmap='viridis', s=10)
                cbar = plt.colorbar(sc)
                cbar.set_label("Time [s]")
            else:
                for sol, label in zip(self.sols, self.labels):
                    y = sol.y.T
                    x_vals = eval_var(x_var, y)
                    y_vals = eval_var(y_var, y)
                    ax.plot(x_vals, y_vals, label=label)
            ax.set_xlabel(str(x_var))
            ax.set_ylabel(str(y_var))
        else:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            if time_color:
                for sol in self.sols:
                    y = sol.y.T
                    t = sol.t
                    x_vals = eval_var(x_var, y)
                    y_vals = eval_var(y_var, y)
                    z_vals = eval_var(z_var, y)
                    sc = ax.scatter(x_vals, y_vals, z_vals, c=t, cmap='viridis', s=10)
                cbar = plt.colorbar(sc)
                cbar.set_label("Time [s]")
            else:
                for sol, label in zip(self.sols, self.labels):
                    y = sol.y.T
                    x_vals = eval_var(x_var, y)
                    y_vals = eval_var(y_var, y)
                    z_vals = eval_var(z_var, y)
                    ax.plot(x_vals, y_vals, z_vals, label=label)
            ax.set_xlabel(str(x_var))
            ax.set_ylabel(str(y_var))
            ax.set_zlabel(str(z_var))

        if title:
            ax.set_title(title)
        ax.grid(True)
        if not time_color:
            ax.legend()

        return fig
