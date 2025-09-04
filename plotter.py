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

    def phase_plot(self, x_var, y_var, z_var=None, *, color=None, figsize=(8,6), title=None):
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
                indices = [self.var_names.index(v) if isinstance(v, str) else v for v in vars_list]
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
                    sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap='viridis', s=10)
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
            ax = fig.add_subplot(111, projection='3d')
            for sol, label in zip(self.sols, self.labels):
                y = sol.y.T
                t = sol.t
                x_vals, x_label = eval_var(x_var, y, t)
                y_vals, y_label = eval_var(y_var, y, t)
                z_vals, z_label = eval_var(z_var, y, t)
                c_vals, c_label = eval_var(color, y, t)

                if c_vals is not None:
                    sc = ax.scatter(x_vals, y_vals, z_vals, c=c_vals, cmap='viridis', s=10)
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
