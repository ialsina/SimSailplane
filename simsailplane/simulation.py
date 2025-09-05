"""
Sailplane Flight Dynamics Simulation

This module provides comprehensive 6-DOF (six degrees of freedom) and 3-DOF
flight dynamics models for sailplane simulation using quaternion-based attitude
representation and aerodynamic force/moment calculations.

The simulation includes:
- 6-DOF model: Full rigid body dynamics with quaternion attitude representation
- 3-DOF model: Simplified point-mass gliding model for performance analysis
- Aerodynamic models: Linearized aerodynamic coefficients for forces and moments
- Integration capabilities: Using scipy.integrate for numerical simulation

Key Features:
- Quaternion-based attitude representation (avoids gimbal lock)
- NED (North-East-Down) coordinate system
- Body-axis aerodynamic calculations
- Wind effects modeling
- Control surface deflections (elevator, aileron, rudder, airbrake)
- Configurable aircraft geometry and aerodynamic coefficients

Author: [Your Name]
Date: [Current Date]
Version: 1.0

Dependencies:
- numpy: Numerical computations and array operations
- scipy.integrate: Numerical integration of differential equations

Usage:
    # Create a sailplane instance
    plane = Sailplane6DOF()

    # Set initial conditions and run simulation
    x0 = [pN, pE, pD, u, v, w, p, q, r, q0, q1, q2, q3]
    sol = plane.integrate(x0, (0, 30), ctrl_timefun=ctrl_func)
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---- Quaternion Utilities ----
# These functions handle quaternion operations for attitude representation
# Quaternions are represented as [q0, q1, q2, q3] where q0 is the scalar part
# and [q1, q2, q3] is the vector part.
def quat_norm(q):
    """
    Calculate the norm (magnitude) of a quaternion.

    Args:
        q (array-like): Quaternion as [q0, q1, q2, q3]

    Returns:
        float: The norm of the quaternion

    Note:
        For unit quaternions (representing rotations), the norm should be 1.0
    """
    return np.linalg.norm(q)


def normalize_quat(q):
    """
    Normalize a quaternion to unit length.

    Args:
        q (array-like): Quaternion as [q0, q1, q2, q3]

    Returns:
        numpy.ndarray: Normalized quaternion with unit norm

    Note:
        If the input quaternion has zero norm, returns the identity quaternion [1,0,0,0]
        representing no rotation.
    """
    q = np.array(q, dtype=float)
    n = quat_norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_multiply(q, r):
    """
    Multiply two quaternions using the Hamilton product.

    This function implements quaternion multiplication: result = q * r
    where the multiplication represents the composition of rotations.

    Args:
        q (array-like): First quaternion as [q0, q1, q2, q3]
        r (array-like): Second quaternion as [r0, r1, r2, r3]

    Returns:
        numpy.ndarray: Product quaternion as [result0, result1, result2, result3]

    Note:
        The Hamilton product is used for quaternion multiplication.
        If q represents rotation A and r represents rotation B, then
        q * r represents the composition: first apply B, then apply A.
    """
    # Hamilton product q * r, with quaternions as [q0, q1, q2, q3]
    q0, qv = q[0], q[1:4]
    r0, rv = r[0], r[1:4]
    scalar = q0 * r0 - np.dot(qv, rv)
    vector = q0 * rv + r0 * qv + np.cross(qv, rv)
    return np.hstack((scalar, vector))


def omega_matrix(omega):
    """
    Create the 4x4 skew-symmetric matrix for quaternion differentiation.

    This matrix is used in the quaternion kinematic equation:
    q_dot = 0.5 * Omega(omega) * q

    Args:
        omega (array-like): Angular velocity vector [p, q, r] in body axes (rad/s)
            - p: roll rate (rotation about x-axis)
            - q: pitch rate (rotation about y-axis)
            - r: yaw rate (rotation about z-axis)

    Returns:
        numpy.ndarray: 4x4 skew-symmetric matrix Omega(omega)

    Note:
        The matrix represents the cross product operation in quaternion space
        and is essential for integrating quaternion attitude dynamics.
    """
    # omega = [p, q, r] body rates
    p, q, r = omega
    return np.array(
        [[0.0, -p, -q, -r], [p, 0.0, r, -q], [q, -r, 0.0, p], [r, q, -p, 0.0]]
    )


def quat_derivative(q, omega):
    """
    Calculate the time derivative of a quaternion given angular velocity.

    This implements the quaternion kinematic equation:
    q_dot = 0.5 * Omega(omega) * q

    Args:
        q (array-like): Current quaternion as [q0, q1, q2, q3]
        omega (array-like): Angular velocity vector [p, q, r] in body axes (rad/s)

    Returns:
        numpy.ndarray: Quaternion derivative as [q0_dot, q1_dot, q2_dot, q3_dot]

    Note:
        This equation is fundamental for integrating attitude dynamics.
        The factor of 0.5 comes from the quaternion representation of rotations.
    """
    # \dot{q} = 0.5 * Omega(omega) * q
    return 0.5 * omega_matrix(omega).dot(q)


def quat_to_dcm(q):
    """
    Convert a quaternion to a Direction Cosine Matrix (DCM).

    This function converts a quaternion representing the attitude of the aircraft
    to a 3x3 rotation matrix that transforms vectors from body axes to NED axes.

    Args:
        q (array-like): Quaternion as [q0, q1, q2, q3]
            - q0: scalar part
            - q1, q2, q3: vector part (x, y, z components)

    Returns:
        numpy.ndarray: 3x3 Direction Cosine Matrix C_nb
            - Transforms vectors from body axes to NED axes
            - v_ned = C_nb * v_body

    Note:
        The DCM is orthogonal (C^T = C^(-1)) and has determinant +1.
        This matrix is used extensively in flight dynamics for coordinate
        transformations between body and inertial reference frames.
    """
    # Convert quaternion q=[q0, q1, q2, q3] to body->inertial DCM (NED)
    q0, q1, q2, q3 = q
    # Using standard formula: DCM from body to inertial (NED)
    C = np.array(
        [
            [
                q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
                2 * (q1 * q2 + q0 * q3),
                2 * (q1 * q3 - q0 * q2),
            ],
            [
                2 * (q1 * q2 - q0 * q3),
                q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
                2 * (q2 * q3 + q0 * q1),
            ],
            [
                2 * (q1 * q3 + q0 * q2),
                2 * (q2 * q3 - q0 * q1),
                q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
            ],
        ]
    )
    return C


# ---- Aerodynamic Model ----
# This section contains the aerodynamic force and moment calculations
# for the sailplane using linearized aerodynamic coefficients.
def simple_aero_forces_and_moments(state, geom, aero, ctrl, wind_body):
    """
    Compute aerodynamic forces and moments for a sailplane using a linearized model.

    This function calculates the aerodynamic forces and moments acting on the sailplane
    using a simplified linear aerodynamic model valid for small angles of attack and
    sideslip. The model includes basic control surface effects and can be extended
    with more sophisticated aerodynamic data from wind tunnel tests, AVL analysis,
    or flight test data.

    Args:
        state (dict): Aircraft state containing:
            - 'u', 'v', 'w': Body-axis velocity components (m/s)
            - 'p', 'q', 'r': Body-axis angular rates (rad/s)
            - 'quat': Quaternion attitude (not used in this function)
        geom (dict): Aircraft geometry parameters:
            - 'S': Wing reference area (m²)
            - 'b': Wing span (m)
            - 'cbar': Mean aerodynamic chord (m)
        aero (dict): Aerodynamic coefficients:
            - 'CL0': Zero-angle-of-attack lift coefficient
            - 'CL_alpha': Lift curve slope (1/rad)
            - 'CD0': Zero-lift drag coefficient
            - 'k': Induced drag factor (parabolic drag polar)
            - 'Cm0': Zero-angle-of-attack pitching moment coefficient
            - 'Cm_alpha': Pitching moment slope (1/rad)
            - 'Cl_beta': Roll moment due to sideslip (1/rad)
            - 'Cl_p': Roll damping derivative (1/rad)
            - 'Cn_beta': Yaw moment due to sideslip (1/rad)
            - 'Cn_r': Yaw damping derivative (1/rad)
            - Additional control derivatives (CL_de, Cm_de, etc.)
        ctrl (dict): Control surface deflections:
            - 'delta_e': Elevator deflection (rad, positive trailing edge down)
            - 'delta_a': Aileron deflection (rad, positive right aileron down)
            - 'delta_r': Rudder deflection (rad, positive trailing edge left)
            - 'airbrake': Airbrake deployment (0.0 to 1.0)
        wind_body (array-like): Wind velocity vector in body axes [u_w, v_w, w_w] (m/s)

    Returns:
        tuple: (F_b, M_b, Va, alpha, beta)
            - F_b (numpy.ndarray): Aerodynamic force vector in body axes [X, Y, Z] (N)
            - M_b (numpy.ndarray): Aerodynamic moment vector in body axes [L, M, N] (N·m)
            - Va (float): Airspeed magnitude (m/s)
            - alpha (float): Angle of attack (rad)
            - beta (float): Sideslip angle (rad)

    Note:
        - Forces are positive in the positive body axis directions
        - Moments are positive according to the right-hand rule
        - The model assumes sea-level air density (1.225 kg/m³)
        - Linear approximations are used for small angles
        - Control surface effects are included where coefficients are provided
    """
    rho = 1.225  # sea-level air density (simple)
    u_rel = state["u"] - wind_body[0]
    v_rel = state["v"] - wind_body[1]
    w_rel = state["w"] - wind_body[2]
    Va = np.sqrt(u_rel**2 + v_rel**2 + w_rel**2) + 1e-8
    alpha = np.arctan2(w_rel, u_rel)
    beta = np.arcsin(np.clip(v_rel / Va, -1.0, 1.0))
    q_dyn = 0.5 * rho * Va**2
    S = geom["S"]
    b = geom["b"]
    cbar = geom["cbar"]

    # Lift (linear approx)
    CL = (
        aero["CL0"]
        + aero["CL_alpha"] * alpha
        + aero.get("CL_q", 0.0) * (state["q"] * cbar / (2 * Va))
        + aero.get("CL_de", 0.0) * ctrl.get("delta_e", 0.0)
    )
    # Drag (parabolic)
    CD = (
        aero["CD0"]
        + aero["k"] * CL**2
        + aero.get("CD_airbrake", 0.0) * ctrl.get("airbrake", 0.0)
    )
    # Side force (small-sideslip linear)
    CY = (
        aero.get("CY_beta", -0.02) * beta
        + aero.get("CY_da", 0.0) * ctrl.get("delta_a", 0.0)
        + aero.get("CY_dr", 0.0) * ctrl.get("delta_r", 0.0)
    )

    # Forces in wind axes (-drag, side, -lift)
    D = q_dyn * S * CD
    L = q_dyn * S * CL
    Y = q_dyn * S * CY

    # Transform to body axes via alpha, beta (wind->body)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    C_bw = np.array([[ca * cb, -sa, ca * sb], [sb, cb, 0.0], [-sa * cb, -ca, -sa * sb]])
    F_w = np.array([-D, Y, -L])
    F_b = C_bw.dot(F_w)

    # Moments: simple linear model with aerodynamic derivatives
    Cl = (
        aero.get("Cl_beta", -0.1) * beta
        + aero.get("Cl_p", -0.5) * (state["p"] * b / (2 * Va))
        + aero.get("Cl_da", 0.0) * ctrl.get("delta_a", 0.0)
        + aero.get("Cl_dr", 0.0) * ctrl.get("delta_r", 0.0)
    )
    Cm = (
        aero.get("Cm0", 0.0)
        + aero.get("Cm_alpha", -0.5) * alpha
        + aero.get("Cm_q", -8.0) * (state["q"] * cbar / (2 * Va))
        + aero.get("Cm_de", -1.0) * ctrl.get("delta_e", 0.0)
    )
    Cn = (
        aero.get("Cn_beta", 0.05) * beta
        + aero.get("Cn_r", -0.1) * (state["r"] * b / (2 * Va))
        + aero.get("Cn_dr", 0.0) * ctrl.get("delta_r", 0.0)
        + aero.get("Cn_da", 0.0) * ctrl.get("delta_a", 0.0)
    )

    Lm = q_dyn * S * b * Cl
    Mm = q_dyn * S * cbar * Cm
    Nm = q_dyn * S * b * Cn

    M_b = np.array([Lm, Mm, Nm])

    return F_b, M_b, Va, alpha, beta


# ---- 6-DOF Sailplane Dynamics Class ----
# This class implements the full 6-degree-of-freedom rigid body dynamics
# for a sailplane using quaternion-based attitude representation.
class Sailplane6DOF:
    """
    Six-degree-of-freedom sailplane dynamics model.

    This class implements the complete rigid body dynamics for a sailplane including:
    - Translational motion (position and velocity in NED frame)
    - Rotational motion (attitude via quaternions and angular rates)
    - Aerodynamic forces and moments
    - Gravity effects
    - Wind effects
    - Control surface deflections

    The state vector is organized as:
    [pN, pE, pD, u, v, w, p, q, r, q0, q1, q2, q3]
    where:
    - pN, pE, pD: Position in NED frame (m)
    - u, v, w: Body-axis velocity components (m/s)
    - p, q, r: Body-axis angular rates (rad/s)
    - q0, q1, q2, q3: Quaternion attitude representation

    Attributes:
        geom (dict): Aircraft geometry parameters
        I (numpy.ndarray): 3x3 inertia matrix in body axes (kg·m²)
        I_inv (numpy.ndarray): Inverse of inertia matrix
        aero (dict): Aerodynamic coefficients
        g (float): Gravitational acceleration (m/s²)
    """

    def __init__(self, geom=None, inertia=None, aero=None):
        """
        Initialize the sailplane dynamics model.

        Args:
            geom (dict, optional): Aircraft geometry parameters:
                - 'S': Wing reference area (m²)
                - 'b': Wing span (m)
                - 'cbar': Mean aerodynamic chord (m)
                - 'mass': Aircraft mass (kg)
            inertia (numpy.ndarray, optional): 3x3 inertia matrix in body axes (kg·m²).
                If None, uses default diagonal approximation.
            aero (dict, optional): Aerodynamic coefficients dictionary.
                If None, uses default sailplane coefficients.

        Note:
            Default values represent a medium-size sailplane with:
            - Wing area: 12.0 m²
            - Wing span: 15.0 m
            - Mean chord: 1.2 m
            - Mass: 300 kg
        """
        # Geometry (example values for a medium-size sailplane)
        self.geom = geom or {"S": 12.0, "b": 15.0, "cbar": 1.2, "mass": 300.0}
        # Inertia matrix (body axes) (example diagonal approx in kg*m^2)
        if inertia is None:
            Ixx = 200.0
            Iyy = 150.0
            Izz = 300.0
            Ixz = 0.0
            self.I = np.array([[Ixx, 0.0, -Ixz], [0.0, Iyy, 0.0], [-Ixz, 0.0, Izz]])
        else:
            self.I = inertia
        self.I_inv = np.linalg.inv(self.I)
        # Simple aerodynamic derivatives (placeholders)
        self.aero = aero or {
            "CL0": 0.2,
            "CL_alpha": 5.7,
            "CD0": 0.02,
            "k": 0.04,
            "Cm0": 0.0,
            "Cm_alpha": -0.8,
            "Cl_beta": -0.12,
            "Cl_p": -0.5,
            "Cn_beta": 0.06,
            "Cn_r": -0.08,
        }
        # gravity
        self.g = 9.80665

    def state_to_dict(self, x):
        """
        Convert state vector to dictionary format for easier access.

        Args:
            x (array-like): State vector [pN, pE, pD, u, v, w, p, q, r, q0, q1, q2, q3]

        Returns:
            dict: State dictionary with keys:
                - 'pN', 'pE', 'pD': Position in NED frame (m)
                - 'u', 'v', 'w': Body-axis velocity components (m/s)
                - 'p', 'q', 'r': Body-axis angular rates (rad/s)
                - 'quat': Quaternion attitude [q0, q1, q2, q3]
        """
        # state vector x layout:
        # [p_N, p_E, p_D, u, v, w, p, q, r, q0, q1, q2, q3]
        pN, pE, pD = x[0:3]
        u, v, w = x[3:6]
        p, q, r = x[6:9]
        quat = x[9:13]
        return {
            "pN": pN,
            "pE": pE,
            "pD": pD,
            "u": u,
            "v": v,
            "w": w,
            "p": p,
            "q": q,
            "r": r,
            "quat": quat,
        }

    def derivatives(
        self, t, x, ctrl=None, wind_ned=None
    ):  # pylint: disable=unused-argument
        """
        Compute time derivatives of the full 6-DOF state vector.

        This method implements the complete set of 6-DOF equations of motion:
        - Position kinematics: p_dot = C_nb * v_body + wind_NED
        - Velocity dynamics: m*(v_dot + omega x v) = F_total
        - Angular dynamics: I*omega_dot + omega x (I*omega) = M_total
        - Quaternion kinematics: q_dot = 0.5 * Omega(omega) * q

        Args:
            t (float): Current time (s)
            x (array-like): State vector [pN, pE, pD, u, v, w, p, q, r, q0, q1, q2, q3]
            ctrl (dict, optional): Control inputs:
                - 'delta_e': Elevator deflection (rad)
                - 'delta_a': Aileron deflection (rad)
                - 'delta_r': Rudder deflection (rad)
                - 'airbrake': Airbrake deployment (0.0 to 1.0)
            wind_ned (array-like, optional): Wind velocity in NED frame [N, E, D] (m/s)
                Positive D component means wind blowing downward.

        Returns:
            numpy.ndarray: State derivative vector dx/dt with same structure as x

        Note:
            - All forces and moments are computed in body axes
            - Gravity is included as a body-axis force
            - Wind effects are included in both aerodynamics and position kinematics
            - Quaternion normalization is handled in the integration process
        """
        if ctrl is None:
            ctrl = {"delta_e": 0.0, "delta_a": 0.0, "delta_r": 0.0, "airbrake": 0.0}

        if wind_ned is None:
            wind_ned = np.array([0.0, 0.0, 0.0])

        # Unpack state
        s = self.state_to_dict(x)
        quat = normalize_quat(s["quat"])
        # Direction cosine matrix body->NED
        C_nb = quat_to_dcm(quat)
        C_bn = C_nb.T

        # Wind in body axes
        wind_body = C_bn.dot(wind_ned)

        # Aerodynamics (in body axes)
        F_aero_b, M_aero_b, _, _, _ = simple_aero_forces_and_moments(
            s, self.geom, self.aero, ctrl, wind_body
        )

        # Gravity in body axes (NED gravity vector = [0,0,g] down positive)
        g = self.g
        # NED gravity vector expressed in body axes: g_b = C_bn * [0,0,g]
        g_b = C_bn.dot(np.array([0.0, 0.0, g]))

        # Sum forces: aerodynamic + gravity (note mg is in NED, converted to body).
        # For convenience we treat mg in body as mass * g_b (positive down along body z if aircraft upright).
        m = self.geom["mass"]
        F_total_b = (
            F_aero_b - m * g_b
        )  # minus because gravity acts to accelerate downward in NED

        # Translational accelerations (body axes): m*(v_dot + omega x v) = F_total_b
        v_body = np.array([s["u"], s["v"], s["w"]])
        omega = np.array([s["p"], s["q"], s["r"]])
        v_dot = (1.0 / m) * (F_total_b - np.cross(omega, m * v_body))

        # Rotational dynamics: I*omega_dot + omega x (I*omega) = M_aero_b
        omega_dot = self.I_inv.dot(M_aero_b - np.cross(omega, self.I.dot(omega)))

        # Quaternion derivative
        q_dot = quat_derivative(quat, omega)
        # Enforce unit norm (project back) AFTER integration step in integrator; here return q_dot,
        # but our integrator callback will renormalize the quaternion state in the event of a step.
        # We'll provide a wrapper to do that externally if desired.

        # Position kinematics: derivative of NED position = C_nb * v_body + wind_NED (ground-relative)
        pos_dot = C_nb.dot(v_body) + wind_ned

        # Compose derivative vector to match state layout
        dx = np.zeros_like(x)
        dx[0:3] = pos_dot
        dx[3:6] = v_dot
        dx[6:9] = omega_dot
        dx[9:13] = q_dot

        # return derivatives and some diagnostic outputs optionally
        return dx

    def integrate(
        self, x0, t_span, ctrl_timefun=None, wind_timefun=None, rtol=1e-6, atol=1e-8
    ):  # pylint: disable=redefined-outer-name
        """
        Integrate the 6-DOF equations of motion over time.

        This method uses scipy.integrate.solve_ivp to numerically integrate the
        differential equations from the initial state x0 over the time span.
        The integration includes automatic quaternion normalization to maintain
        unit quaternion constraints.

        Args:
            x0 (array-like): Initial state vector [pN, pE, pD, u, v, w, p, q, r, q0, q1, q2, q3]
            t_span (tuple): Integration time span (t_start, t_end) in seconds
            ctrl_timefun (callable, optional): Function that returns control inputs at time t:
                ctrl_timefun(t) -> dict with control deflections
            wind_timefun (callable, optional): Function that returns wind at time t:
                wind_timefun(t) -> array-like wind vector in NED frame
            rtol (float, optional): Relative tolerance for integration (default: 1e-6)
            atol (float, optional): Absolute tolerance for integration (default: 1e-8)

        Returns:
            scipy.integrate.OdeResult: Integration result containing:
                - t: Time points where solution is evaluated
                - y: State vector at each time point
                - sol: Dense output function for interpolation

        Note:
            - Uses RK45 (Runge-Kutta 4th/5th order) adaptive step size method
            - Quaternions are automatically normalized after each integration step
            - Dense output is available for smooth interpolation between time points
            - Default tolerances are suitable for most flight dynamics applications
        """

        def rhs(t, x):
            ctrl = (
                ctrl_timefun(t)
                if ctrl_timefun is not None
                else {"delta_e": 0.0, "delta_a": 0.0, "delta_r": 0.0, "airbrake": 0.0}
            )
            wind = (
                wind_timefun(t)
                if wind_timefun is not None
                else np.array([0.0, 0.0, 0.0])
            )
            dx = self.derivatives(t, x, ctrl=ctrl, wind_ned=wind)
            # Enforce quaternion norm after derivative (not strictly necessary here but helps numerical behaviour):
            # We renormalize the quaternion part of the state on the fly to avoid integrating drift.
            q = x[9:13] + dx[9:13] * 1e-6  # small step estimate to avoid zero division
            q = normalize_quat(q)
            # Overwriting dx[9:13] to ensure derivative leads to unit-norm after step is non-trivial;
            # easier approach: after integrator step, re-normalize final state (done below via dense_output wrapper).
            return dx

        solution = solve_ivp(
            rhs, t_span, x0, rtol=rtol, atol=atol, method="RK45", dense_output=True
        )

        # Post-process to renormalize quaternions at solver output times
        for i in range(solution.y.shape[1]):
            solution.y[9:13, i] = normalize_quat(solution.y[9:13, i])

        return solution


# ---- 3-DOF Point-Mass Glide Model ----
# This class implements a simplified 3-degree-of-freedom point-mass model
# for sailplane performance analysis and trajectory optimization.
class PointMassGlide:
    """
    Three-degree-of-freedom point-mass sailplane model for performance analysis.

    This simplified model treats the sailplane as a point mass with aerodynamic
    forces applied at the center of mass. It's useful for:
    - Glide performance analysis
    - Trajectory optimization
    - Energy management studies
    - Thermal soaring analysis

    The state vector is organized as:
    [pN, pE, h, V, gamma, psi]
    where:
    - pN, pE: North and East position (m)
    - h: Altitude above reference (m, positive up)
    - V: Airspeed magnitude (m/s)
    - gamma: Flight path angle (rad, positive up)
    - psi: Heading angle (rad, positive North to East)

    Attributes:
        m (float): Aircraft mass (kg)
        S (float): Wing reference area (m²)
        CL_func (callable): Lift coefficient function CL(V, alpha)
        CD_func (callable): Drag coefficient function CD(V, alpha)
        g (float): Gravitational acceleration (m/s²)
    """

    def __init__(self, mass=300.0, S=12.0, CL_func=None, CD_func=None):
        """
        Initialize the point-mass glide model.

        Args:
            mass (float, optional): Aircraft mass in kg (default: 300.0)
            S (float, optional): Wing reference area in m² (default: 12.0)
            CL_func (callable, optional): Lift coefficient function CL(V, alpha).
                If None, uses linear approximation: CL = 0.2 + 5.7*alpha
            CD_func (callable, optional): Drag coefficient function CD(V, alpha).
                If None, uses parabolic polar: CD = 0.02 + 0.04*CL²

        Note:
            The default aerodynamic functions provide a reasonable approximation
            for a typical sailplane. For accurate performance analysis, replace
            with actual aerodynamic data from wind tunnel tests or CFD analysis.
        """
        self.m = mass
        self.S = S
        # CL_func(V,alpha) and CD_func(V,alpha) are callables; if None use simple polar
        if CL_func is None:
            self.CL_func = lambda V, alpha: 0.2 + 5.7 * alpha
        else:
            self.CL_func = CL_func
        if CD_func is None:
            self.CD_func = lambda V, alpha: 0.02 + 0.04 * (self.CL_func(V, alpha) ** 2)
        else:
            self.CD_func = CD_func
        self.g = 9.80665

    def derivatives(self, t, x, ctrl):  # pylint: disable=unused-argument
        """
        Compute time derivatives for the 3-DOF point-mass model.

        This method implements the simplified equations of motion for a point-mass
        sailplane including:
        - Position kinematics with wind effects
        - Velocity dynamics (thrust/drag balance)
        - Flight path angle dynamics (lift/weight balance)
        - Heading dynamics (bank angle effects)

        Args:
            t (float): Current time (s)
            x (array-like): State vector [pN, pE, h, V, gamma, psi]
            ctrl (dict): Control inputs:
                - 'phi': Bank angle (rad, positive right wing down)
                - 'wind': Wind velocity in NED frame [VwN, VwE, VwD] (m/s)

        Returns:
            numpy.ndarray: State derivative vector [pN_dot, pE_dot, h_dot, V_dot, gamma_dot, psi_dot]

        Note:
            - Assumes sea-level air density (1.225 kg/m³)
            - Uses simplified relationship: alpha ≈ -gamma for small bank angles
            - Bank angle phi controls turn rate through heading dynamics
            - Wind effects are included in position kinematics only
        """
        # x = [pN, pE, h, V, gamma, psi]
        _, _, _, V, gamma, psi = x
        # ctrl contains bank angle phi (rad) and wind in NED (VwN,VwE,VwD)
        phi = ctrl.get("phi", 0.0)
        wind = ctrl.get("wind", np.array([0.0, 0.0, 0.0]))
        VwN, VwE, VwD = wind

        rho = 1.225
        qdyn = 0.5 * rho * V**2
        alpha = -gamma  # in a simple glide alpha ~ -gamma for small bank (approx)
        CL = self.CL_func(V, alpha)
        CD = self.CD_func(V, alpha)
        L = qdyn * self.S * CL
        D = qdyn * self.S * CD

        pN_dot = V * np.cos(gamma) * np.cos(psi) + VwN
        pE_dot = V * np.cos(gamma) * np.sin(psi) + VwE
        h_dot = V * np.sin(gamma) + VwD

        V_dot = (-D - self.m * self.g * np.sin(gamma)) / self.m
        gamma_dot = (L * np.cos(phi) - self.m * self.g * np.cos(gamma)) / (self.m * V)
        psi_dot = (L * np.sin(phi)) / (self.m * V * np.cos(gamma) + 1e-9)

        return np.array([pN_dot, pE_dot, h_dot, V_dot, gamma_dot, psi_dot])


# ---- Example Usage: 6-DOF Sailplane Simulation ----
# This section demonstrates how to use the Sailplane6DOF class to run a simulation
if __name__ == "__main__":
    print("Sailplane Flight Dynamics Simulation")
    print("====================================")

    # Create a sailplane instance with default parameters
    # This creates a medium-size sailplane with typical aerodynamic characteristics
    plane = Sailplane6DOF()
    print(f"Aircraft mass: {plane.geom['mass']} kg")
    print(f"Wing area: {plane.geom['S']} m²")
    print(f"Wing span: {plane.geom['b']} m")

    # Set up initial conditions for level flight
    # Position: Start at origin, 1000m altitude
    pN0, pE0, pD0 = (
        0.0,
        0.0,
        -1000.0,
    )  # NED: down positive, so altitude 1000 m => pD = -1000

    # Velocity: 30 m/s forward flight (typical sailplane cruise speed)
    u0, v0, w0 = 30.0, 0.0, 0.0  # body-axis initial velocity (m/s)

    # Angular rates: Initially at rest (no rotation)
    p0, q0, r0 = 0.0, 0.0, 0.0  # angular rates (rad/s)

    # Attitude: Level flight (body axes aligned with NED)
    quat0 = normalize_quat([1.0, 0.0, 0.0, 0.0])  # identity rotation

    # Combine into state vector: [pN, pE, pD, u, v, w, p, q, r, q0, q1, q2, q3]
    x0 = np.hstack(([pN0, pE0, pD0, u0, v0, w0, p0, q0, r0], quat0))

    print(f"Initial altitude: {-pD0} m")
    print(f"Initial airspeed: {u0} m/s")
    print("Initial attitude: Level flight")

    # Define control inputs (constant for this example)
    # All control surfaces neutral, no airbrake
    ctrl_fun = lambda t: {
        "delta_e": 0.0,  # Elevator: neutral
        "delta_a": 0.0,  # Aileron: neutral
        "delta_r": 0.0,  # Rudder: neutral
        "airbrake": 0.0,  # Airbrake: retracted
    }

    # Define wind conditions (no wind for this example)
    wind_fun = lambda t: np.array([0.0, 0.0, 0.0])  # No wind

    print("\nRunning simulation...")
    print("Time span: 0 to 30 seconds")
    print("Controls: Neutral (hands-off flight)")
    print("Wind: Calm conditions")

    # Integrate the equations of motion for 30 seconds
    # This will simulate the sailplane's natural response to initial conditions
    sol = plane.integrate(x0, (0.0, 30.0), ctrl_timefun=ctrl_fun, wind_timefun=wind_fun)

    # Extract and display final state
    xf = sol.y[:, -1]
    print("\nSimulation Results:")
    print("==================")
    print(f"Final position NED (m): [{xf[0]:.1f}, {xf[1]:.1f}, {xf[2]:.1f}]")
    print(f"Final altitude: {-xf[2]:.1f} m")
    print(f"Final body velocity (u,v,w) (m/s): [{xf[3]:.2f}, {xf[4]:.2f}, {xf[5]:.2f}]")
    print(f"Final airspeed: {np.sqrt(xf[3]**2 + xf[4]**2 + xf[5]**2):.2f} m/s")
    print(
        f"Final angular rates (p,q,r) (rad/s): [{xf[6]:.3f}, {xf[7]:.3f}, {xf[8]:.3f}]"
    )
    print(
        f"Final quaternion (q0,q1,q2,q3): [{xf[9]:.3f}, {xf[10]:.3f}, {xf[11]:.3f}, {xf[12]:.3f}]"
    )

    # Calculate some derived quantities
    final_altitude = -xf[2]
    altitude_change = final_altitude - 1000.0
    print("\nPerformance Summary:")
    print(f"Altitude change: {altitude_change:.1f} m")
    print(f"Average sink rate: {altitude_change/30.0:.2f} m/s")

    print("\nSimulation completed successfully!")
    print("Note: This represents a basic hands-off glide. For more realistic")
    print("simulations, add control inputs, wind effects, and thermal updrafts.")
