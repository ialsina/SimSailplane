from argparse import ArgumentParser
from dataclasses import dataclass
from numpy import pi

deg = pi / 180

@dataclass
class PlaneArguments:
    # Geometry parameters
    wing_area: float = 12.0  # S (m²)
    wing_span: float = 15.0  # b (m)
    mean_chord: float = 1.2  # cbar (m)
    mass: float = 300.0  # kg
    
    # Inertia parameters (diagonal approximation)
    Ixx: float = 200.0  # kg·m²
    Iyy: float = 150.0  # kg·m²
    Izz: float = 300.0  # kg·m²
    Ixz: float = 0.0    # kg·m²
    
    # Aerodynamic coefficients
    CL0: float = 0.2
    CL_alpha: float = 5.7
    CD0: float = 0.02
    k: float = 0.04
    Cm0: float = 0.0
    Cm_alpha: float = -0.8
    Cl_beta: float = -0.12
    Cl_p: float = -0.5
    Cn_beta: float = 0.06
    Cn_r: float = -0.08

    @classmethod
    def get_argument_parser(cls, parser=None) -> ArgumentParser:
        if parser is None:
            parser = ArgumentParser(description="Aircraft parameters")
        
        # Get default values from the dataclass
        defaults = cls()
        
        # Geometry parameters
        parser.add_argument("--wing_area", type=float, default=defaults.wing_area, help="Wing area (m²)")
        parser.add_argument("--wing_span", type=float, default=defaults.wing_span, help="Wing span (m)")
        parser.add_argument("--mean_chord", type=float, default=defaults.mean_chord, help="Mean aerodynamic chord (m)")
        parser.add_argument("--mass", type=float, default=defaults.mass, help="Aircraft mass (kg)")
        
        # Inertia parameters
        parser.add_argument("--Ixx", type=float, default=defaults.Ixx, help="Roll moment of inertia (kg·m²)")
        parser.add_argument("--Iyy", type=float, default=defaults.Iyy, help="Pitch moment of inertia (kg·m²)")
        parser.add_argument("--Izz", type=float, default=defaults.Izz, help="Yaw moment of inertia (kg·m²)")
        parser.add_argument("--Ixz", type=float, default=defaults.Ixz, help="Cross moment of inertia (kg·m²)")
        
        # Aerodynamic coefficients
        parser.add_argument("--CL0", type=float, default=defaults.CL0, help="Zero-angle-of-attack lift coefficient")
        parser.add_argument("--CL_alpha", type=float, default=defaults.CL_alpha, help="Lift curve slope (1/rad)")
        parser.add_argument("--CD0", type=float, default=defaults.CD0, help="Zero-lift drag coefficient")
        parser.add_argument("--k", type=float, default=defaults.k, help="Induced drag factor")
        parser.add_argument("--Cm0", type=float, default=defaults.Cm0, help="Zero-angle-of-attack pitching moment coefficient")
        parser.add_argument("--Cm_alpha", type=float, default=defaults.Cm_alpha, help="Pitching moment slope (1/rad)")
        parser.add_argument("--Cl_beta", type=float, default=defaults.Cl_beta, help="Roll moment due to sideslip (1/rad)")
        parser.add_argument("--Cl_p", type=float, default=defaults.Cl_p, help="Roll damping derivative (1/rad)")
        parser.add_argument("--Cn_beta", type=float, default=defaults.Cn_beta, help="Yaw moment due to sideslip (1/rad)")
        parser.add_argument("--Cn_r", type=float, default=defaults.Cn_r, help="Yaw damping derivative (1/rad)")
        
        return parser

    @classmethod
    def from_arguments(cls, args) -> "PlaneArguments":
        return cls(
            wing_area=args.wing_area,
            wing_span=args.wing_span,
            mean_chord=args.mean_chord,
            mass=args.mass,
            Ixx=args.Ixx,
            Iyy=args.Iyy,
            Izz=args.Izz,
            Ixz=args.Ixz,
            CL0=args.CL0,
            CL_alpha=args.CL_alpha,
            CD0=args.CD0,
            k=args.k,
            Cm0=args.Cm0,
            Cm_alpha=args.Cm_alpha,
            Cl_beta=args.Cl_beta,
            Cl_p=args.Cl_p,
            Cn_beta=args.Cn_beta,
            Cn_r=args.Cn_r,
        )

    @classmethod
    def parse_args(cls, args=None) -> "PlaneArguments":
        parser = cls.get_argument_parser()
        if args is None:
            args = parser.parse_args()
        return cls.from_arguments(args)

    def to_plane_params(self) -> dict:
        """Convert to parameters suitable for Sailplane6DOF constructor"""
        import numpy as np
        
        # Create inertia matrix
        inertia = np.array([
            [self.Ixx, 0.0, -self.Ixz],
            [0.0, self.Iyy, 0.0],
            [-self.Ixz, 0.0, self.Izz]
        ])
        
        return {
            'geom': {
                'S': self.wing_area,
                'b': self.wing_span,
                'cbar': self.mean_chord,
                'mass': self.mass
            },
            'inertia': inertia,
            'aero': {
                'CL0': self.CL0,
                'CL_alpha': self.CL_alpha,
                'CD0': self.CD0,
                'k': self.k,
                'Cm0': self.Cm0,
                'Cm_alpha': self.Cm_alpha,
                'Cl_beta': self.Cl_beta,
                'Cl_p': self.Cl_p,
                'Cn_beta': self.Cn_beta,
                'Cn_r': self.Cn_r,
            }
        }


@dataclass
class SimArguments:
    steps: int = 30
    discretization_per_control: tuple = (3, 3, 3, 3)
    seed: int = 42
    num: int = 1
    dt: float = 0.5
    bounds: tuple = ((-20 * deg, 20 * deg), (-15 * deg, 15 * deg), (-15 * deg, 15 * deg), (0, 1))
    max_rate: tuple = (30 * deg, 20 * deg, 20 * deg, 0.5)
    initial: tuple = (0, 0, 0, 0)
    initial_state: tuple = (0, 0, -1000, 30, 0, 0, 0, 0, 0, 1, 0, 0, 0)  # [pN, pE, pD, u, v, w, p, q, r, q0, q1, q2, q3]
    wind_ned: tuple = (0, 0, 0)

    @classmethod
    def get_argument_parser(cls, parser=None) -> ArgumentParser:
        if parser is None:
            parser = ArgumentParser(description="Simulation arguments")
        
        # Get default values from the dataclass
        defaults = cls()
        
        parser.add_argument("--steps", type=int, default=defaults.steps)
        parser.add_argument("--discretization_per_control", nargs=4, type=int, default=defaults.discretization_per_control)
        parser.add_argument("--seed", type=int, default=defaults.seed)
        parser.add_argument("--num", type=int, default=defaults.num)
        parser.add_argument("--dt", type=float, default=defaults.dt)
        parser.add_argument("--bounds", nargs=8, type=float, default=list(sum(defaults.bounds, ())))
        parser.add_argument("--max_rate", nargs=4, type=float, default=defaults.max_rate)
        parser.add_argument("--initial", nargs=4, type=float, default=defaults.initial)
        parser.add_argument("--initial_state", nargs=13, type=float, default=defaults.initial_state)
        parser.add_argument("--wind_ned", nargs=3, type=float, default=defaults.wind_ned)
        return parser

    @classmethod
    def from_arguments(cls, args) -> "SimArguments":
        # Convert bounds from flat list to list of tuples
        bounds = [
            (args.bounds[0], args.bounds[1]),  # aileron bounds
            (args.bounds[2], args.bounds[3]),  # elevator bounds
            (args.bounds[4], args.bounds[5]),  # rudder bounds
            (args.bounds[6], args.bounds[7]),  # airbrake bounds
        ]
        
        return cls(
            steps=args.steps,
            discretization_per_control=tuple(args.discretization_per_control),
            seed=args.seed,
            num=args.num,
            dt=args.dt,
            bounds=tuple(bounds),
            max_rate=tuple(args.max_rate),
            initial=tuple(args.initial),
            initial_state=tuple(args.initial_state),
            wind_ned=tuple(args.wind_ned),
        )

    @classmethod
    def parse_args(cls, args=None) -> "SimArguments":
        parser = cls.get_argument_parser()
        if args is None:
            args = parser.parse_args()
        return cls.from_arguments(args)