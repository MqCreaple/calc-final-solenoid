from manim import *
from collections.abc import *
from colour import *
import numpy as np

class MySur(VMobject):
    def __init__(
            self,
            func: Callable[[float, float], np.ndarray],
            u_range: Sequence[float] = [0, 1],
            v_range: Sequence[float] = [0, 1],
            resolution: Sequence[int] = [32, 32],
            stroke_color: Color = WHITE,
            stroke_width: float = 1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.u_range = u_range
        self.v_range = v_range
        self.resolution = resolution
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.u_curves: Sequence[ParametricFunction] = []              # curves that has a constant u
        self.v_curves: Sequence[ParametricFunction] = []              # curves that has a constant v
        self._init_curves(func)

    def _init_curves(self, func: Callable[[float, float], np.ndarray]):
        for u in np.linspace(self.u_range[0], self.u_range[1], self.resolution[0]):
            curve = ParametricFunction(
                lambda v: func(u, v),
                t_range = self.v_range,
                color = self.stroke_color,
                stroke_width = self.stroke_width,
            )
            self.u_curves.append(curve)
            self.add(curve)
        for v in np.linspace(self.v_range[0], self.v_range[1], self.resolution[1]):
            curve = ParametricFunction(
                lambda u: func(u, v),
                t_range = self.u_range,
                color = self.stroke_color,
                stroke_width = self.stroke_width,
            )
            self.v_curves.append(curve)
            self.add(curve)

    def get_u_from_index(self, u: float):
        return np.interp(u, [0, self.resolution[0]], self.u_range)

    def get_v_from_index(self, v: float):
        return np.interp(v, [0, self.resolution[1]], self.v_range)

def solenoid_wire(radius, zlen, coil_cnt, **kwargs):
    coil_height = zlen / coil_cnt
    return ParametricFunction(
        lambda t: np.array([
            radius * np.cos(2 * PI * t),
            radius * np.sin(2 * PI * t),
            t * coil_height
        ]),
        t_range=[-coil_cnt / 2, coil_cnt / 2],
        **kwargs
    )

class Solenoid(ParametricFunction):
    def __init__(self, radius : float, zlen : float, coil_cnt : float, **kwargs):
        self.radius = radius
        self.zlen = zlen
        self.coil_cnt = coil_cnt
        self.coil_height = zlen / coil_cnt
        super().__init__(
            lambda t: np.array([
                radius * np.cos(2 * PI * t),
                radius * np.sin(2 * PI * t),
                t * self.coil_height
            ]),
            t_range=[-coil_cnt / 2, coil_cnt / 2],
            **kwargs
        )
    # get the points on the cross-section plane, that the solenoid wire passes through it
    def get_pts_pass_out_plane(self, angle : float) -> Sequence[np.ndarray]:
        # if angle=0, the wire when t=0 will pass through the plane
        angle = angle % (2 * PI)
        t = angle / (2 * PI) - self.coil_cnt / 2
        return [self.function(t + n) for n in range(0, self.coil_cnt)]
    def get_pts_pass_in_plane(self, angle : float) -> Sequence[np.ndarray]:
        angle = angle % (2 * PI)
        t = angle / (2 * PI) - self.coil_cnt / 2
        return [self.function(t + n + 0.5) for n in range(0, self.coil_cnt)]
    def get_inout_symbols(self, radius : float, ang : float, reverse : bool, **kwargs) -> VGroup:
        in_poses = self.get_pts_pass_in_plane(0)
        out_poses = self.get_pts_pass_out_plane(0)
        if reverse:
            in_poses, out_poses = out_poses, in_poses
        in_symbols = VGroup(*[current_in_symbol(pos, radius, **kwargs).rotate(PI/2, RIGHT) for pos in in_poses])
        out_symbols = VGroup(*[current_out_symbol(pos, radius, **kwargs).rotate(PI/2, RIGHT) for pos in out_poses])
        ret = VGroup(in_symbols, out_symbols)
        shift_val = ang / (2 * PI) * self.coil_height
        ret.rotate(ang, OUT).shift(np.array([0,0,shift_val]))
        return ret
    
def current_in_symbol(center: np.array, radius: float, **kwargs) -> VGroup:
        circ = Circle(radius, **kwargs).move_to(center)
        # draw the cross symbol
        line1 = Line(start=circ.get_left(), end=circ.get_right(), **kwargs).rotate(PI/4, OUT)
        line2 = Line(start=circ.get_top(), end=circ.get_bottom(), **kwargs).rotate(PI/4, OUT)
        return VGroup(circ, line1, line2)

def current_out_symbol(center: np.array, radius: float, **kwargs) -> VGroup:
        circ = Circle(radius, **kwargs).move_to(center)
        print(circ.stroke_width)
        # draw the dot symbol
        dot = Dot(circ.get_center(), **kwargs)
        return VGroup(circ, dot)

def vec_by_polar(r : float, theta : float) -> np.ndarray:
    return r * np.array([np.cos(theta), np.sin(theta), 0])

class TipedCircle(VGroup):
    def __init__(self, radius : float, tip_cnt : int, is_ccw : bool, arr_color, **kwargs):
        self.circ = Circle(radius=radius, **kwargs)
        self.tip_cnt = tip_cnt
        ang_offset = 2*PI/self.tip_cnt
        self.tips = []
        for i in range(self.tip_cnt):
            tip_angle = i*ang_offset + PI/2
            tip_point = vec_by_polar(radius, i * ang_offset)

            tip = Triangle(color = arr_color, fill_color = arr_color, fill_opacity=1)
            tip.rotate(30 * DEGREES)
            tip.rotate(tip_angle).scale(
                self.circ.stroke_width / 40).move_to(tip_point)
            self.tips.append(tip)
        tmp = VGroup(self.circ, *self.tips)
        if not is_ccw:
            tmp.rotate(PI, RIGHT)
        super().__init__(tmp)

def face_camera(object: Mobject, theta: float, phi: float):
        """
        Rotate the object to face the camera with given `theta` and `phi` angles.
        """
        return object.rotate(phi, axis = RIGHT).rotate(90 * DEGREES + theta, axis = OUT)