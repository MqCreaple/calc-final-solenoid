from manim import *
from collections.abc import *
from colour import *
import numpy as np

class MySurface(VMobject):
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