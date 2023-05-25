from manim import *
from manim_physics import *
class MagnetismExample(Scene):
    def construct(self):
        current1 = Current(LEFT * .5)
        current2 = Current(RIGHT * .5, direction=IN)
        field = MagneticField(current1, current2)
        self.add(field, current1, current2)
        