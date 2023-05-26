from manim import *
import sys
sys.path.append(".\\manim-physics\\")
from manim_physics import *
print(sys.path)


class solenoid_magfield(ThreeDScene):
    def solenoid_wire(radius, zlen, coil_cnt):
        coil_height = zlen / coil_cnt
        return ParametricFunction(
            lambda t: np.array([
                radius * np.cos(2 * PI * t),
                radius * np.sin(2 * PI * t),
                t * coil_height
            ]),
            t_range=[-coil_cnt / 2, coil_cnt / 2],
            color=YELLOW, 
        )

    def construct(self):
        self.set_camera_orientation(theta=45 * DEGREES, phi=60 * DEGREES)
        # self.wait(1)
        solenoid = solenoid_magfield.solenoid_wire(1, 10, 20).rotate(PI / 2, RIGHT)
        wire = Wire(solenoid, samples=512)
        mag_field = MagneticField(wire,
                                  x_range=[-4, 4],
                                  y_range=[-4, 4],
                                  )
        
        circ_mag_field = MagneticField(
            Wire(Circle(2).rotate(PI / 2, UP)),
            x_range=[-4, 4],
            y_range=[-4, 4],
        )
        self.add(wire, mag_field, circ_mag_field)
