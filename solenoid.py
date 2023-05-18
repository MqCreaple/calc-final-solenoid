from manim import *

class Solenoid(ThreeDScene):
    def cylinder_surface(self, rad: float, z_len: float):
        return Surface(
            lambda u, v: np.array([
                rad * np.cos(TAU * v),
                rad * np.sin(TAU * v),
                u
            ]),
            resolution=(6, 32),
            u_range=[-z_len / 2, z_len / 2],
            v_range=[0, 1],
        )
    def construct(self):
        axes = ThreeDAxes(x_range = [-2, 2], y_range = [-2, 2], z_range = [-2, 2])
        self.add(axes)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(self.cylinder_surface(1, 8)))
        self.wait(2)