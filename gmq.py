from manim import *
import numpy as np
import helper

class BiotSavart(ThreeDScene):
    """
    Demonstration of Biot-Savart Law
    """
    def rotate_to_face_camera(self, object: Mobject):
        return object.rotate(self.camera.get_phi(), axis = RIGHT).rotate(90 * DEGREES + self.camera.get_theta(), axis = OUT)

    def magnetic_vec(self, current: Vector, observer: Vector):
        ret = Vector(direction = 3.0 * np.cross(current.get_end(), observer.get_end()) / (np.linalg.norm(observer.get_end()) ** 2), color = BLUE)
        ret.shift(observer.get_end())
        return ret

    def construct(self):
        axes = ThreeDAxes(x_range = [-2, 2], y_range = [-2, 2], z_range = [-2, 2])
        self.add(axes)
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)
        
        # demonstrate Biot-Savart Law
        current = Vector(direction = np.array([0, 0, 1]), color = YELLOW)
        current_label = self.rotate_to_face_camera(MathTex(r'I\mathrm d\vec l', color = YELLOW)).next_to(current, DOWN)
        observer = Vector(direction = np.array([2, 3, 0]), color = RED)
        magnetic = self.magnetic_vec(current, observer)
        magnetic_label = self.rotate_to_face_camera(MathTex(r'\vec B', color = BLUE)).next_to(magnetic, UP)
        self.play(Create(current), Write(current_label), Create(magnetic), Write(magnetic_label))
        self.wait(0.5)
        self.play(Create(observer))
        self.wait(1)

        observer_new = Vector(direction = [2 / 2, 3 / 2, 0], color = RED)
        magnetic_new = self.magnetic_vec(current, observer_new)
        magnetic_label_new = magnetic_label.copy().next_to(magnetic_new, UP)
        self.play(Transform(observer, observer_new), Transform(magnetic, magnetic_new), Transform(magnetic_label, magnetic_label_new))
        self.wait(1)

        observer_new = Vector(direction = [1, 3, 0], color = RED)
        magnetic_new = self.magnetic_vec(current, observer_new)
        magnetic_label_new = magnetic_label.copy().next_to(magnetic_new, UP)
        self.play(Transform(observer, observer_new), Transform(magnetic, magnetic_new), Transform(magnetic_label, magnetic_label_new))
        self.wait(1)

        observer_new = Vector(direction = [2.5, 1.5, 0], color = RED)
        magnetic_new = self.magnetic_vec(current, observer_new)
        magnetic_label_new = magnetic_label.copy().next_to(magnetic_new, UP)
        self.play(Transform(observer, observer_new), Transform(magnetic, magnetic_new), Transform(magnetic_label, magnetic_label_new))
        self.wait(1)

class SolenoidSlice(ThreeDScene):
    """
    Demonstrate the two different methods to integrate along a solenoid.
    """
    def rotate_to_face_camera(self, object: Mobject):
        return object.rotate(self.camera.get_phi(), axis = RIGHT).rotate(90 * DEGREES + self.camera.get_theta(), axis = OUT)

    def cylinder_curve(self, rad: float, z_len: float, n: int):
        return ParametricFunction(
            lambda u: np.array([
                rad * np.cos(TAU * u),
                rad * np.sin(TAU * u),
                u * z_len / n
            ]),
            t_range = (-n / 2, n / 2, 0.01),
            color = YELLOW,
        )

    def cylinder_surface(self, rad: float, z_len: float, n: int):
        return helper.MySurface(
            lambda u, v: np.array([
                rad * np.cos(TAU * v),
                rad * np.sin(TAU * v),
                u
            ]),
            resolution=(n, 20),
            u_range=[-z_len / 2, z_len / 2],
            v_range=[0, 1],
            stroke_color=YELLOW,
        )

    def construct(self):
        solenoid_radius = 1.5

        axes = ThreeDAxes(x_range = [-2, 2], y_range = [-2, 2], z_range = [-2, 2])
        self.add(axes)
        self.set_camera_orientation(phi=80 * DEGREES, theta=45 * DEGREES)
        
        curve = self.cylinder_curve(solenoid_radius, 16, 40)
        self.play(Create(curve, run_time=2))
        self.wait(3)
        curve2 = self.cylinder_curve(solenoid_radius, 8, 40)
        self.play(Transform(curve, curve2))
        self.wait(2)

        surface = self.cylinder_surface(solenoid_radius, 8, 40).set_fill(opacity = 0.0)
        self.play(Uncreate(curve), Create(surface))
        self.wait(7)

        ## script: The essence of integration is to sum up a lot of small things, although each of these small pieces is different.
        dot = Dot3D(point = axes.coords_to_point(0, 0, 0), color = RED)
        dot_label = self.rotate_to_face_camera(MathTex(r'O', color = RED)).next_to(dot, IN)
        magnetic = Vector([0, 0, 0], color = BLUE)
        magnetic_label = self.rotate_to_face_camera(MathTex(r'\vec B', color = BLUE)).next_to(magnetic, UP)
        self.play(Create(dot), Write(dot_label))
        self.wait(1)

        k = 0.5
        label_added = False
        # animation of sum from u = 10 to 30
        # TODO: continuously move a circle from bottom to top
        magnetic = Vector([0, 0, 0], color = BLUE)
        magnetic_label = self.rotate_to_face_camera(MathTex(r'\vec B', color = BLUE)).next_to(magnetic, UP)
        k = 1.2
        circle = Circle(radius = solenoid_radius, color = BLUE).shift([0, 0, -4])
        circle.generate_target()
        circle.target.shift([0, 0, 8])
        lower_val = k * (-4) / np.sqrt(solenoid_radius ** 2 + 16)
        upper_val = k * 4 / np.sqrt(solenoid_radius ** 2 + 16)
        magnetic.generate_target()
        magnetic.target = Vector([0, 0, upper_val - lower_val], color = BLUE)
        magnetic_label.generate_target()
        magnetic_label.target.next_to(magnetic.target, UP)
        magnetic_rate_func = lambda t: ((k * (t * 8 - 4) / np.sqrt(solenoid_radius ** 2 + (t * 8 - 4) ** 2)) - lower_val) / (upper_val - lower_val)
        self.play(Create(circle), Write(magnetic_label), Create(magnetic))
        self.wait(1)
        self.play(
            MoveToTarget(circle, rate_func = rate_functions.linear, run_time = 2),
            MoveToTarget(magnetic, rate_func = magnetic_rate_func, run_time = 2),
            MoveToTarget(magnetic_label, rate_func = magnetic_rate_func, run_time = 2)
        )
        self.wait(2)
        self.play(Uncreate(circle), Uncreate(magnetic), Uncreate(magnetic_label))

        # animation of sum off all v curves
        magnetic = Vector([0, 0, 0], color = BLUE)
        magnetic_label = self.rotate_to_face_camera(MathTex(r'\vec B', color = BLUE)).next_to(magnetic, UP)
        line = Line(start = [solenoid_radius, 0, -4], end = [solenoid_radius, 0, 4], color = BLUE)
        magnetic.generate_target()
        magnetic.target = Vector([0, 0, upper_val - lower_val], color = BLUE)
        magnetic_label.generate_target()
        magnetic_label.target.next_to(magnetic.target, UP)
        self.play(Create(line), Create(magnetic), Write(magnetic_label))
        self.wait(1)
        self.play(
            Rotate(line, angle = TAU, about_point = ORIGIN, rate_func = rate_functions.linear, run_time = 2),
            MoveToTarget(magnetic, rate_func = rate_functions.linear, run_time = 2),
            MoveToTarget(magnetic_label, rate_func = rate_functions.linear, run_time = 2)
        )
        self.wait(2)
        self.play(Uncreate(line), Uncreate(magnetic), Uncreate(magnetic_label))
        self.wait(5)

        ## script: Well in this case, it turns out that first integrating vertically then integrate the vertical pile of wires around a circle is simpler.
        ## script: Let's focus on one small section of the solenoid:

        line_pile = []
        animations = []
        for v in range(len(surface.v_curves)):
            animations.append(Uncreate(surface.v_curves[v]))
        self.play(*animations)
        animations = []
        for u in range(len(surface.u_curves)):
            height = surface.get_u_from_index(u)
            segment = Vector([0, 0.2, 0], color=YELLOW).shift([solenoid_radius, -0.1, height])
            line_pile.append(segment)
            animations.append(Transform(surface.u_curves[u], segment))
        self.play(*animations)
        self.wait(2)
        self.move_camera(phi=90 * DEGREES, theta=-90 * DEGREES)
        self.wait(2)