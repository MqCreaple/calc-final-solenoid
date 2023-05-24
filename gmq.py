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
        for u in range(10, 30):
            tiny_cylinder = Cylinder(radius=solenoid_radius, height=8 / 40, resolution=[1, 20], show_ends=False).shift([0, 0, surface.get_u_from_index(u)])
            tiny_cylinder.set_fill(BLUE, opacity = 0.5)
            self.play(Create(tiny_cylinder, run_time = 0.15))
            magnetic_new = Vector([0, 0, magnetic.get_vector()[2] + k / (surface.get_u_from_index(u) ** 2 + solenoid_radius ** 2)], color = BLUE)
            magnetic_label_new = magnetic_label.copy().next_to(magnetic_new, UP)
            label_animation = Transform(magnetic_label, magnetic_label_new, run_time = 0.15)
            if not label_added:
                label_animation = Write(magnetic_label, run_time = 0.15)
                label_added = True
            self.play(Transform(magnetic, magnetic_new, run_time = 0.15), label_animation, Uncreate(tiny_cylinder, run_time = 0.15))
        self.wait(1)
        self.play(Uncreate(magnetic), Unwrite(magnetic_label))

        # animation of sum off all v curves
        magnetic = Vector([0, 0, 0], color = BLUE)
        magnetic_label = self.rotate_to_face_camera(MathTex(r'\vec B', color = BLUE)).next_to(magnetic, UP)
        k = 0.15
        label_added = False
        for v in range(len(surface.v_curves)):
            next_curve = (v + 1) % len(surface.v_curves)
            tiny_stripe = ThreeDVMobject()
            tiny_stripe.set_points_as_corners([
                surface.v_curves[v].points[0],
                surface.v_curves[v].points[-1],
                surface.v_curves[next_curve].points[-1],
                surface.v_curves[next_curve].points[0],
            ]).set_fill(BLUE, opacity = 0.5)
            self.play(Create(tiny_stripe, run_time = 0.15))
            magnetic_new = Vector([0, 0, magnetic.get_vector()[2] + k], color = BLUE)
            magnetic_label_new = magnetic_label.copy().next_to(magnetic_new, UP)
            label_animation = Transform(magnetic_label, magnetic_label_new, run_time = 0.15)
            if not label_added:
                label_animation = Write(magnetic_label, run_time = 0.15)
                label_added = True
            self.play(Transform(magnetic, magnetic_new, run_time = 0.15), label_animation, Uncreate(tiny_stripe, run_time = 0.15))
        self.wait(1)
        self.play(Uncreate(magnetic), Unwrite(magnetic_label))  