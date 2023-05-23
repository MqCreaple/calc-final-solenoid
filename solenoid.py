from manim import *
import numpy as np

class BiotSavart(ThreeDScene):
    def rotate_to_face_camera(self, object: Mobject):
        return object.rotate(self.camera.get_phi(), axis = RIGHT).rotate(90 * DEGREES + self.camera.get_theta(), axis = OUT)

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