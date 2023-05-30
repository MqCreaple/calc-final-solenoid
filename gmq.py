from manim import *
import numpy as np
import helper
from collections.abc import *

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

solenoid_radius = 1.5
class SolenoidSlice(ThreeDScene):
    """
    Demonstrate the two different methods to integrate along a solenoid.
    """
    def face_camera(self, object: Mobject):
        return helper.face_camera(object, self.camera.get_theta(), self.camera.get_phi())

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
        dot = Dot3D(point = axes.coords_to_point(0.3, 0, 0), color = RED)
        dot_label = self.face_camera(MathTex(r'O', color = RED)).next_to(dot, IN)
        magnetic = Vector([0, 0, 0], color = BLUE)
        magnetic_label = self.face_camera(MathTex(r'\vec B', color = BLUE)).next_to(magnetic, UP)
        self.play(Create(dot), Write(dot_label))
        self.wait(1)

        k = 0.5
        label_added = False
        # animation of sum from u = 10 to 30
        magnetic = Vector([0, 0, 0], color = BLUE).shift(dot.get_center())
        magnetic_label = self.face_camera(MathTex(r'\vec B', color = BLUE)).next_to(magnetic, UP)
        k = 1.2
        circle = Circle(radius = solenoid_radius, color = BLUE).shift([0, 0, -4])
        circle.generate_target()
        circle.target.shift([0, 0, 8])
        lower_val = k * (-4) / np.sqrt(solenoid_radius ** 2 + 16)
        upper_val = k * 4 / np.sqrt(solenoid_radius ** 2 + 16)
        magnetic.generate_target()
        magnetic.target = Vector([0, 0, upper_val - lower_val], color = BLUE).shift(dot.get_center())
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
        magnetic = Vector([0, 0, 0], color = BLUE).shift(dot.get_center())
        magnetic_label = self.face_camera(MathTex(r'\vec B', color = BLUE)).next_to(magnetic, UP)
        line = Line(start = [solenoid_radius, 0, -4], end = [solenoid_radius, 0, 4], color = BLUE)
        magnetic.generate_target()
        magnetic.target = Vector([0, 0, upper_val - lower_val], color = BLUE).shift(dot.get_center())
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
        angle = -PI / 6
        x = solenoid_radius * np.cos(angle)
        y = solenoid_radius * np.sin(angle)
        line_pile: Sequence[Line] = []
        animations = []
        for v in range(len(surface.v_curves)):
            animations.append(Uncreate(surface.v_curves[v]))
        self.play(*animations)
        animations = []
        for u in range(len(surface.u_curves)):
            height = surface.get_u_from_index(u)
            segment = Line([0, 0, 0], np.array([y, -x, 0]) * 0.1 / solenoid_radius, color=YELLOW).move_to([x, y, height])
            line_pile.append(segment)
            # create a cross symbol that will faces the camera after rotation
            animations.append(Transform(surface.u_curves[u], segment))
        self.play(*animations)
        self.wait(2)

        ## script: For convenience, let's denote the direction from observation point to the wire pile as $\alpha$.
        ## script: Also, let's shift the observation point to the origin.
        # animate a line to indicate the radius of solenoid
        alpha_vec = Arrow3D(start = dot.get_center(), end = [x, y, 0], color = RED)
        alpha_vec_label = self.face_camera(
            MathTex(r'\vec\alpha=\begin{bmatrix}x\\y\\0\end{bmatrix}', color = RED)
        ).next_to(alpha_vec, LEFT + UP)
        self.play(Create(alpha_vec), Write(alpha_vec_label))
        self.wait(2)
        dot.generate_target()
        dot.target.move_to([0, 0, 0])
        dot_label.generate_target()
        dot_label.target.next_to(dot.target, IN)
        alpha_vec.generate_target()
        alpha_vec.target.shift(dot.target.get_center() - dot.get_center())
        alpha_vec_label.generate_target()
        alpha_vec_label.target.next_to(alpha_vec.target, LEFT + UP)
        for line in surface.u_curves:
            line.generate_target()
            line.target.shift(dot.target.get_center() - dot.get_center())
        self.play(
            MoveToTarget(dot), MoveToTarget(dot_label),
            MoveToTarget(alpha_vec), MoveToTarget(alpha_vec_label),
            *[MoveToTarget(line) for line in surface.u_curves]
        )
        self.wait(2)

        self.play(Uncreate(alpha_vec), Unwrite(alpha_vec_label))
        self.wait(1)

        # move camera to face the line pile
        cam_phi, cam_theta, _, _, _ = self.camera.get_value_trackers()
        dot_label.generate_target()
        dot_label.target = MathTex(r'O', color = RED).rotate(angle).rotate(PI / 2, axis = np.array([x, y, 0])).next_to(dot, IN)
        self.play(
            cam_phi.animate.set_value(90 * DEGREES),
            cam_theta.animate.set_value(-90 * DEGREES + angle),
            MoveToTarget(dot_label)
        )

        cross_symbol_pile: Sequence[VGroup] = []
        for line in surface.u_curves:
            cross_symbol_pile.append(
                self.face_camera(helper.current_in_symbol(line.get_center(), 0.05, color = YELLOW, stroke_width = 1))
            )
        animations = []
        # animation of changing every short line to a cross symbol
        for i in range(len(line_pile)):
            animations.append(Transform(surface.u_curves[i], cross_symbol_pile[i]))
        self.play(*animations)
        self.wait(2)

        # demonstrate the cross product of $Idl$ and $r$
        line_pile_index = 25
        # TODO

        self.play(FadeOut(dot_label), FadeOut(dot), FadeOut(axes), *[FadeOut(symbol) for symbol in surface.u_curves])

class IntegrateSlice(Scene):
    def cross_symbol(self, x: float, y: float, size = 0.2, color: str = WHITE):
        line1 = Line(start = [x - size, y - size, 0], end = [x + size, y + size, 0], color = color, stroke_width = 15 * size)
        line2 = Line(start = [x - size, y + size, 0], end = [x + size, y - size, 0], color = color, stroke_width = 15 * size)
        circle = Circle(radius = size * 1.25, color = color, stroke_width = 15 * size).shift([x, y, 0])
        return VGroup(line1, line2, circle)
    
    def construct(self):
        axes = Axes()
        axes_labels = Axes.get_axis_labels(axes, x_label = r"\alpha", y_label = r"z")
        cross_symbol_pile = []
        for u in np.linspace(-4, 4, 40):
            cross_symbol_pile.append(self.cross_symbol(solenoid_radius, u, size = 0.05, color = YELLOW))
        observe_point = Dot([0, 0, 0], color = RED)
        observe_point_label = MathTex(r'O', color = RED).next_to(observe_point, DOWN)
        self.play(
            FadeIn(axes), FadeIn(axes_labels),
            FadeIn(observe_point), FadeIn(observe_point_label),
            *[FadeIn(symbol) for symbol in cross_symbol_pile]
        )
        self.wait(1)

        ## script: Recall the Biot-Savart law:
        biot_savart = MathTex(
            r'\mathrm d\vec B',
            r' = \frac{\mu_0}{4\pi}',
            r'{I', r'\mathrm d\vec l', r'\times', r'\hat r',
            r'\over',
            r'r^2}'
        )
        biot_savart[0].set_color(BLUE)
        biot_savart[2].set_color(YELLOW)
        biot_savart[5].set_color(RED); biot_savart[7].set_color(RED)
        biot_savart.to_corner(UL)
        self.play(Write(biot_savart))
        self.wait(1)
        
        ## script: But remember, the pile of wire is infinitely dense. Therefore, we should replace the current $I$
        self.play(FocusOn(biot_savart[2]))
        self.wait(2)
        ## script: with the current density $\lambda_I\mathrm dz$.
        biot_savart.generate_target()
        biot_savart.target = MathTex(
            r'\mathrm d\vec B',
            r' = \frac{\mu_0}{4\pi}',
            r'{\lambda_I\mathrm dz', r'\mathrm d\vec l', r'\times', r'\hat r',
            r'\over',
            r'r^2}'
        )
        biot_savart.target[0].set_color(BLUE)
        biot_savart.target[2].set_color(YELLOW)
        biot_savart.target[5].set_color(RED); biot_savart.target[7].set_color(RED)
        biot_savart.target.to_corner(UL)
        self.play(MoveToTarget(biot_savart))
        self.wait(3)

        # demonstrate the current density
        brace = BraceBetweenPoints([solenoid_radius, 0, 0], [solenoid_radius, 1.0, 0])
        brace_label_1 = DecimalNumber(1.0, num_decimal_places = 1, color = YELLOW).next_to(brace, RIGHT)
        brace_label_2 = MathTex(r'\lambda_I', color = YELLOW).next_to(brace_label_1, RIGHT)
        self.play(GrowFromCenter(brace), Write(brace_label_1), Write(brace_label_2))
        def brace_1_updater(b):
            b.next_to(brace, RIGHT)
            b.set_value(np.linalg.norm(brace.get_top() - brace.get_bottom()))
        brace_label_1.add_updater(brace_1_updater)
        brace_label_2.add_updater(lambda m: m.next_to(brace_label_1, RIGHT))
        brace.generate_target()
        brace.target = BraceBetweenPoints([solenoid_radius, 0, 0], [solenoid_radius, 2.0, 0])
        self.play(MoveToTarget(brace))
        brace.generate_target()
        brace.target = BraceBetweenPoints([solenoid_radius, 0, 0], [solenoid_radius, 3.0, 0])
        self.play(MoveToTarget(brace))
        self.wait(1)
        brace_label_1.remove_updater(brace_1_updater)
        self.play(Uncreate(brace), Unwrite(brace_label_1), Unwrite(brace_label_2))
        self.wait(3)

        ## We can decompose the magnetic field into two components:
        r_vec = Vector([-solenoid_radius, 1.25, 0], color = RED).shift([solenoid_radius, -1.25, 0])
        r_vec_label = MathTex(r'\vec r', color = RED).next_to(r_vec, RIGHT)
        r_vec_label.add_updater(lambda m: m.next_to(r_vec, RIGHT))
        self.play(Create(r_vec), Write(r_vec_label))
        k = 3.5
        magnetic = Vector(k * np.cross(np.array([0, 0, -1]), r_vec.get_vector()) / np.linalg.norm(r_vec.get_vector()) ** 3, color = BLUE)
        magnetic.add_updater(lambda v: v.become(Vector(k * np.cross(np.array([0, 0, -1]), r_vec.get_vector()) / np.linalg.norm(r_vec.get_vector()) ** 3, color = BLUE)))
        magnetic_label = MathTex(r'\mathrm d\vec B', color = BLUE).next_to(magnetic, UP)
        magnetic_label.add_updater(lambda m: m.next_to(magnetic, UP))
        magnetic_z = Line([0, 0, 0], [0, magnetic.get_vector()[1], 0], color = BLUE)
        magnetic_z.add_updater(lambda v: v.become(Line([0, 0, 0], [0, magnetic.get_vector()[1], 0], color = BLUE)))
        magnetic_z_label = MathTex(r'\mathrm dB_z', color = BLUE).next_to(magnetic_z, LEFT)
        magnetic_z_label.add_updater(lambda m: m.next_to(magnetic_z, LEFT))
        self.play(Create(magnetic), Write(magnetic_label))
        self.wait(1)
        self.play(Create(magnetic_z), Write(magnetic_z_label))

        r_vec.generate_target()
        r_vec.target = Vector([-solenoid_radius, 0, 0], color = RED).shift([solenoid_radius, 0, 0])
        self.play(MoveToTarget(r_vec))

        r_vec.generate_target()
        r_vec.target = Vector([-solenoid_radius, -1.25, 0], color = RED).shift([solenoid_radius, 1.25, 0])
        self.remove(magnetic_z_label)
        magnetic_z_label = MathTex(r'\mathrm dB_z', color = BLUE).next_to(magnetic_z, RIGHT)
        magnetic_z_label.add_updater(lambda m: m.next_to(magnetic_z, RIGHT))
        self.add(magnetic_z_label)
        self.play(MoveToTarget(r_vec))
        self.wait(3)
        
        magnetic_z_label.clear_updaters()
        magnetic_z_label.generate_target()
        magnetic_z_label.target = MathTex(r'\mathrm dB_z', color = BLUE).next_to(magnetic_z, LEFT)
        self.play(MoveToTarget(magnetic_z_label), Uncreate(magnetic), Uncreate(magnetic_label))

        ## script: Let's first focus on the $\alpha$ component of magnetic field.
        # TODO: prove that the alpha component is zero

        ## script: Now, let's focus on the $z$ component of magnetic field.
        angle = Angle(axes.x_axis, Line(r_vec.get_end(), r_vec.get_start()), radius = 0.5)
        angle_label = MathTex(r'\theta').next_to(angle, RIGHT).shift(0.2 * UP)
        self.play(Create(angle), Write(angle_label))
        self.wait(1)
        biot_savart_target = MathTex(
            r'\mathrm dB_z',
            r' = \frac{\mu_0}{4\pi}',
            r'{\lambda_I\mathrm dz', r'\mathrm dl',
            r'\over',
            r'r^2}', r'\cdot\cos\theta'
        ).to_corner(UL)
        biot_savart_target[0].set_color(BLUE)
        biot_savart_target[2].set_color(YELLOW)
        biot_savart_target[5].set_color(RED)
        self.play(TransformMatchingTex(biot_savart, biot_savart_target))
        self.wait(2)

        r_vec_label.clear_updaters()
        r_vec_label.generate_target()
        r_vec_label.target = MathTex(r'\vec r', color = RED).next_to(r_vec, UP)
        self.play(Uncreate(magnetic_z), Uncreate(magnetic_z_label), MoveToTarget(r_vec_label))
        self.wait(3)

        ## script: Remember: the rdistance from wire pile to the observation point $\alpha$ and the vertical distance is $z$.
        alpha_brace = BraceBetweenPoints([0, 0, 0], [solenoid_radius, 0, 0], DOWN)
        alpha_brace_label = MathTex(r'\alpha').next_to(alpha_brace, DOWN)
        self.play(Create(alpha_brace), Write(alpha_brace_label))
        z_brace = BraceBetweenPoints([solenoid_radius, 0, 0], r_vec.get_start(), RIGHT)
        z_brace_label = MathTex(r'z').next_to(z_brace, RIGHT)
        self.play(Create(z_brace), Write(z_brace_label))
        self.wait(1)

        ## script: Replace $\cos\theta$ with $\frac{\alpha}{\sqrt{\alpha^2 + z^2}}$
        r_equation = MathTex(r'r', r' = \sqrt{\alpha^2 + z^2}').next_to(biot_savart_target, DOWN)
        r_equation[0].set_color(RED)
        cos_equation = MathTex(r'\cos\theta', r' = \frac{\alpha}{\sqrt{\alpha^2 + z^2}}').next_to(r_equation, DOWN)
        self.play(Write(r_equation))
        self.play(Write(cos_equation))
        self.wait(3)
        formula = MathTex(
            r'\mathrm dB_z',
            r' = \frac{\mu_0}{4\pi}\frac{\lambda_I\frac{\alpha}{\sqrt{\alpha^2+z^2}}}{\alpha^2+z^2}\mathrm dz\mathrm dl',
        ).to_corner(UL)
        formula[0].set_color(BLUE)
        self.play(
            ReplacementTransform(VGroup(biot_savart_target, r_equation, cos_equation), formula),
            Uncreate(angle), Unwrite(angle_label),
        )
        self.wait(2)
        formula.generate_target()
        formula.target = MathTex(
            r'\mathrm dB_z',
            r'=', r'\frac{\mu_0}{4\pi}\frac{\lambda_I\alpha}{(\alpha^2+z^2)^{\frac 32}}',
            r'\mathrm dz', r'\mathrm dl'
        ).to_corner(UL)
        formula.target[0].set_color(BLUE)
        self.play(MoveToTarget(formula))
        self.wait(4)
        self.play(
            Uncreate(alpha_brace), Unwrite(alpha_brace_label),
            Uncreate(z_brace), Unwrite(z_brace_label),
            Uncreate(r_vec), Uncreate(r_vec_label),
            Uncreate(observe_point), Unwrite(observe_point_label),
            *[Uncreate(symbol) for symbol in cross_symbol_pile],
            Uncreate(axes), Uncreate(axes_labels),
        )
        formula.generate_target()
        formula.target.move_to([0, 0, 0])
        self.play(MoveToTarget(formula))
        self.wait(2)

        ## script: Now let's rewrite it in integral form:
        formula_target = MathTex(
            r'B_z', r'=',
            r'\oint_{\text{circle}}',
            r'\int_{-\infty}^{+\infty}',
            r'\frac{\mu_0}{4\pi}\frac{\lambda_I\alpha}{(\alpha^2+z^2)^{\frac 32}}',
            r'\mathrm dz', r'\mathrm dl',
        )
        formula_target[0].set_color(BLUE)
        formula_target[2].set_color(ORANGE)
        formula_target[3].set_color(YELLOW)
        formula_target[5].set_color(YELLOW)
        formula_target[6].set_color(ORANGE)
        self.play(TransformMatchingShapes(formula, formula_target))
        formula = formula_target
        self.wait(2)

        ## script: Now, let's integrate the vertical component first.
        formula_target = MathTex(
            r'\int_{-\infty}^{+\infty}',
            r'\frac{\mu_0}{4\pi}\frac{\lambda_I\alpha}{(\alpha^2+z^2)^{\frac 32}}',
            r'\mathrm dz'
        )
        formula_target[0].set_color(YELLOW)
        formula_target[2].set_color(YELLOW)
        self.play(TransformMatchingTex(formula, formula_target))
        formula = formula_target
        self.wait(4)

        ## script: To solve this, first pull out all the constants:
        formula_target = MathTex(
            r'\frac{\mu_0}{4\pi}\lambda_I\alpha',
            r'\int_{-\infty}^{+\infty}',
            r'\frac{1}{(\alpha^2+z^2)^{\frac 32}}',
            r'\mathrm dz'
        )
        formula_target[1].set_color(YELLOW)
        formula_target[3].set_color(YELLOW)
        self.play(TransformMatchingTex(formula, formula_target))
        formula = formula_target
        self.wait(3)

        ## script: Note that the core of this integral is the $(\alpha^2 + z^2)^{3/2}$ piece:
        self.play(FocusOn(formula_target[2]))
        self.wait(5)

        ## script: Since $z$ is the integral variable and $\alpha$ is a constant, we can use a trig substitution
        ## script: for expression of the form $\alpha^2 + z^2$.
        z_sub = MathTex(r'z', r'=', r'\alpha\tan\theta').next_to(formula, DOWN)
        self.play(Write(z_sub))
        self.wait(2)

        formula_target = MathTex(
            r'\frac{\mu_0}{4\pi}\lambda_I\alpha',
            r'\int_{-\infty}^{+\infty}',
            r'{1', r'\over', r'[\alpha^2(1 + \tan^2\theta)]^{\frac 32}}',
            r'\mathrm d(\alpha\tan\theta)'
        )
        formula_target[1].set_color(YELLOW)
        formula_target[5].set_color(YELLOW)
        self.play(Transform(formula, formula_target), Unwrite(z_sub))
        self.wait(2)

        ## script: Now, let's differentiate the $\alpha\tan\theta$. Remember to change the bound of integration.
        formula_target = MathTex(
            r'\frac{\mu_0}{4\pi}\lambda_I\alpha',
            r'\int_{-\frac\pi 2}^{+\frac\pi 2}',
            r'{1', r'\over', r'[\alpha^2(1 + \tan^2\theta)]^{\frac 32}}',
            r'\alpha\sec^2\theta',
            r'\mathrm d\theta'
        )
        formula_target[1].set_color(YELLOW)
        formula_target[6].set_color(YELLOW)
        self.play(
            ReplacementTransform(formula[0], formula_target[0]),
            ReplacementTransform(formula[1], formula_target[1]),
            ReplacementTransform(formula[2], formula_target[2]),
            ReplacementTransform(formula[3], formula_target[3]),
            ReplacementTransform(formula[4], formula_target[4]),
            ReplacementTransform(formula[5], formula_target[5:])
        )
        formula = formula_target
        self.wait(3)

        ## script: Now simplify the expression.
        formula_target = MathTex(
            r'\frac{\mu_0}{4\pi}\lambda_I\alpha',
            r'\int_{-\frac\pi 2}^{+\frac\pi 2}',
            r'{\alpha\sec^2\theta', r'\over', r'(\alpha^2\sec^2\theta)^{\frac 32}}}',
            r'\mathrm d\theta',
        )
        formula_target[1].set_color(YELLOW)
        formula_target[5].set_color(YELLOW)
        self.play(
            ReplacementTransform(formula[0], formula_target[0]),
            ReplacementTransform(formula[1], formula_target[1]),
            ReplacementTransform(VGroup(formula[2], formula[5]), formula_target[2]),
            ReplacementTransform(formula[3], formula_target[3]),
            ReplacementTransform(formula[4], formula_target[4]),
            ReplacementTransform(formula[6], formula_target[5]),
        )
        formula = formula_target
        self.wait(4)

        ## script: then simplify the denomenator:
        formula_target = MathTex(
            r'\frac{\mu_0}{4\pi}\lambda_I\alpha',
            r'\int_{-\frac\pi 2}^{+\frac\pi 2}',
            r'{\alpha\sec^2\theta', r'\over', r'\alpha^3\sec^3\theta}}',
            r'\mathrm d\theta',
        )
        formula_target[1].set_color(YELLOW)
        formula_target[5].set_color(YELLOW)
        self.play(
            Transform(formula[0], formula_target[0]),
            Transform(formula[1], formula_target[1]),
            Transform(formula[2], formula_target[2]),
            Transform(formula[3], formula_target[3]),
            Transform(formula[4], formula_target[4]),
            Transform(formula[5], formula_target[5]),
        )
        self.wait(3)