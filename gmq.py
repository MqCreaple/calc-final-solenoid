from manim import *
import numpy as np
import helper
from collections.abc import *

class BiotSavart(ThreeDScene):
    """
    Demonstration of Biot-Savart Law
    """
    def face_camera(self, object: Mobject):
        return helper.face_camera(object, self.camera.get_theta(), self.camera.get_phi())

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
        current_label = self.face_camera(MathTex(r'I\mathrm d\vec l', color = YELLOW)).next_to(current, DOWN)
        observer = Vector(direction = np.array([2, 3, 0]), color = RED)
        magnetic = self.magnetic_vec(current, observer)
        magnetic_label = self.face_camera(MathTex(r'\vec B', color = BLUE)).next_to(magnetic, UP)
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
        pile_azimuth = -PI / 6
        x = solenoid_radius * np.cos(pile_azimuth)
        y = solenoid_radius * np.sin(pile_azimuth)
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
            MathTex(r'\vec\alpha', color = RED)
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

        # rotate the axes to let the current flow along x axis
        line_pile_index = 25
        line_pile_center = surface.u_curves[line_pile_index].get_center()
        line_pile_dir = surface.u_curves[line_pile_index].get_start() - surface.u_curves[line_pile_index].get_end()
        line_pile_angle = np.arctan(line_pile_dir[1] / line_pile_dir[0]) + PI
        self.play(axes.animate.rotate(line_pile_angle))
        self.wait(2)

        # demonstrate the cross product of $Idl$ and $r$
        current = Arrow3D(start = line_pile_center, end = line_pile_center + line_pile_dir * 8, color = YELLOW)
        current_label = self.face_camera(MathTex(r'I\mathrm d\vec l=\begin{bmatrix}I\\0\\0\end{bmatrix}\mathrm dl', color = YELLOW)).next_to(current, DOWN + IN)
        self.play(Create(current), Write(current_label))
        self.wait(1)
        r_vec = Arrow3D(start = line_pile_center, end = [0, 0, 0], color = RED)
        r_vec_label = self.face_camera(MathTex(r'\vec r=\begin{bmatrix}-x\\-y\\-z\end{bmatrix}', color = RED)).next_to(r_vec, LEFT + UP)
        self.play(Create(r_vec), Write(r_vec_label))
        self.wait(3)

        km = 2.0
        magnetic = Arrow3D(start = [0, 0, 0], end = np.cross(current.get_end() - current.get_start(), r_vec.get_end() - r_vec.get_start()) * km, color = BLUE)
        magnetic_label = self.face_camera(MathTex(r'I\mathrm d\vec l\times\vec r', color = BLUE)).next_to(magnetic, UP + IN)
        self.play(Create(magnetic), Write(magnetic_label))
        self.wait(1)

        # write out the formula of  $Idl \times r$
        formula = self.face_camera(MathTex(
            r'&I\mathrm d\vec l', r'\times', r'\vec r\\',
            r'& =', r'\begin{bmatrix}I\\0\\0\end{bmatrix}\mathrm dl', r'\times', r'\begin{bmatrix}-x\\-y\\-z\end{bmatrix}',
            r'\\& =', r'\begin{bmatrix}0\\-Iz\\-Iy\end{bmatrix}', r'\mathrm dl'
        ).shift([3.3, -4, 1.6]))
        formula[0].set_color(YELLOW)
        formula[2].set_color(RED)
        formula[4].set_color(YELLOW)
        formula[6].set_color(RED)
        formula[8].set_color(BLUE)
        self.play(
            ReplacementTransform(current_label.copy(), formula[0]),
            Write(formula[1]),
            ReplacementTransform(r_vec_label.copy(), formula[2]),
        )
        self.wait(2)
        self.play(Write(formula[3]))
        self.play(
            ReplacementTransform(current_label.copy(), formula[4]),
            Write(formula[5]),
            ReplacementTransform(r_vec_label.copy(), formula[6]),
        )
        self.wait(3)
        self.play(Write(formula[7:]))
        self.wait(2)
        self.play(Indicate(formula[8]))
        self.wait(2)
        self.play(Unwrite(formula))
        self.wait(1)

        # move camera around to show the 3D relationships between two vectors
        cam_phi, cam_theta, _, _, _ = self.camera.get_value_trackers()
        new_theta = line_pile_angle - 90 * DEGREES
        new_phi = 75 * DEGREES
        dot_label.add_updater(lambda m: m.become(self.face_camera(MathTex(r'O', color = RED)).next_to(dot, IN)))
        current_label.add_updater(lambda m: m.become(self.face_camera(MathTex(r'I\mathrm d\vec l', color = YELLOW)).next_to(current, DOWN + IN)))
        r_vec_label.add_updater(lambda m: m.become(self.face_camera(MathTex(r'\vec r', color = RED)).next_to(r_vec, LEFT + UP)))
        magnetic_label.add_updater(lambda m: m.become(self.face_camera(MathTex(r'I\mathrm d\vec l\times\vec r', color = BLUE)).next_to(magnetic, UP + IN)))
        self.play(
            cam_phi.animate.set_value(new_phi),
            cam_theta.animate.set_value(new_theta)
        )
        self.wait(2)
        
        new_theta = line_pile_angle
        new_phi = 90 * DEGREES
        self.play(
            cam_phi.animate.set_value(new_phi),
            cam_theta.animate.set_value(new_theta)
        )
        self.wait(2)
        self.play(Unwrite(current_label), Unwrite(r_vec_label), Unwrite(magnetic_label))

        cross_symbol_pile: Sequence[VGroup] = []
        for line in surface.u_curves:
            cross_symbol_pile.append(
                self.face_camera(helper.current_in_symbol(line.get_center(), 0.075, color = YELLOW, stroke_width = 1))
            )
        animations = []
        # animation of changing every short line to a cross symbol
        for i in range(len(line_pile)):
            animations.append(Transform(surface.u_curves[i], cross_symbol_pile[i]))
        self.play(*animations, FadeOut(current))
        self.wait(2)

        print(dot_label in self.mobjects)

        self.play(*[FadeOut(obj) for obj in self.mobjects])

class IntegrateSlice(Scene):
    def cross_symbol(self, x: float, y: float, size = 0.2, color: str = WHITE):
        line1 = Line(start = [x - size, y - size, 0], end = [x + size, y + size, 0], color = color, stroke_width = 15 * size)
        line2 = Line(start = [x - size, y + size, 0], end = [x + size, y - size, 0], color = color, stroke_width = 15 * size)
        circle = Circle(radius = size * 1.25, color = color, stroke_width = 15 * size).shift([x, y, 0])
        return VGroup(line1, line2, circle)
    
    def construct(self):
        axes = Axes()
        axes_labels = Axes.get_axis_labels(axes, x_label = r"y", y_label = r"z")
        cross_symbol_pile = []
        for u in np.linspace(-4, 4, 40):
            cross_symbol_pile.append(helper.current_in_symbol([solenoid_radius, u, 0], 0.075, color = YELLOW, stroke_width = 1))
        observe_point = Dot([0, 0, 0], color = RED)
        observe_point_label = MathTex(r'O', color = RED).next_to(observe_point, DOWN)
        self.play(
            FadeIn(axes), FadeIn(axes_labels),
            FadeIn(observe_point), FadeIn(observe_point_label),
            *[FadeIn(symbol) for symbol in cross_symbol_pile]
        )
        self.wait(1)

        ## script: Recall the Biot-Savart law:
        formula = MathTex(
            r'\mathrm d\vec B',
            r' = \frac{\mu_0}{4\pi}',
            r'{I', r'\mathrm d\vec l', r'\times', r'\vec r',
            r'\over',
            r'r^3}'
        )
        formula[0].set_color(BLUE)
        formula[2].set_color(YELLOW)
        formula[5].set_color(RED); formula[7].set_color(RED)
        formula.to_corner(UL)
        self.play(Write(formula))
        self.wait(1)
        
        ## script: But remember, the pile of wire is infinitely dense. Therefore, we should replace the current $I$
        self.play(FocusOn(formula[2]))
        self.wait(2)
        ## script: with the current density $\lambda_I\mathrm dz$.
        formula.generate_target()
        formula.target = MathTex(
            r'\mathrm d\vec B',
            r' = \frac{\mu_0}{4\pi}',
            r'{\lambda_I\mathrm dz', r'\mathrm d\vec l', r'\times', r'\vec r',
            r'\over',
            r'r^3}'
        ).to_corner(UL)
        formula.target[0].set_color(BLUE)
        formula.target[2].set_color(YELLOW)
        formula.target[5].set_color(RED); formula.target[7].set_color(RED)
        self.play(MoveToTarget(formula))
        self.wait(2)

        brace = BraceBetweenPoints([solenoid_radius, 0, 0], [solenoid_radius, 1, 0])
        brace_label1 = DecimalNumber(1, num_decimal_places = 1).set_color(YELLOW).next_to(brace, RIGHT)
        brace_label2 = MathTex(r'\lambda_I').set_color(YELLOW).next_to(brace_label1, RIGHT)
        self.play(Create(brace), Write(brace_label1), Write(brace_label2))
        self.wait(1)
        brace_label1.add_updater(lambda m: m.next_to(brace, RIGHT).set_value(np.linalg.norm(brace.get_top() - brace.get_bottom())))
        brace_label2.add_updater(lambda m: m.next_to(brace_label1, RIGHT))
        brace.generate_target()
        brace.target = BraceBetweenPoints([solenoid_radius, 0, 0], [solenoid_radius, 2, 0])
        self.play(MoveToTarget(brace))
        self.wait(1)
        brace.generate_target()
        brace.target = BraceBetweenPoints([solenoid_radius, 0, 0], [solenoid_radius, 3, 0])
        self.play(MoveToTarget(brace))
        self.wait(1)
        self.play(Uncreate(brace), Unwrite(brace_label1), Unwrite(brace_label2))
        self.wait(2)

        formula.generate_target()
        formula.target = MathTex(
            r'\mathrm d\vec B',
            r' = \frac{\mu_0}{4\pi}',
            r'{\begin{bmatrix}\lambda_I\\0\\0\end{bmatrix}\mathrm dz', r'\mathrm dl', r'\times', r'\begin{bmatrix}-x\\-y\\-z\end{bmatrix}',
            r'\over',
            r'(x^2+y^2+z^2)^{\frac 32}}'
        ).to_corner(UL)
        formula.target[0].set_color(BLUE)
        formula.target[2].set_color(YELLOW)
        formula.target[5].set_color(RED); formula.target[7].set_color(RED)
        self.play(MoveToTarget(formula))
        self.wait(4)

        formula_target = MathTex(
            r'\mathrm d\vec B',
            r' = \frac{\mu_0}{4\pi}',
            r'{\begin{bmatrix}0\\ \lambda_Iz \\ \lambda_Iy \end{bmatrix}\mathrm dz', r'\mathrm dl',
            r'\over',
            r'(x^2+y^2+z^2)^{\frac 32}}'
        ).to_corner(UL)
        formula_target[0].set_color(BLUE)
        self.play(
            ReplacementTransform(formula[0], formula_target[0]),
            ReplacementTransform(formula[1], formula_target[1]),
            ReplacementTransform(VGroup(formula[2], formula[4:6]), formula_target[2]),
            ReplacementTransform(formula[3], formula_target[3]),
            ReplacementTransform(formula[6], formula_target[4]),
            ReplacementTransform(formula[7], formula_target[5]),
        )
        formula = formula_target
        self.wait(4)

        formula_y = MathTex(
            r'\mathrm dB_y',
            r'=\frac{\mu_0}{4\pi}', r'{\lambda_Iz', r'\over', r'(x^2+y^2+z^2)^{\frac 32}}',
            r'\mathrm dz', r'\mathrm dl',
        ).to_corner(UL)
        formula_z = MathTex(
            r'\mathrm dB_z',
            r'=\frac{\mu_0}{4\pi}', r'{\lambda_Iy', r'\over', r'(x^2+y^2+z^2)^{\frac 32}}',
            r'\mathrm dz', r'\mathrm dl'
        ).next_to(formula_y, DOWN)
        formula_y[0].set_color(BLUE)
        formula_z[0].set_color(BLUE)
        self.play(
            ReplacementTransform(formula[0], VGroup(formula_y[0], formula_z[0])),
            ReplacementTransform(formula[1], VGroup(formula_y[1], formula_z[1])),
            ReplacementTransform(formula[2], VGroup(formula_y[2], formula_y[5], formula_z[2], formula_z[5])),
            ReplacementTransform(formula[4], VGroup(formula_y[3], formula_z[3])),
            ReplacementTransform(formula[3], VGroup(formula_y[6], formula_z[6])),
            ReplacementTransform(formula[5], VGroup(formula_y[4], formula_z[4]))
        )
        self.wait(2)
        self.play(Uncreate(formula_y), formula_z.animate.to_corner(UL))
        formula = formula_z
        self.wait(3)
        self.play(
            #* Uncreate every graphics object
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
            r'\frac{\mu_0}{4\pi}\frac{\lambda_I y}{(x^2+y^2+z^2)^{\frac 32}}',
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
            r'\frac{\mu_0}{4\pi}\frac{\lambda_I y}{(x^2+y^2+z^2)^{\frac 32}}',
            r'\mathrm dz'
        )
        formula_target[0].set_color(YELLOW)
        formula_target[2].set_color(YELLOW)
        self.play(TransformMatchingTex(formula, formula_target))
        formula = formula_target
        self.wait(4)

        ## script: To solve this, first pull out all the constants:
        formula_target = MathTex(
            r'\frac{\mu_0}{4\pi}\lambda_I y',
            r'\int_{-\infty}^{+\infty}',
            r'\frac{1}{(x^2+y^2+z^2)^{\frac 32}}',
            r'\mathrm dz'
        )
        formula_target[1].set_color(YELLOW)
        formula_target[3].set_color(YELLOW)
        self.play(TransformMatchingTex(formula, formula_target))
        formula = formula_target
        self.wait(3)

        ## script: Note that the core of this integral is the $(x^2 + y^2 + z^2)^{3/2}$ piece:
        self.play(FocusOn(formula_target[2]))
        self.wait(5)

        ## script: For simplicity, let's denote $x^2+y^2$ as $\alpha^2$.
        ## script: Since $z$ is the integral variable and $\alpha$ is a constant, we can use a trig substitution
        ## script: for expression of the form $\alpha^2 + z^2$.
        alpha_sub = MathTex(r'\alpha^2', r'=', r'x^2+y^2').next_to(formula, DOWN)
        self.play(Write(alpha_sub))
        self.wait(2)
        formula_target = MathTex(
            r'\frac{\mu_0}{4\pi}\lambda_I y',
            r'\int_{-\infty}^{+\infty}',
            r'\frac{1}{(\alpha^2+z^2)^{\frac 32}}',
            r'\mathrm dz'
        )
        formula_target[1].set_color(YELLOW)
        formula_target[3].set_color(YELLOW)
        self.play(Transform(formula, formula_target))
        self.wait(2)

        z_sub = MathTex(r'z', r'=', r'\alpha\tan\theta').next_to(alpha_sub, DOWN)
        self.play(Write(z_sub))
        self.wait(2)

        formula_target = MathTex(
            r'\frac{\mu_0}{4\pi}\lambda_I y',
            r'\int_{-\infty}^{+\infty}',
            r'{1', r'\over', r'[\alpha^2(1 + \tan^2\theta)]^{\frac 32}}',
            r'\mathrm d(\alpha\tan\theta)'
        )
        formula_target[1].set_color(YELLOW)
        formula_target[5].set_color(YELLOW)
        self.play(Transform(formula, formula_target), Unwrite(z_sub), Unwrite(alpha_sub))
        self.wait(2)

        ## script: Now, let's differentiate the $\alpha\tan\theta$. Remember to change the bound of integration.
        formula_target = MathTex(
            r'\frac{\mu_0}{4\pi}\lambda_I y',
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
            r'\frac{\mu_0}{4\pi}\lambda_I y',
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
            r'\frac{\mu_0}{4\pi}\lambda_I y',
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

        formula_target = MathTex(
            r'\frac{\mu_0}{4\pi}\frac{\lambda_I y}{\alpha^2}',
            r'\int_{-\frac\pi 2}^{+\frac\pi 2}',
            r'{1', r'\over', r'\sec\theta}',
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
        formula_target = MathTex(
            r'\frac{\mu_0}{4\pi}\frac{\lambda_I y}{x^2+y^2}',
            r'\int_{-\frac\pi 2}^{+\frac\pi 2}',
            r'\cos\theta',
            r'\mathrm d\theta',
        )
        formula_target[1].set_color(YELLOW)
        formula_target[3].set_color(YELLOW)
        self.play(
            ReplacementTransform(formula[0], formula_target[0]),
            ReplacementTransform(formula[1], formula_target[1]),
            ReplacementTransform(formula[2:5], formula_target[2]),
            ReplacementTransform(formula[5], formula_target[3]),
        )
        formula = formula_target
        self.wait(3)

        ## script: Integrating $\cos\theta$ is straightforward:
        formula_target = MathTex(
            r'\frac{\mu_0}{4\pi}\frac{\lambda_I y}{x^2+y^2}',
            r'\sin\theta',
            r'\Big |_{-\frac\pi 2}^{+\frac\pi 2}',
        )
        formula_target[2].set_color(YELLOW)
        self.play(
            ReplacementTransform(formula[0], formula_target[0]),
            ReplacementTransform(formula[2], formula_target[1]),
            ReplacementTransform(VGroup(formula[1], formula[3]), formula_target[2]),
        )
        formula = formula_target
        self.wait(3)
        formula_target = MathTex(
            r'\frac{\mu_0}{4\pi}\frac{\lambda_I y}{x^2+y^2}',
            r'\cdot 2'
        )
        self.play(
            ReplacementTransform(formula[0], formula_target[0]),
            ReplacementTransform(formula[1:3], formula_target[1]),
        )
        formula = formula_target
        self.wait(3)
        formula_target = MathTex(r'\frac{\mu_0}{2\pi}\frac{\lambda_I y}{x^2+y^2}')
        self.play(Transform(formula, formula_target))
        self.wait(3)

        formula_target_1 = MathTex(
            r'& \frac{\mathrm dB_z}{\mathrm dl}\\',
            r'=& \int_{-\infty}^{\infty}\frac{\mu_0}{4\pi}\frac{\lambda_I y}{(x^2+y^2+z^2)^{\frac 32}}\mathrm dz\\',
            r'=& \frac{\mu_0}{4\pi}\lambda_Iy \int_{-\infty}^{\infty}\frac{1}{(x^2+y^2+z^2)^{\frac 32}}\mathrm dz\\',
            r'\stackrel{\alpha^2=x^2+y^2}{=}& \frac{\mu_0}{2\pi}\lambda_Iy \int_{-\infty}^{\infty}\frac{1}{[\alpha^2(1+\tan^2\theta)]^{\frac 32}}\mathrm d(\alpha\tan\theta)\\',
            r'=& \frac{\mu_0}{4\pi}\lambda_Iy \int_{-frac{\pi}{2}}^{+\frac{\pi}{2}}\frac{\alpha\sec^2\theta}{(\alpha^2\sec^2\theta)^{\frac 32}}\mathrm d\theta\\',
        ).scale(0.75).to_edge(LEFT)
        formula_target_2 = MathTex(
            r'=& \frac{\mu_0}{4\pi}\lambda_Iy \int_{-frac{\pi}{2}}^{+\frac{\pi}{2}}\frac{\alpha\sec^2\theta}{\alpha^3\sec^3\theta}\mathrm d\theta\\',
            r'=& \frac{\mu_0}{4\pi}\frac{\lambda_Iy}{\alpha^2} \int_{-\frac{\pi}{2}}^{+\frac{\pi}{2}}\frac{1}{\sec\theta}\mathrm d\theta\\',
            r'=& \frac{\mu_0}{4\pi}\frac{\lambda_Iy}{x^2+y^2} \int_{-\frac{\pi}{2}}^{+\frac{\pi}{2}}\cos\theta\mathrm d\theta\\',
            r'=& \frac{\mu_0}{4\pi}\frac{\lambda_Iy}{x^2+y^2} \sin\theta\Big |_{-\frac{\pi}{2}}^{+\frac{\pi}{2}}\\',
            r'=&', r'\frac{\mu_0}{2\pi}\frac{\lambda_Iy}{x^2+y^2}\\',
        ).scale(0.75).to_edge(RIGHT)
        self.play(Write(formula_target_1), Write(formula_target_2[:-1]), TransformMatchingShapes(formula, formula_target_2[-1]), run_time = 3)
        formula = formula_target
        self.wait(3)
        self.play(Unwrite(formula_target_1), Unwrite(formula_target_2))

class Final(Scene):
    def construct(self):
        prompt = Text("Solve: ").set_color(YELLOW)
        prompt_formula = MathTex(
            r'\int_0^{2\pi}',
            r'{1-\beta', r'\cos\theta', r'\over', r'\beta^2-2\beta', r'\cos\theta', r'+1}',
            r'\mathrm d\theta'
        )
        VGroup(prompt, prompt_formula).arrange(RIGHT)
        self.play(Write(prompt), Write(prompt_formula))
        self.play(VGroup(prompt, prompt_formula).animate.to_edge(UP))
        self.play(prompt_formula.animate.set_color(YELLOW))
        self.wait(2)
        self.play(Indicate(prompt_formula[2], color = RED), Indicate(prompt_formula[5], color = RED))
        self.wait(2)

        formula = MathTex(
            r'{\frac 12(\beta^2-2\beta\cos\theta+1)', r'+', r'\frac{1-\beta^2}{2}', r'\over', r'\beta^2-2\beta', r'\cos\theta', r'+1}'
        )
        self.play(
            ReplacementTransform(prompt_formula[1:3].copy(), formula[0:3]),
            ReplacementTransform(prompt_formula[3].copy(), formula[3]),
            ReplacementTransform(prompt_formula[4].copy(), formula[4]),
            ReplacementTransform(prompt_formula[5].copy(), formula[5]),
            ReplacementTransform(prompt_formula[6].copy(), formula[6]),
        )
        self.wait(2)
        formula_target = MathTex(
            r'\int_0^{2\pi}',
            r'\frac 12', r'+', r'{\frac{1-\beta^2}{2}', r'\over', r'\beta^2-2\beta', r'\cos\theta', r'+1}',
            r'\mathrm d\theta'
        )
        self.play(
            ReplacementTransform(formula[0], formula_target[1]),
            ReplacementTransform(formula[1], formula_target[2]),
            ReplacementTransform(formula[2], formula_target[3]),
            ReplacementTransform(formula[3], formula_target[4]),
            ReplacementTransform(formula[4], formula_target[5]),
            ReplacementTransform(formula[5], formula_target[6]),
            ReplacementTransform(formula[6], formula_target[7]),
        )
        self.wait(2)
        self.play(
            ReplacementTransform(prompt_formula[0].copy(), formula_target[0]),
            ReplacementTransform(prompt_formula[7].copy(), formula_target[8]),
        )
        formula = formula_target
        self.wait(2)

        formula_target = MathTex(
            r'\frac 12\cdot 2\pi+',
            r'\int_0^{2\pi}',
            r'\frac{1-\beta^2}{2}',
            r'{1', r'\over', r'\beta^2-2\beta\cos\theta+1}',
            r'\mathrm d\theta'
        )
        self.play(
            ReplacementTransform(formula[1:3], formula_target[0]),
            ReplacementTransform(formula[0], formula_target[1]),
            ReplacementTransform(formula[3], formula_target[2]),
            ReplacementTransform(formula[4], formula_target[3:5]),
            TransformMatchingShapes(formula[5:8], formula_target[5]),
            ReplacementTransform(formula[8], formula_target[6]),
        )
        formula = formula_target
        self.wait(2)

        formula_target = MathTex(
            r'\pi+',
            r'\frac{1-\beta^2}{2}',
            r'\int_0^{2\pi}',
            r'{1', r'\over', r'\beta^2-2\beta', r'\cos\theta', r'+1}',
            r'\mathrm d\theta'
        )
        self.play(TransformMatchingTex(formula, formula_target))
        formula = formula_target
        self.wait(2)

        prompt.generate_target()
        prompt_formula.generate_target()
        prompt_formula.target = formula.copy()
        prompt_formula.target[2:].set_color(YELLOW)
        VGroup(prompt_formula.target[0:2], prompt.target, prompt_formula.target[2:]).arrange(RIGHT).to_edge(UP)
        self.play(MoveToTarget(prompt), FadeOut(prompt_formula), ReplacementTransform(formula.copy(), prompt_formula.target))
        prompt_formula = prompt_formula.target
        self.wait(2)

        formula_target = formula[2:].copy().center()
        self.play(Unwrite(formula[0:2]), ReplacementTransform(formula[2:], formula_target))
        formula = formula_target
        self.wait(5)

        cos_to_tan_formula = MathTex(
            r'\cos\theta', r'=',
            r'{2', r'\over', r'1+\tan^2{\frac{\theta}{2}}}', r'-', r'1'
        ).next_to(formula, DOWN)
        cos_to_tan_formula[0].set_color(RED)
        self.play(Write(cos_to_tan_formula), formula[4].animate.set_color(RED))
        self.wait(2)
        formula_target = MathTex(
            r'\int_0^{2\pi}',
            r'{1', r'\over', r'\beta^2', r'-2\beta', r'\left(\frac{2}{1+\tan^2\frac{\theta}{2}}-1\right)', r'+1}',
            r'\mathrm d\theta'
        )
        self.play(Transform(formula, formula_target))
        self.wait(1)

        self.play(Unwrite(cos_to_tan_formula))
        formula_target = MathTex(
            r'\int_0^{2\pi}',
            r'{1', r'\over', r'\beta^2', r'-\frac{4\beta}{1+\tan^2\frac{\theta}{2}}', r'+2\beta', r'+1}',
            r'\mathrm d\theta'
        )
        self.wait(1)
        self.play(
            ReplacementTransform(formula[0], formula_target[0]),
            ReplacementTransform(formula[1], formula_target[1]),
            ReplacementTransform(formula[2], formula_target[2]),
            ReplacementTransform(formula[3], formula_target[3]),
            ReplacementTransform(formula[4:6], formula_target[4:6]),
            ReplacementTransform(formula[6], formula_target[6]),
            ReplacementTransform(formula[7], formula_target[7]),
        )
        formula = formula_target
        self.wait(2)

        formula_target = MathTex(
            r'\int_0^{2\pi}',
            r'{1', r'\over', r'(\beta^2+2\beta+1)', r'-\frac{4\beta}{1+\tan^2\frac{\theta}{2}}}',
            r'\mathrm d\theta'
        )
        self.play(
            ReplacementTransform(formula[0], formula_target[0]),
            ReplacementTransform(formula[1], formula_target[1]),
            ReplacementTransform(formula[2], formula_target[2]),
            TransformMatchingShapes(VGroup(formula[3], formula[5:7]), formula_target[3]),
            ReplacementTransform(formula[4], formula_target[4]),
            ReplacementTransform(formula[7], formula_target[5]),
        )
        formula = formula_target
        self.wait(2)

        formula_target = MathTex(
            r'\int_0^{2\pi}',
            r'{1', r'\over', r'(\beta+1)^2', r'-\frac{4\beta}{1+\tan^2\frac{\theta}{2}}}',
            r'\mathrm d\theta'
        )
        self.play(Transform(formula, formula_target))
        self.wait(2)

        formula_target = MathTex(
            r'\int_0^{2\pi}',
            r'{1+\tan^2\frac{\theta}{2}', r'\over', r'(\beta+1)^2', r'\left(1', r'+\tan^2\frac{\theta}{2}\right)', r'-4\beta}',
            r'\mathrm d\theta'
        )
        self.play(
            ReplacementTransform(formula[0], formula_target[0]),
            ReplacementTransform(formula[1], formula_target[1]),
            ReplacementTransform(formula[2], formula_target[2]),
            ReplacementTransform(formula[3], formula_target[3]),
            FadeIn(formula_target[4:6]),
            ReplacementTransform(formula[4], formula_target[6]),
            ReplacementTransform(formula[5], formula_target[7]),
        )
        formula = formula_target
        self.wait(2)

        formula_target = MathTex(
            r'\int_0^{2\pi}',
            r'{1+\tan^2\frac{\theta}{2}', r'\over', r'(\beta+1)^2', r'+(\beta+1)^2\tan^2\frac{\theta}{2}', r'-4\beta}',
            r'\mathrm d\theta'
        )
        self.play(
            ReplacementTransform(formula[0], formula_target[0]),
            ReplacementTransform(formula[1], formula_target[1]),
            ReplacementTransform(formula[2], formula_target[2]),
            FadeOut(formula[3]),
            ReplacementTransform(formula[4], formula_target[3]),
            ReplacementTransform(formula[5], formula_target[4]),
            ReplacementTransform(formula[6], formula_target[5]),
            ReplacementTransform(formula[7], formula_target[6]),
        )
        formula = formula_target
        self.wait(2)

        helper_formula = MathTex(
            r'(\beta+1)^2', r'-', r'4\beta', r'=', r'(\beta-1)^2'
        ).next_to(formula, DOWN)
        helper_formula[0].set_color(RED)
        helper_formula[2].set_color(RED)
        self.play(Write(helper_formula), formula[3].animate.set_color(RED), formula[5].animate.set_color(RED))
        self.wait(2)

        helper_formula_2 = MathTex(
            r'\tan^2\phi', r'+', r'1', r'=', r'\sec^2\phi'
        ).next_to(helper_formula, DOWN)
        helper_formula_2[0].set_color(YELLOW)
        self.play(Write(helper_formula_2), formula[1].animate.set_color(YELLOW))
        self.wait(2)

        formula_target = MathTex(
            r'\int_0^{2\pi}',
            r'{\sec^2\frac{\theta}{2}', r'\over', r'(\beta-1)^2', r'+(\beta+1)^2\tan^2\frac{\theta}{2}}',
            r'\mathrm d\theta'
        )
        self.play(
            ReplacementTransform(formula[0], formula_target[0]),
            ReplacementTransform(formula[1], formula_target[1]),
            ReplacementTransform(formula[2], formula_target[2]),
            ReplacementTransform(VGroup(formula[3], formula[5]), formula_target[3]),
            ReplacementTransform(formula[4], formula_target[4]),
            ReplacementTransform(formula[6], formula_target[5]),
        )
        self.play(Unwrite(helper_formula), Unwrite(helper_formula_2))
        formula = formula_target
        self.wait(2)
