from manim import *
from helper import *
import numpy as np
import sys
sys.path.append(".\\manim-physics\\")
from manim_physics import *


def get_rt_vir_vec(vec: np.array):
    # only for 2D vectos
    return np.array([-vec[1], vec[0], 0])


def get_lf_vir_vec(vec: np.array):
    # only for 2D vectos
    return np.array([vec[1], -vec[0], 0])


class CoordTransFromStackToCirc(ThreeDScene):
    def rotate_to_face_camera(self, *objs: Mobject):
        for obj in objs:
            yield obj.rotate(self.camera.get_phi(), axis=RIGHT).rotate(90 * DEGREES + self.camera.get_theta(), axis=OUT)

    def add_fixed_in_frame_updaters(self, *mobs : Mobject):
        for mob in mobs:
            mob.add_updater(lambda m: self.add_fixed_in_frame_mobjects(m))

    def construct(self):

        STACK_HEIGHT = 3.0
        # demonstrate the scene where wire stacking together in the x direction coordinate
        axes = ThreeDAxes(x_range=[-5, 5], y_range=[-5, 5], z_range=[-5, 5])
        x_label = axes.get_x_axis_label("y")
        y_label = axes.get_y_axis_label("x")
        z_label = axes.get_z_axis_label("z", rotation=-PI/2)
        self.add(axes)
        self.add(x_label, y_label, z_label)
        self.set_camera_orientation(phi=65 * DEGREES, theta=45 * DEGREES)
        
        
        # change y and x with prime to indicate that when doint circular integration, the theta is 
        
        current_vecs_20 = []
        current_vecs_50 = []
        current_vec_vg_20 = VGroup()
        current_vec_vg_50 = VGroup()
        wire_stack_text = Text("stack of wire segments").move_to(
            [0, 0, 4.5]).next_to(axes, OUT).scale(.4)
        self.add(*self.rotate_to_face_camera(wire_stack_text))
        for i in np.linspace(-STACK_HEIGHT, STACK_HEIGHT, 20):
            current_vecs_20.append(
                Vector(direction=[0, .4, 0], color=YELLOW).move_to([0, 0, i]))
        for i in np.linspace(-STACK_HEIGHT, STACK_HEIGHT, 50):
            current_vecs_50.append(
                Vector(direction=[0, .4, 0], color=YELLOW).move_to([0, 0, i]))
        current_vec_vg_20.add(*current_vecs_20)
        current_vec_vg_50.add(*current_vecs_50)
        self.play(Transform(current_vec_vg_20, current_vec_vg_50), run_time=2)
        self.wait(1)

        # put on grids on the xy plane, put some points there and show how this stack of wires generate mag fields
        xy_plane = NumberPlane(x_range=[-5, 5], y_range=[-5, 5], background_line_style={
            "stroke_opacity": .6,
            "stroke_color": BLUE_D
        })
        self.play(Create(xy_plane), run_time=1)
        self.wait(1)

        # put one point and connect vectors from the stack to the point
        pt_locs = np.array([
            [1, 2],
            [0, 2],
            [-1, 2],
            [2, 0]
        ])
        print(pt_locs)
        pt_locs = np.concatenate(
            (pt_locs, np.zeros((4, 1))), axis=1)  # since z = 0
        init_pt = Dot3D(pt_locs[0], color=RED, radius=.2)
        self.play(Create(init_pt), run_time=1)
        TO_PT_VEC_CNT = 5
        # arrows from different hights on the stack to the point
        to_pt_arrs: list[Arrow3D] = []
        for i in np.linspace(-STACK_HEIGHT, STACK_HEIGHT, TO_PT_VEC_CNT):
            to_pt_arrs.append(Arrow3D(
                start=[0, 0, i], end=pt_locs[0], color=BLUE, thickness=0.02, base_radius=.05))
        to_pt_arr_vg = VGroup(*to_pt_arrs)
        self.play(Create(to_pt_arr_vg), run_time=1)
        self.wait(1)

        for i, loc in enumerate(pt_locs[1:]):
            init_pt.generate_target()
            init_pt.target.move_to(loc)
            # create animation that change the tip of the arrows to the new point
            arr_anims: list[Animation] = []
            for j, arr in enumerate(to_pt_arrs):
                new_arr = Arrow3D(start=arr.get_start(), end=loc,
                                  color=BLUE, thickness=0.02, base_radius=.05)
                arr.generate_target()
                arr.target.become(new_arr)
                arr_anims.append(MoveToTarget(arr))
            # play the animation of when the point moves to the new location, the arrows follow
            self.play(MoveToTarget(init_pt), *arr_anims, run_time=1)

        self.wait(1)
        # reduce all the arrows to the center one, and mark it as \vec r
        non_center_arrs = to_pt_arrs[:2] + to_pt_arrs[3:]
        repl_anims = []
        for arr in non_center_arrs:
            repl_anims.append(Transform(arr, to_pt_arrs[2]))

        self.play(*repl_anims, run_time=1.5)
        self.wait(1)
        for arr in non_center_arrs:
            self.remove(arr)

        rvec_text = MathTex(r"\vec r", color=BLUE).move_to(
            to_pt_arrs[2].get_center() + UP * .5).scale(1.5).rotate(PI / 2)
        # change rvec to a 2d version of arrow
        rvec = Arrow(start=ORIGIN, end=pt_locs[-1], color=BLUE)
        annotated_rvec = VGroup(rvec, rvec_text)
        self.play(ReplacementTransform(
            to_pt_arrs[2], annotated_rvec), run_time=.5)

        # now create a circle with radius 10, and turn the perspective from top view
        CIRC_R = 8
        circ = Circle(radius=CIRC_R, color=MAROON).move_to(
            np.array([CIRC_R, 0, 0]))
        CIRC_LEFT = np.array([0, 0, 0])
        CIRC_RIGHT = np.array([CIRC_R * 2, 0, 0])
        CIRC_CENT = np.array([CIRC_R, 0, 0])
        diameter_line = DashedLine(
            start=CIRC_LEFT, end=CIRC_RIGHT, color=MAROON)
        self.play(Create(circ), Create(diameter_line), run_time=1)
        
        # cvec points from the center of the original coord system to the center of circle
        cvec = Arrow(start=np.array([0, 0, 0]), end=CIRC_CENT, color=GREEN)
        cvec_text = MathTex(r"\vec c", color=GREEN).move_to(
            cvec.get_center() + UP * .5).scale(1.5)
        
        # add updators that makes cvec_text face upward
    
    
        orig_coord_objs = VGroup(axes, x_label, y_label, z_label, wire_stack_text,
                                 current_vec_vg_50, current_vec_vg_20, xy_plane, cvec, cvec_text)  # objects in the original coordinate system
        trans_vec = np.array([CIRC_R, 0, 0])  # vector of translation
        self.move_camera(phi=0, theta=0, zoom=.4,
                         run_time=2, frame_center=trans_vec)
        # now turn the original stack of wire plus its coordinate, add \vec c and y^\prime

        ROTATW_TOT_TM = 10.0

        ROT_DEMO_START_ANG = PI
        ROT_DEMO_ANG_OFFSET_ABS = 35 * DEGREES
        ROT_DEMO_END_ANG = -(PI + ROT_DEMO_ANG_OFFSET_ABS)
        ROT_DEMO_ANG_OFFSET = ROT_DEMO_END_ANG - ROT_DEMO_START_ANG

        bz_formula_tex = MathTex(r"\frac{\mu_0}{2\pi}", r"{\lambda_I", r"y", r"\over", r"x^2", r"+", r"y^2}", "\mathrm{d}l").to_corner(UL)
        # paly indication animation on y^2 and x^2 + y^2 in the formula
        # indicates that we want to figure out how to represent these two values in theta
        # which is the integration variable we will use in the circular integration
        self.add_fixed_in_frame_mobjects(bz_formula_tex)
        self.remove(bz_formula_tex)
        self.play(Write(bz_formula_tex), run_time=1)
        self.wait(1)
        self.add_fixed_in_frame_updaters(bz_formula_tex)
        self.play(Indicate(bz_formula_tex[2]), run_time=1)
        self.wait(1)
        self.play(Indicate(bz_formula_tex[4:7]), run_time=1)
        self.wait(1)
        d_theta_tex = MathTex(r"\mathrm{d}\theta").next_to(bz_formula_tex, RIGHT, buff=.8)
        arr_from_dtheta_to_bz_form = Arrow(start=d_theta_tex.get_left(), end=bz_formula_tex.get_right(), color=WHITE)
        question_mark_on_arr = Text("?", color=WHITE).next_to(arr_from_dtheta_to_bz_form, UP, buff=.2)
        self.add_fixed_in_frame_mobjects(d_theta_tex, arr_from_dtheta_to_bz_form, question_mark_on_arr)
        self.remove(d_theta_tex, arr_from_dtheta_to_bz_form, question_mark_on_arr)
        self.play(AnimationGroup(
            Write(arr_from_dtheta_to_bz_form),
            Write(d_theta_tex),
            Write(question_mark_on_arr),
            lag_ratio=.1
        ), run_time=2)
        self.wait(1)
        
        
        # # remove the original one since there will be update animation
        # self.play(FadeOut(annotated_rvec))
        # rvec_text.add_updater(vec_text_updater)
        def annotated_rvec_updater(mob, alpha, start_ang, end_ang):
            ang = interpolate(start_ang, end_ang, alpha)
            # change of rvec
            print(ang)
            vecto_circum = CIRC_R * np.array([np.cos(ang), np.sin(ang), 0])
            circum_coord = CIRC_CENT + vecto_circum
            print(circum_coord,  vecto_circum)
            new_rvec = Arrow(start=circum_coord,
                             end=init_pt.get_center(), color=BLUE)
            # move the text to the mid point of that rvec

            rvec_val = init_pt.get_center() - circum_coord
            rvec_unit_tan = get_rt_vir_vec(
                rvec_val) / np.linalg.norm(rvec_val)
            new_rvec_text_pos = circum_coord + \
                (init_pt.get_center() - circum_coord) / 2 + rvec_unit_tan * .3
            rvec_tex_str = rvec_text.get_tex_string()
            rvec_tex_color = rvec_text.get_color()
            new_rvec_text = MathTex(rvec_tex_str, color=rvec_tex_color).scale(1.5).move_to(new_rvec_text_pos).rotate(PI/2)
            mob.become(VGroup(new_rvec, new_rvec_text))

        orig_coord_rot_anim = Rotate(
            orig_coord_objs, angle=ROT_DEMO_ANG_OFFSET, about_point=CIRC_CENT, axis=OUT)
        self.play(orig_coord_rot_anim, UpdateFromAlphaFunc(annotated_rvec,
                                                           update_function=lambda mob, alpha: annotated_rvec_updater(
                                                               mob, alpha, ROT_DEMO_START_ANG, ROT_DEMO_END_ANG)),
                  run_time=ROTATW_TOT_TM, rate_fun=linear)
        self.wait(1)

        # now let the orig coordinate turn to some position, do a close up and show that y^\prime is related to r dot c

        final_cvec_val = CIRC_R * \
            np.array([np.cos(ROT_DEMO_END_ANG), np.sin(
                ROT_DEMO_END_ANG), 0])
        final_mid_circum_coord = final_cvec_val / 2 + CIRC_CENT
        final_circum_coord = final_cvec_val + CIRC_CENT
        final_rvec_val = init_pt.get_center() - final_circum_coord
        self.move_camera(frame_center=final_mid_circum_coord,
                         zoom=.65, run_time=1)
        self.wait(1)

        yprime_mag: float = final_cvec_val.dot(
            final_rvec_val) / np.linalg.norm(final_cvec_val)
        yprime_vec_val = final_cvec_val / \
            np.linalg.norm(final_cvec_val) * yprime_mag
        # y component (from orig coord sys) of rvec (projected onto cvec)
        yprime = Arrow(start=final_circum_coord,
                       end=final_circum_coord + yprime_vec_val, color=RED)
        yprime_text = MathTex(r"y^\prime", color=RED).move_to(
            yprime.get_center() + get_rt_vir_vec(yprime_vec_val) * .3).scale(1.5)
        
        self.play(Create(yprime), Write(yprime_text), run_time=1)

        # connecting the end of yprime to the end of rvec
        yprime_proj_val = final_rvec_val - yprime_vec_val
        proj_st = final_circum_coord + yprime_vec_val
        proj_ed = proj_st + yprime_proj_val
        yprime_proj = DashedLine(start=proj_st, end=proj_ed, color=TEAL)


        # inciate that we can replace xprime^2 + yprime^2 with r^2
        
        self.play(Indicate(rvec_text), run_time=1.5)
        self.play(Indicate(bz_formula_tex[4:7]), run_time=1.5) # tex[5] is yprime^2
        
        print(rvec_text in self.mobjects)
        
        rvec_text_cpy = MathTex(r"|\vec{r}|^2", color=BLUE).move_to(UP * .114514)
        self.add_fixed_in_frame_mobjects(rvec_text_cpy)
        rvec_text_cpy.move_to(bz_formula_tex[4:7].get_center())
        self.remove(rvec_text_cpy)
        # self.play(
        #     rvec_text_cpy.animate.move_to(bz_formula_tex[5].get_center()),
        #     ReplacementTransform(bz_formula_tex[5], rvec_text_cpy),
        # )
        # TODO: make issue on github repo about this
        self.play(
            FadeIn(rvec_text_cpy),
            FadeOut(bz_formula_tex[4:7])
        )
        
        self.play(Indicate(bz_formula_tex[2])) # indicate y
        self.play(Indicate(yprime_text), 
                  Circumscribe(yprime),
                  run_time = 1.5
                  ) # indicate yprime
        yprime_text_cpy = MathTex(r"y^\prime", color=RED).move_to(UP * .114514)
        self.add_fixed_in_frame_mobjects(yprime_text_cpy)
        yprime_text_cpy.move_to(bz_formula_tex[2].get_center())
        self.remove(yprime_text_cpy)
        self.play(
            FadeIn(yprime_text_cpy),
            FadeOut(bz_formula_tex[2])
        )
        
        
        print(rvec_text in self.mobjects)
        
        self.wait(1)
        self.add_fixed_in_frame_updaters(rvec_text_cpy)
                
        
        self.play(Create(yprime_proj), run_time=1.5)
        yprime_formula = MathTex(r"y^\prime", r"=" ,r"{\vec c", r"\cdot", r"\vec r", r"\over", r"|", r"\vec c", r"|}").rotate(
            PI / 2).to_corner(DR).rotate(-PI/2).shift(UL * 1.5)
        yprime_formula[0].set_color(RED) # yprime
        yprime_formula[2].set_color(GREEN) # cvec
        yprime_formula[4].set_color(BLUE) # rvec
        yprime_formula[7].set_color(GREEN) # cvec
        self.add_fixed_in_frame_mobjects(yprime_formula)
        self.remove(yprime_formula)
        self.play(Write(yprime_formula), run_time=1)
        self.wait(1)
        self.add_fixed_in_frame_updaters(yprime_formula)
        
        # now move the camera to make the triangle flat
        
        dot_product_triang = VGroup(
            rvec, rvec_text, yprime, yprime_text, yprime_proj, cvec, cvec_text
        )   # the triangle of projection
        cam_trackers = {
            "phi": self.camera.phi_tracker,
            "theta": self.camera.theta_tracker,
        }
        
        # turn the signs on the triangle so that they face upwards
        rtex_up = MathTex(r"\vec r", color=BLUE).move_to(rvec_text.get_center()).scale(1.5).rotate(PI - ROT_DEMO_ANG_OFFSET_ABS)
        yptex_up = MathTex(r"y^\prime", color=RED).move_to(yprime_text.get_center()).scale(1.5).rotate(PI - ROT_DEMO_ANG_OFFSET_ABS)
        ctex_up = MathTex(r"\vec c", color=GREEN).move_to(cvec_text.get_center()).scale(1.5).rotate(PI - ROT_DEMO_ANG_OFFSET_ABS)

        self.play(
            cam_trackers["theta"].animate.set_value(PI/2 - ROT_DEMO_ANG_OFFSET_ABS),
            FadeOut((orig_coord_objs - cvec - cvec_text)),
            rvec_text.animate.become(rtex_up),
            yprime_text.animate.become(yptex_up),
            cvec_text.animate.become(ctex_up),
            run_time=3
        )
        
        cr_angle = Angle(cvec, rvec, radius=.8, quadrant=(1, 1), other_angle=True)
        
        phi_ang_tex = MathTex(r"\phi", color=WHITE).next_to(cr_angle, LEFT, buff=.2)
        self.play(
            Create(cr_angle),
            Write(phi_ang_tex),
        )
        self.wait(1)
        dotproduct_formula = MathTex(r"\vec c", r"\cdot", r"\vec r", r"=", r"|", r"\vec c", r"|",  r"\cdot", r"|", r"\vec r", r"|", r"\cos", r"\phi").set_color_by_tex_to_color_map({
            r"\vec c": GREEN,
            r"\vec r": BLUE,
            r"\phi": WHITE,
        }).to_corner(DR).shift(UL * 1.5)
        self.add_fixed_in_frame_mobjects(dotproduct_formula); 
        dotproduct_formula.to_corner(DL).shift(UR * 1.5)
        self.remove(dotproduct_formula)
        self.play(Write(dotproduct_formula), run_time=1)
        self.play(Indicate(dotproduct_formula[-4]), Indicate(dotproduct_formula[-1]), Indicate(dotproduct_formula[-2])) # indicate rvec cos phi
        self.wait(1)
        # script: even we figured out how to represent yprime, but our final goal is to represent all variables in the form of theta
        self.play(Indicate(d_theta_tex))
        self.wait(1)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=2)
        self.wait(1)


class CircIntFormula(ZoomedScene):
    def __init__(self):
        
        ZoomedScene.__init__(self, **{
            "zoom_factor": 0.17,
            "zoomed_display_height": 4,
            "zoomed_display_width": 5,
            "image_frame_stroke_width": 20,
            "zoomed_camera_config": {
                "default_frame_stroke_width": 3,
            },
        })
    def construct(self):
        HEI = self.camera.frame_height
        WID = self.camera.frame_width
        print(HEI, WID)
        # zoom out for a factor of .7
        ZOOM_FACTOR = .8
        TIP_LEN_TO_LEN_RATIO = .15
        ARR_STROKE_WID = 3
        TIP_LEN = .3
        TIP_WID = .1
        TIP_CFG = {"tip_length": TIP_LEN, "tip_width": TIP_WID, "tip_shape" : ArrowTriangleFilledTip}
        self.camera.frame_height = HEI / ZOOM_FACTOR
        self.camera.frame_width = WID / ZOOM_FACTOR
        WID /= ZOOM_FACTOR
        HEI /= ZOOM_FACTOR
        
        CIRC_INIT_RAD : float = HEI / 2 * .75
        circ_radius_tracker = ValueTracker(CIRC_INIT_RAD)
        circ_center_tracker = (ValueTracker(0), ValueTracker(0))
        circ_center = lambda : np.array([circ_center_tracker[0].get_value(), circ_center_tracker[1].get_value(), .0])
        circ_radius = lambda : circ_radius_tracker.get_value()
        
        circ = Circle(radius=CIRC_INIT_RAD, color=MAROON) # let the circle move with the tracker 
        circ.add_updater(lambda m: m.move_to(circ_center()))
        circ.add_updater(lambda m: m.set_width(circ_radius() * 2))
        
        ax = Axes(axis_config={'tip_shape': StealthTip}, x_length=circ_radius() * 2, y_length=circ_radius() * 2, x_range=[-circ_radius(), circ_radius(), 1], y_range=[-circ_radius(), circ_radius(), 1], color=WHITE)
    
        XLINE = Line(start=LEFT * WID / 2, end=RIGHT * WID / 2, color=WHITE) # does not actually display, used for getting the angle
        XLINE.add_updater(lambda m: m.move_to(circ_center()))
        YLINE = Line(start=DOWN * HEI / 2, end=UP * HEI / 2, color=WHITE)
        YLINE.add_updater(lambda m: m.move_to(circ_center()))
        xylabels = ax.get_axis_labels(x_label='x', y_label='y')
        xylabel_cur_shift = xylabels.get_center() - circ_center()
        xylabels.add_updater(lambda m: m.move_to(circ_center() + xylabel_cur_shift))
        
        # ax.add(xylabels)
        AX_OFFSET_FROM_CIRC_CENT = circ_center() - ax.get_center()        
        ax.add_updater(lambda m: m.move_to(circ_center() - AX_OFFSET_FROM_CIRC_CENT))
        # set the size of the axes so that it covers the circle perfectly
        def ax_udpater(mob : Axes):
            tar = Axes(axis_config={'tip_shape': StealthTip, **TIP_CFG}, x_length=circ_radius() * 2, y_length=circ_radius(
            ) * 2, x_range=[-circ_radius(), circ_radius(), 1], y_range=[-circ_radius(), circ_radius(), 1], color=WHITE).move_to(circ_center() - AX_OFFSET_FROM_CIRC_CENT)
            ax.become(tar)
        ax.add_updater(ax_udpater)
        
        self.add(ax, circ)
        obs_dot_pos = lambda : np.array([circ_radius() * .7, 0, 0])
        obs_dot = Dot(obs_dot_pos(), color=RED)
        obs_dot.add_updater(lambda m : m.move_to(obs_dot_pos() + circ_center()))
        
        INIT_THETA : float = .01
        self.add(obs_dot)
        
        # cvec points from the point of integration to the  center of the circle
        cvec_theta_tex = Variable(INIT_THETA, MathTex(r"\theta", color=WHITE), num_decimal_places=2).scale(.5)
        
        cvec = Arrow(start=ORIGIN, end=ORIGIN, color=GREEN,
                     max_tip_length_to_length_ratio=TIP_LEN_TO_LEN_RATIO)
        def integration_pt_pos() -> np.ndarray:
            return vec_by_polar(circ_radius(), cvec_theta_tex.value.get_value()) + circ_center()
        def cvec_updater(mob : Arrow):
            mob.become(Line(start=integration_pt_pos(),
                       end=circ_center(), color=GREEN).add_tip(**TIP_CFG))
        cvec.add_updater(cvec_updater)
        cvec_tex = MathTex(r"\vec c", color=GREEN).scale(.8)
        def cvec_tex_updater(mob: MathTex):
            cvec_midpos = (cvec.get_start() + cvec.get_end()) / 2
            # shift cvec_tex to the top of midpos 
            cvec_tan_vec = np.array([-cvec_midpos[1], cvec_midpos[0], .0]) / np.linalg.norm(cvec_midpos)
            cvec_tex_pos = cvec_midpos + cvec_tan_vec * .3
            mob.become(MathTex(r"\vec c", color=GREEN).move_to(cvec_tex_pos))
        cvec_tex.add_updater(cvec_tex_updater)
        self.add(cvec, cvec_tex)
        
        # rvec points from the point of integration to the point of observation 
        rvec = Line(start=ORIGIN, end=ORIGIN, color=BLUE)
        rvec_tex = MathTex(r"\vec r", color=BLUE).scale(.8)
        def rvec_updater(mob : Mobject):
            mob.become(Line(start=integration_pt_pos(), end=obs_dot.get_center(
            ), color=BLUE).add_tip(**TIP_CFG))
        def rvec_tex_updater(mob : MathTex):
            rvec_val = rvec.get_end() - rvec.get_start()
            rvec_tan = np.array([-rvec_val[1], rvec_val[0], .0]) / np.linalg.norm(rvec_val)
            rvec_tex_pos = (rvec.get_start() + rvec.get_end()) / 2 + rvec_tan * .3
            mob.become(MathTex(r"\vec r", color=BLUE,).move_to(rvec_tex_pos))
        rvec.add_updater(rvec_updater)
        rvec_tex.add_updater(rvec_tex_updater)
        self.add(rvec, rvec_tex)
        
        # yprime is the projection of rvec onto cvec
        yprime = Line(start=ORIGIN, end=ORIGIN, color=RED)
        yprime_tex = MathTex(r"y'", color=RED).scale(.8)
        def yprime_updater(mob : Mobject):
            rvec_val = rvec.get_end() - rvec.get_start()
            cvec_val = cvec.get_end() - cvec.get_start()
            yprime_mag = np.dot(rvec_val, cvec_val) / np.linalg.norm(cvec_val) 
            yprime_val = cvec_val / np.linalg.norm(cvec_val) * yprime_mag
            mob.become(Line(start=cvec.get_start(), end=cvec.get_start(
            ) + yprime_val, color=RED).add_tip(**TIP_CFG))
        def yprime_tex_updater(mob : Mobject):
            yprime_val = yprime.get_end() - yprime.get_start()
            yprime_tan = np.array([yprime_val[1], -yprime_val[0], .0]) / np.linalg.norm(yprime_val)
            yprime_tex_pos = (yprime.get_start() + yprime.get_end()) / 2 + yprime_tan * .3
            mob.become(MathTex(r"y'", color=RED).move_to(yprime_tex_pos))
        yprime.add_updater(yprime_updater)
        yprime_tex.add_updater(yprime_tex_updater)
        self.add(yprime, yprime_tex)
        
        
        # add updaters for theta angle sign
        angle_sign : Angle = Angle(XLINE, YLINE) # temporary value, will be updated by its updater
        ANG_SIGN_RAD = .5
        def angle_sign_updater(mob : Angle):
            tmp = Angle(XLINE, cvec, radius=ANG_SIGN_RAD,
                        quadrant=[1, -1], color=WHITE)
            mob.become(tmp)
        angle_sign.add_updater(angle_sign_updater)
        self.add(cvec_theta_tex, angle_sign)
        
        def cvec_theta_tex_updater(mob : Variable):
            # move to the top left of angle sign 
            between_cvec_xline = vec_by_polar(
                1, cvec_theta_tex.value.get_value() / 2)
            cvec_theta_tex_pos = between_cvec_xline * (ANG_SIGN_RAD + .3) + circ_center()
            mv_vec = cvec_theta_tex_pos - cvec_theta_tex.get_left()
            mob.shift(mv_vec)
            
        circum_x_tex = Variable(0, "x", color=WHITE, num_decimal_places=2).scale(.6)
        def circum_x_updater(mob : Variable):
            mob.tracker.set_value(integration_pt_pos()[0])
            mob.move_to(integration_pt_pos() * 1.2)
        circum_y_tex = Variable(0, "y", color=WHITE, num_decimal_places=2).scale(.6)
        def circum_y_updater(mob : Variable):
            mob.tracker.set_value(integration_pt_pos()[1])
            mob.next_to(circum_x_tex, RIGHT, buff=.1)
        circum_x_tex.add_updater(circum_x_updater)
        circum_y_tex.add_updater(circum_y_updater)
        self.add(circum_x_tex, circum_y_tex)

        
        cvec_theta_tex.add_updater(cvec_theta_tex_updater)
        theta_var_tracker = cvec_theta_tex.tracker
        self.play(theta_var_tracker.animate.set_value(40 * DEGREES + 2 * PI), run_time=6)
        
        # move all the previous scene to upper left corner, and start writing formula 
        rotation_int_mobs_lst = [
            rvec, rvec_tex, cvec_theta_tex, circum_x_tex, circum_y_tex, circ, ax, cvec, cvec_tex, obs_dot, yprime, yprime_tex, angle_sign
        ]
        for mob in rotation_int_mobs_lst:
            mob.update()
        self.wait(1)
        
        circ_left_buf = .5
        new_cent_x = -WID / 2 + circ_left_buf + circ_radius() * .7
        self.play(circ_center_tracker[0].animate.set_value(new_cent_x), circ_radius_tracker.animate.set_value(circ_radius() * .7), run_time=1.5)
        self.wait(1)
        
        bz_formula_tex = MathTex(r"\frac{\mu_0}{2\pi}{\lambda_I", r"y^\prime", r"\over", r"|\vec{r}|^2}", r"\mathrm d l").set_color_by_tex_to_color_map({"y^\prime": RED, r"\vec{r}^2": BLUE}).to_corner(UR)
        yprime_formula = MathTex(r"y^\prime", r"=" ,r"{\vec c", r"\cdot", r"\vec r", r"\over", r"|", r"\vec c", r"|}").set_color_by_tex_to_color_map({"y^\prime": RED, r"\vec c": GREEN, r"\vec r": BLUE}).next_to(bz_formula_tex, DOWN)
        self.play(Write(yprime_formula), Write(bz_formula_tex))
        q_theta_tex = MathTex(r"\theta").next_to(yprime_formula, RIGHT, buff=1)
        arr_from_theta_to_yprime_form = Arrow(q_theta_tex.get_left(), yprime_formula.get_right(), buff=.1)
        qmark_on_arr = MathTex(r"?").next_to(arr_from_theta_to_yprime_form, UP, buff=.1)
        self.play(
            AnimationGroup(
                Write(arr_from_theta_to_yprime_form),
                Write(q_theta_tex),
                Write(qmark_on_arr),
                lag_ratio=.15
            ),
            run_time=2
        )
        self.wait(1)
        # now show how to represent 
        integration_pt_formula = MathTex(r"\scriptsize{\text{point of integration}}", r"=", r""" 
                                        R\cdot \begin{bmatrix}
                                        \cos \theta \\
                                        \sin \theta
                                        \end{bmatrix}
                                        """).to_edge(UP)
        integration_pt = Dot(ORIGIN, color=YELLOW)
        integration_pt.add_updater(lambda mob : mob.move_to(integration_pt_pos()))
        self.play(Write(integration_pt_formula))
        self.wait(1)
        self.play(Write(integration_pt))
        # change theta on the circle a litte, and indicate (wiggle) where is the integration point 
        TWEAK_ANGLE = 15 * DEGREES
        self.play(
            theta_var_tracker.animate.increment_value(TWEAK_ANGLE),
            Indicate(integration_pt),
            run_time=1.5
        )
        for mob in rotation_int_mobs_lst:
            mob.update()
        
        self.play(
            theta_var_tracker.animate.increment_value(-TWEAK_ANGLE),
            Indicate(integration_pt),
            run_time=1.5
        )
        self.wait(1)
        
        # script: since cvec points from the point of consideration to the origin, we have:
        cvec_formula = MathTex(r"\vec c", r"=", r"-", r"R\cdot \begin{bmatrix} \cos \theta \\ \sin \theta \end{bmatrix}").next_to(integration_pt_formula, DOWN).set_color_by_tex_to_color_map({r"\vec c": GREEN}).scale(.8)
        self.play(Write(cvec_formula))
        self.wait(1)
        
        observation_pt_formula = MathTex(r"\scriptsize{\text{point of observation}}", r"=", r"""
                                            \begin{bmatrix}
                                                x\\
                                                0
                                            \end{bmatrix}
                                         """).next_to(cvec_formula, DOWN).scale(.8)
        self.play(
            Write(observation_pt_formula), 
            Indicate(obs_dot),
            run_time=1.5
        )
        self.wait(1)
        
        # script: rvec is the vector from the point of integration to the point of observation
        rvec_formula = MathTex(r"\vec r", r"=", r""" 
                                    \begin{bmatrix}
                                        x\\
                                        0
                                    \end{bmatrix} - 
                                    R\begin{bmatrix}
                                        \cos \theta \\
                                        \sin \theta
                                    \end{bmatrix}
                                """, r"=", r""" 
                                    \begin{bmatrix}
                                        x - R\cos \theta \\
                                        -R\sin \theta
                                    \end{bmatrix}
                                """).set_color_by_tex_to_color_map({r"\vec r": BLUE}).next_to(observation_pt_formula, DOWN).scale(.8).shift(RIGHT * .1)
        self.play(
            AnimationGroup(
                Circumscribe(rvec, run_time=1.5),
                Write(rvec_formula[:-2], run_time=1.5),
                lag_ratio=.4
            )
            )   
        self.wait(1)
        self.play(Write(rvec_formula[-2:]))
        self.wait(1)
        
        yprime_mat_formula = yprime_formula.copy().scale(.8).next_to(rvec_formula, DOWN)
        self.play(Write(yprime_mat_formula))
        # now start to transform the formula
        self.wait(1)
        # replace cvec with vector representation
        yprime_mat_formula2_tex_string_by_part = [yprime_mat_formula[i].get_tex_string() for i in range(len(yprime_mat_formula))]
        yprime_mat_formula2_tex_string_by_part[2] = r"{" + cvec_formula[3].get_tex_string() # cvec in matrix form
        yprime_mat_formula2 = MathTex(*yprime_mat_formula2_tex_string_by_part).scale(.8).next_to(rvec_formula, DOWN).set_color_by_tex_to_color_map({r"\vec r": BLUE, r"y^\prime": RED, r"\vec c": GREEN})
        yprime_mat_formula2[2].set_color(GREEN)
        self.play(
            TransformMatchingTex(yprime_mat_formula, yprime_mat_formula2),
            run_time=2
        )
        self.wait(1)

        yprime_mat_formula3_tex_string_by_part = yprime_mat_formula2_tex_string_by_part.copy() # replace rvec with vector representation
        yprime_mat_formula3_tex_string_by_part[4] = rvec_formula[-1].get_tex_string()
        yprime_mat_formula3 = MathTex(*yprime_mat_formula3_tex_string_by_part).scale(.8).next_to(rvec_formula, DOWN).set_color_by_tex_to_color_map({r"\vec r": BLUE, r"y^\prime": RED, r"\vec c": GREEN})
        yprime_mat_formula3[2].set_color(GREEN); yprime_mat_formula3[4].set_color(BLUE)
        self.play(
            TransformMatchingTex(yprime_mat_formula2, yprime_mat_formula3),
            run_time=2
        )
        self.wait(1)
        
        # now change |\vev c| to R, script: since cvec points from one point in the circumference to the origin, its magnitude must be radius of the circle
        yprime_mat_formula4_tex_string_by_part = yprime_mat_formula3_tex_string_by_part[:-3].copy()
        yprime_mat_formula4_tex_string_by_part.append(r"R}")
        yprime_mat_formula4 = MathTex(*yprime_mat_formula4_tex_string_by_part).scale(.8).next_to(rvec_formula, DOWN).set_color_by_tex_to_color_map({r"\vec r": BLUE, r"y^\prime": RED, r"\vec c": GREEN})
        yprime_mat_formula4[2].set_color(GREEN); yprime_mat_formula4[4].set_color(BLUE)
        self.play(
            TransformMatchingTex(yprime_mat_formula3, yprime_mat_formula4),
            run_time=2
        )
        self.wait(1)
        
        # cancle out the R both in the numerator and denominator
        self.play(
            Indicate(yprime_mat_formula4[2][0]), # R on the numerator
            Indicate(yprime_mat_formula4[-1]),   # R on the denominator
        )
        self.wait(1)
        yprime_mat_formula5 = MathTex(r"y^\prime", r"=", r"""
                                        \begin{bmatrix}
                                        \cos\theta \\ 
                                        \sin\theta
                                        \end{bmatrix}""", r"\cdot", r"""
                                        \begin{bmatrix}
                                        x - R\cos\theta \\ 
                                        -R\sin\theta
                                        \end{bmatrix}"""
                                      ).next_to(rvec_formula, DOWN).scale(.8)
        yprime_mat_formula5[0].set_color(RED);
        yprime_mat_formula5[2].set_color(GREEN); yprime_mat_formula5[-1].set_color(BLUE)
        self.play(
            TransformMatchingTex(yprime_mat_formula4, yprime_mat_formula5),
            run_time=2
        )
        self.wait(1)
        
        yprime_mat_formula6 = MathTex(r"y^\prime", r"=", r"(-\cos\theta)", r"\cdot" ,r"(x - R\cos\theta)", r"+", r"\\"
                                    r"(\sin\theta)", r"\cdot" ,r"(-R\sin\theta)")
        yprime_mat_formula6[0].set_color(RED)
        yprime_mat_formula6.next_to(rvec_formula, DOWN).scale(.8)
        self.play(
            TransformMatchingTex(yprime_mat_formula5, yprime_mat_formula6),
            run_time=2
        )
        self.wait(1)
        
        # now clear the previous formula from the top
        
        formula_clear_objs = [integration_pt_formula,
                              cvec_formula, observation_pt_formula, rvec_formula]
        formula_clear_anims = [Unwrite(obj) for obj in formula_clear_objs]
        self.play(
            AnimationGroup(
            AnimationGroup(*formula_clear_anims),
            yprime_mat_formula6.animate.to_edge(UP),
            lag_ratio=.5
            ),
            run_time=3
        )
        self.wait(1)
        
        yprime_mat_formula7 = MathTex(
            "=R", r"\left(-", r"{x", r"\over", r"R}", r"\cos\theta", "+", r"\cos^2\theta", "+", r"\sin^2\theta", r"\right)"
        ).scale(.8).next_to(yprime_mat_formula6, DOWN)
        
        self.play(
            Write(yprime_mat_formula7),
            run_time=2
        )
        self.wait(1)
        
        # replace x/r with beta
        yprime_mat_formula8 = MathTex(
            "=R", r"\left(-", r"\beta\cos\theta", "+", r"\cos^2\theta", "+", r"\sin^2\theta", r"\right)"
        ).scale(.8).next_to(yprime_mat_formula6, DOWN)
        self.play(TransformMatchingTex(yprime_mat_formula7, yprime_mat_formula8), run_time=2)
        
        # indicate the part to use pythagorean identity
        self.play(
            Indicate(yprime_mat_formula8[-4:]), run_time=2)
        yprime_mat_formula9_tex_string_by_part = [yprime_mat_formula8[i].get_tex_string() for i in range(len(yprime_mat_formula8))]
        yprime_mat_formula9_tex_string_by_part = yprime_mat_formula9_tex_string_by_part[:-4].copy()
        yprime_mat_formula9_tex_string_by_part.append(r"1"); yprime_mat_formula9_tex_string_by_part.append(r"\right)") 
        yprime_mat_formula9 = MathTex(*yprime_mat_formula9_tex_string_by_part).scale(.8).next_to(yprime_mat_formula6, DOWN)
        self.play(TransformMatchingTex(yprime_mat_formula8, yprime_mat_formula9), run_time=2)
        self.wait(1)
        
        # move the bz formula to the down side of this, and plug it in
        self.play(bz_formula_tex.animate.next_to(yprime_mat_formula9, DOWN), run_time=2)
        self.wait(1)
        
        # replace the yprime in the bz formula
        bz_formula_other_tex_str =[yprime_mat_formula9[1:][i].get_tex_string() for i in range(len(yprime_mat_formula9[1:]))]
        bz_formula_2 = MathTex(
            r"{2k\lambda_I", r"R", *bz_formula_other_tex_str, r"\over", r"|\vec{r}|^2}", "\mathrm d l").scale(.8).next_to(yprime_mat_formula9, DOWN, buff=.4)
        bz_formula_2[-2].set_color(BLUE)
        for sub_tex in bz_formula_other_tex_str:
            bz_formula_2.set_color_by_tex(sub_tex, RED)
        self.play(TransformMatchingTex(bz_formula_tex, bz_formula_2), run_time=2)
        self.wait(1)
        
        # use matrix form of rvec to replace the rvec inside the bz formula2 
        rvec_formula_cpy = MathTex(rvec_formula.get_tex_string()).next_to(yprime_formula, DOWN).scale(.8).set_color(BLUE)
        self.play(Write(rvec_formula_cpy), run_time=1)
        
        bz_formula_3_by_part_tex = [bz_formula_2[i].get_tex_string() for i in range(len(bz_formula_2))]
        bz_formula_3_by_part_tex[-2] = r"(x - R\cos\theta)^2 + (R\sin\theta)^2}"
        bz_formula_3 = MathTex(*bz_formula_3_by_part_tex).scale(.8).next_to(yprime_mat_formula9, DOWN, buff=.4)
        for sub_tex in bz_formula_other_tex_str:
            bz_formula_3.set_color_by_tex(sub_tex, RED)
        bz_formula_3[-2].set_color(BLUE); 
        self.play(TransformMatchingTex(bz_formula_2, bz_formula_3), run_time=2)
        self.wait(1)
        self.play(Unwrite(rvec_formula_cpy), run_time=1)
        self.wait(1)
        
        bz_formula_4 = MathTex(
            "{2R", r"(1-\beta\cos\theta)", "k", r"\lambda_I", r"\over", r"x^2 - 2xR\cos\theta +", r"R^2\cos^2\theta + R^2\sin^2\theta}", "\mathrm d l"
        ).next_to(bz_formula_3, DOWN).scale(.8)
        bz_formula_4[5:7].set_color(BLUE)
        bz_formula_4[1].set_color(RED)
        self.play(Write(bz_formula_4, run_time=2), circ_center_tracker[0].animate.increment_value(-.3))
        self.wait(1)
        
        # indicate the part to use pythagorean identity
        self.play(Indicate(bz_formula_4[-2]), run_time=2)
        self.wait(1)
        
        bz_formula_5 = MathTex(
            "{2R", r"(1-\beta\cos\theta)", "k", r"\lambda_I", r"\over", r"x^2 - ", r"2x","R", r"\cos\theta +", r"R^2}", "\mathrm d l"
        ).next_to(bz_formula_3, DOWN).scale(.8)
        bz_formula_5[5:-1].set_color(BLUE)
        bz_formula_5[1].set_color(RED)
        self.play(TransformMatchingTex(bz_formula_4, bz_formula_5), circ_center_tracker[0].animate.increment_value(.3), run_time=2)
        self.wait(1)
        
        # indicate the R in denominator that could be reduced
        self.play(
            Indicate(bz_formula_5[-2]), 
            Indicate(bz_formula_5[-4]), run_time=2)
        self.wait(1)
        
        bz_formula_6 = MathTex(
            "{2R", r"(1-\beta\cos\theta)", "k", r"\lambda_I", r"\over", r"R^2", r"(\beta^2 - 2\beta\cos\theta + 1)}", "\mathrm d l"
        ).next_to(bz_formula_3, DOWN).scale(.8)
        bz_formula_6[5:-1].set_color(BLUE)
        bz_formula_6[1].set_color(RED)
        self.play(TransformMatchingTex(bz_formula_5, bz_formula_6), run_time=2)
        self.wait(1)
        
        # indicate the R in both denominator and numerator, and cancel them
        self.play(
            Indicate(bz_formula_6[0]),
            Indicate(bz_formula_6[-3])
        )
        self.wait(1)
        
        bz_formula_7 = MathTex(
            "{2", r"(1-\beta\cos\theta)", "k", r"\lambda_I", r"\over", r"R", r"(\beta^2 - 2\beta\cos\theta + 1)}", "\mathrm d l"
        ).next_to(bz_formula_3, DOWN).scale(.8)
        bz_formula_7[5:-1].set_color(BLUE); bz_formula_7[1].set_color(RED)
        self.play(TransformMatchingTex(bz_formula_6, bz_formula_7), run_time=2)
        self.wait(1)
        
        # now try to replace the dl with d\theta
        self.play(
            Indicate(bz_formula_7[-1])
        )
        
        arr_from_q_theta_tex_to_dl = Line(start = q_theta_tex.get_left() + LEFT * .1, end = bz_formula_7[-1].get_right() + RIGHT + .1).add_tip(**TIP_CFG)
        self.play(
            Indicate(q_theta_tex),
            Write(arr_from_q_theta_tex_to_dl),
            run_time=2
        )
        self.play(Unwrite(arr_from_q_theta_tex_to_dl), run_time=1)
        self.wait(1)

        # clear up the screen 
        clearing_vgrp = VGroup(bz_formula_3, yprime_mat_formula9, yprime_formula,
                               q_theta_tex, arr_from_theta_to_yprime_form, qmark_on_arr, yprime_mat_formula6)
        orig_bz_formula_7_pos = bz_formula_7.get_center()
        self.play(
            AnimationGroup(
                Unwrite(clearing_vgrp),
                bz_formula_7.animate.to_corner(UR),
                lag_ratio = .7
            ),
            run_time=3
        )
        self.wait(1)
    
        zoomed_cam = self.zoomed_camera
        zoomed_disp = self.zoomed_display
        zoomed_frame = zoomed_cam.frame
        zoomed_disp.next_to(circ, RIGHT).set_width(5).shift(RIGHT)
        zoomed_frame.move_to(integration_pt)
        self.remove(circum_x_tex, circum_y_tex)
        self.activate_zooming(True)
        # self.play(
        #     self.get_zoomed_display_pop_out_animation()
        # )
        self.wait(1)
        
        # move theta a little to show what does dl mean 
        THETA_TWEAK_VAL = 8 * DEGREES
        trace = TracedPath(integration_pt_pos, color=PURPLE)
        self.add(trace)
        prev_int_pt_pos = integration_pt_pos()
        self.play(
            theta_var_tracker.animate.increment_value(THETA_TWEAK_VAL),
        )
        self.play(
            theta_var_tracker.animate.increment_value(-THETA_TWEAK_VAL * 2),
        )
        
        dl_tex = MathTex("\mathrm d l").scale(.3).next_to(prev_int_pt_pos, RIGHT, buff=.1)
        self.play(Write(dl_tex))
        self.wait(1)
        
        # write that dl = R d\theta
        dl_formula = MathTex(r"\mathrm d l = R \mathrm d \theta").scale(.8).next_to(bz_formula_7, DOWN)
        self.play(Write(dl_formula))
        self.wait(1)
        self.play(
            self.get_zoomed_display_pop_out_animation(),
            rate_func=lambda t: smooth(1 - t)
        )
        self.wait(1)
        # remove the zooming camera
        # self.play(FadeOut(zoomed_frame))
        self.remove(zoomed_frame, zoomed_disp, zoomed_cam)
        self.wait(1)
        
        # move the previous formula to the center
        shfit_vec = bz_formula_7.get_center() - orig_bz_formula_7_pos
        shfit_vec[1] = .0
        print("shift vec ", shfit_vec)
        self.play(
            bz_formula_7.animate.shift(LEFT * shfit_vec), dl_formula.animate.shift(LEFT * shfit_vec)
        )
        self.wait(1)
        
        final_dbz_formula = MathTex(r"\mathrm d B=", "{2", r"(1-\beta\cos\theta)", "k", r"\lambda_I", r"\over", r"R", r"(\beta^2 - 2\beta\cos\theta + 1)}", r"R", r"\mathrm d \theta").next_to(dl_formula, DOWN).scale(.8)
        self.play(Write(final_dbz_formula))
        self.wait(1)
        # cancle out the R
        self.play(Indicate(final_dbz_formula[-2]), Indicate(final_dbz_formula[-4]))
    
        final_dbz_formula2 = MathTex(r"\mathrm d B=", "{2", r"(1-\beta\cos\theta)", "k", r"\lambda_I", r"\over",
                                      r"(\beta^2 - 2\beta\cos\theta + 1)}", r"\mathrm d \theta").next_to(dl_formula, DOWN).scale(.8)
        self.play(TransformMatchingTex(final_dbz_formula, final_dbz_formula2))
        self.wait(1)
        
        final_bz_formula = MathTex(r"B=", "\int_0^{2\pi}", r"\mathrm d B=", "{2", r"(1-\beta\cos\theta)",
                                   "k", r"\lambda_I", r"\over", r"R", r"(\beta^2 - 2\beta\cos\theta + 1)}", r"R \mathrm d \theta").next_to(final_dbz_formula, DOWN).scale(.8).shift(RIGHT * .5)
        self.play(Write(final_bz_formula))
        self.wait(1)
        
class InfLineAmpLaw(ThreeDScene):
    def ampere_infline_int(self, alpha: float, current: float) -> float:
        # the integrand of the ampere loop law when integrating over an infinite long straight line
        MU_0 = 4 * PI * 1e-7
        return current * alpha

    def construct(self):
        HEI = self.camera.frame_height
        WID = self.camera.frame_width
        # let the wire pass through the screen, into it
        line_start = np.array([0, 0, -10])
        line_end = np.array([0, 0, 10])
        xrg = [-WID / 2, WID / 2, .5]
        yrg = [-HEI / 2, HEI / 2, .5]

        # introduce the ampere loop law
        wire_line = Line(start=line_start, end=line_end)
        # first draw the wire that going out of the screen
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)
        self.play(Create(wire_line), run_time=1)

        # draw some dots to indicate the flow of current
        DOT_CNT = 15
        DOT_TRAVEL_MAX_TM = 4  # seconds
        electron_dots = []
        for i in range(DOT_CNT):
            electron_dots.append(Dot3D(wire_line.get_start(), color=BLUE))
        electron_moving_anim = LaggedStart(
            *[MoveAlongPath(electron_dots[i], wire_line)
              for i in range(DOT_CNT)],
            lag_ratio=(1/DOT_CNT), run_time=DOT_TRAVEL_MAX_TM
        )

        wire = Wire(wire_line, current=2)
        mag_field = MagneticField(wire, x_range=xrg, y_range=yrg)
        sxrg = xrg.copy()
        sxrg[-1] = .3
        syrg = yrg.copy()
        syrg[-1] = .3
        mag_streamlines = StreamLines(
            func=mag_field.func, x_range=sxrg, y_range=syrg, opacity=.5)
        mag_f_create_anim = AnimationGroup(SpinInFromNothing(
            mag_field), SpinInFromNothing(mag_streamlines))
        # sprialIn -> 
        self.play(
            LaggedStart(
                electron_moving_anim,
                mag_f_create_anim,
                lag_ratio=.3,
            )
        )
        self.wait(1)
        # now change the perspective to show how to use ampere loop law
        self.remove(*electron_dots)
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES, run_time=1)
        current_out_sign = current_out_symbol(
            wire_line.get_center(), 0.2, color=WHITE)
        current_amount = MathTex(r"I = 2\text{A}", color=WHITE).next_to(
            current_out_sign, RIGHT)
        self.play(Create(current_out_sign), Write(current_amount), run_time=1)

        # put an amperian loop around the wire
        LOOP_RAD = 3
        ampere_law_text = MathTex(
            r"\oint \vec B\cdot", r"\mathrm d\vec l", r" = \mu_0I_{\text{enclosed}}", color=WHITE)
        ampere_law_text.to_corner(UL)
        ampere_law_text.set_color_by_tex(r"\mathrm d\vec l", BLUE)
        amp_loop = Circle(radius=LOOP_RAD, color=RED).move_to(
            wire_line.get_center())
        self.play(Create(amp_loop), run_time=1)
        self.play(Write(ampere_law_text), run_time=1)
        circ_seg_d_ang = DEGREES * 5
        circ_seg = Arc(radius=LOOP_RAD, start_angle=0, angle=circ_seg_d_ang, color=BLUE,
                       stroke_width=4).move_arc_center_to(amp_loop.get_center()).shift(OUT * 0.01)
        # use as dl
        integration_val = MathTex(
            r"\frac{\oint \vec B \cdot \mathrm d\vec l}{\mu_0} = 0").to_corner(UR)

        def inte_upd(mob: Text, alpha: float) -> None:
            nex_txt = r"\frac{\oint \vec B \cdot \mathrm d\vec l}{\mu_0}" + \
                "={:.2f}".format(self.ampere_infline_int(alpha, wire.current))
            mob.become(MathTex(nex_txt).to_corner(UR))

        integration_anim = UpdateFromAlphaFunc(
            integration_val, update_function=inte_upd)

        move_along_circ_anim = Rotate(
            circ_seg, angle=2 * PI - circ_seg_d_ang, about_point=amp_loop.get_center())
        self.play(Create(circ_seg), Write(integration_val), run_time=1)
        # self.play(integration_anim)
        self.play(move_along_circ_anim, integration_anim, run_time=4)
        self.wait(1)

        LOOP_RAD = 2
        def to_zero_updter(mob : Mobject, alpha : float) -> None:
            mob.become(MathTex(r"\frac{\oint \vec B \cdot \mathrm d\vec l}{\mu_0} = 0.00").to_corner(UR))
        
        self.play(
            UpdateFromAlphaFunc(integration_val, to_zero_updter),
            circ_seg.animate.become(Arc(radius=LOOP_RAD, start_angle=0, angle=circ_seg_d_ang, color=BLUE,
                                    stroke_width=4).move_arc_center_to(amp_loop.get_center()).shift(OUT * 0.01)),
            amp_loop.animate.become(
                Circle(radius=LOOP_RAD, color=RED).move_to(wire_line.get_center())),
            run_time=1.5)

        
        self.wait(1)
        self.play(move_along_circ_anim, integration_anim, run_time=4)
        self.wait(1)


class SolenoidAmpLaw(ThreeDScene):
    def construct(self):
        WID = self.camera.frame_width
        HEI = self.camera.frame_height
        cam_trackers = {
            "phi": self.camera.phi_tracker,
            "theta": self.camera.theta_tracker,
        }
        dbg_axis = ThreeDAxes()
        self.add(dbg_axis)
        xlabel = dbg_axis.get_x_axis_label("x")
        ylabel = dbg_axis.get_y_axis_label("y")
        zlabel = dbg_axis.get_z_axis_label("z")
        self.add(xlabel, ylabel, zlabel)

        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)
        # generate solenoid
        SOL_COIL = 20
        solenoid = Solenoid(2, 10, SOL_COIL, color=WHITE)
        inouts = solenoid.get_inout_symbols(.2, PI/2, False, color=WHITE)
        # rotate the solenoid so that it will be horizontal
        sol_with_sign = VGroup(solenoid, inouts).rotate(90*DEGREES, axis=UP)

        sol_wire = Wire(solenoid, current=2, samples=128)
        self.play(Create(solenoid))

        DOT_CNT = 25
        DOT_TRAVEL_MAX_TM = 8  # seconds
        electron_dots = [Dot3D(solenoid.get_start(), color=BLUE)
                         for _ in range(DOT_CNT)]
        electron_moving_anim = LaggedStart(
            *[MoveAlongPath(electron_dots[i], solenoid, run_time = DOT_TRAVEL_MAX_TM / 3)
              for i in range(DOT_CNT)],
            lag_ratio=(1/DOT_CNT), run_time=DOT_TRAVEL_MAX_TM
        )

        # create magnetic field and streamline in the cross section of the solenoid
        cam_perspective_change_anim = AnimationGroup(
            cam_trackers["phi"].animate.set_value(0),
            cam_trackers["theta"].animate.set_value(-90 * DEGREES),
            run_time = 1.5
        )
        xrg = [-WID/2, WID/2, .5]
        sxrg = [-WID/2, WID/2, .3]
        yrg = [-HEI/2, HEI/2, .5]
        syrg = [-HEI/2, HEI/2, .3]
        mag_field = MagneticField(sol_wire, x_range=xrg, y_range=yrg)
        mag_streamline = StreamLines(
            func=mag_field.func, x_range=sxrg, y_range=syrg, opacity=.5)
        field_gen_anim = AnimationGroup(
            SpinInFromNothing(mag_field), SpinInFromNothing(mag_streamline), run_time=1.5
        )

        show_sign_anim = AnimationGroup(FadeOut(solenoid), FadeOut(*electron_dots),
                                        FadeIn(inouts))

        self.play(
            AnimationGroup(
                electron_moving_anim,
                cam_perspective_change_anim,
                field_gen_anim,
                lag_ratio=.35
            )
        )
        self.play(show_sign_anim, run_time=2)
        self.wait(1)
        
        # put an rectangular amperian loop in the middle of the solenoid
        amp_rect = Rectangle(width=4, height=2, color=WHITE)
        amp_rect.move_to(ORIGIN + OUT * .001)
        self.play(Create(amp_rect))
        int_segment = Square(side_length=.05, color=BLUE, fill_color=BLUE, fill_opacity=1)
        # demonstrate integration using amperian loop
        self.play(MoveAlongPath(int_segment, amp_rect), run_time=2)
        self.wait(1)
        
        # indicate that there are no enclosed current 
        cur_inc_0_txt = MathTex(r"I_{\text{enclosed}} = 0").to_corner(UR)
        
        cent_amp_rect = Rectangle(width=0, height=0, color=WHITE)
        amp_rect_cpy = amp_rect.copy().set_stroke(width=15)
        self.play(amp_rect_cpy.animate.become(cent_amp_rect), run_time=1.5)
        self.remove(amp_rect_cpy, cent_amp_rect)
        self.play(Write(cur_inc_0_txt))
        self.wait(1)
        self.remove(int_segment)
        
        amp_rect_upline = Line(amp_rect.get_corner(UR), amp_rect.get_corner(UL), color=WHITE)
        amp_rect_downline = Line(amp_rect.get_corner(DL), amp_rect.get_corner(DR), color=WHITE)
        amp_rect_leftline = Line(amp_rect.get_corner(UL), amp_rect.get_corner(DL), color=WHITE)
        amp_rect_rightline = Line(amp_rect.get_corner(DR), amp_rect.get_corner(UR), color=WHITE)
        up_int_text =  MathTex(r"\oint \vec{B} \cdot \mathrm d\vec{l} \ (\text{up})").scale(.6).next_to(
            amp_rect_upline.get_start(), UP, buff=.15
        )
        dn_int_text = MathTex(r"\oint \vec{B} \cdot \mathrm d\vec{l} \ (\text{down})").scale(.6).next_to(
            amp_rect_downline.get_start(), DOWN, buff=.15)
        lf_int_text = MathTex(r"\oint \vec{B} \cdot \mathrm d\vec{l} \ (\text{left})").scale(.6).next_to(
            amp_rect_leftline.get_start(), LEFT, buff=.15)
        rt_int_text = MathTex(r"\oint \vec{B} \cdot \mathrm d\vec{l} \ (\text{right})").scale(.6).next_to(
            amp_rect_rightline.get_start(), RIGHT, buff=.15)
        int_segment.move_to(amp_rect_upline.get_start())
        lf_int_segment = int_segment.copy().move_to(amp_rect_leftline.get_start())
        rt_int_segment = int_segment.copy().move_to(amp_rect_rightline.get_start())

        def up_intseg_udpater(mob : Mobject):
            mob.next_to(int_segment, UP, buff=.15)
        def dn_intseg_updater(mob : Mobject):
            mob.next_to(int_segment, DOWN, buff=.15)
        def lf_intseg_updater(mob : Mobject):
            mob.next_to(lf_int_segment, LEFT, buff=.15)
        def rt_intseg_updater(mob : Mobject):
            mob.next_to(rt_int_segment, RIGHT, buff=.15)
        # show the left and right line integra is zero
        
        lf_int_text.add_updater(lf_intseg_updater)
        rt_int_text.add_updater(rt_intseg_updater)
        self.play(
            MoveAlongPath(lf_int_segment, amp_rect_leftline),
            MoveAlongPath(rt_int_segment, amp_rect_rightline),
        )        
        lf_int_text.remove_updater(lf_intseg_updater)
        rt_int_text.remove_updater(rt_intseg_updater)
        self.play(lf_int_text.animate.next_to(lf_int_segment, LEFT, buff=.15),
                rt_int_text.animate.next_to(rt_int_segment, RIGHT, buff=.15))
        lfrt_final_text_str = lf_int_text.get_tex_string() + " = "+ rt_int_text.get_tex_string() + " = 0"
        lfrt_final_text = MathTex(
            lfrt_final_text_str).scale(.6).next_to(cur_inc_0_txt, DOWN, buff=.15).shift(LEFT)
        # add zero to the left and right line integral
        self.play(TransformMatchingTex(
            VGroup(lf_int_text, rt_int_text), lfrt_final_text
        ))
        self.remove(lf_int_segment, rt_int_segment)

        # use circles of magnetic fields to show that the left and right integral cancels out
        
        # top circle is turning counter-clockwise
        TIP_CNT = 8
        lf_circ = TipedCircle(solenoid.radius / 2, TIP_CNT, True, WHITE, color=WHITE).move_to(
            np.array([-solenoid.zlen/2, solenoid.radius, 0])
        )
        rt_circ = TipedCircle(solenoid.radius / 2, TIP_CNT, True, WHITE, color=WHITE).move_to(
            np.array([-solenoid.zlen/2, -solenoid.radius, 0])
        ).next_to(lf_circ, RIGHT, buff=.15)
        mag_circs = VGroup(lf_circ, rt_circ)
        self.play(Create(mag_circs))
        circs_dis = rt_circ.get_center() - lf_circ.get_center()
        move_len = solenoid.zlen - circs_dis
        self.play(mag_circs.animate.shift(RIGHT * move_len), run_time=3)
        self.play(FadeOut(mag_circs))
        self.wait(1)        
        
        # show the integration of top and bottom wire
        up_int_text.add_updater(up_intseg_udpater)
        self.add(up_int_text)
        self.play(MoveAlongPath(int_segment, amp_rect_upline), run_time=2)
        up_int_text.remove_updater(up_intseg_udpater)
        self.play(up_int_text.animate.next_to(int_segment, LEFT, buff=.15))
        self.add(dn_int_text)
        dn_int_text.add_updater(dn_intseg_updater)
        self.play(MoveAlongPath(int_segment, amp_rect_downline), run_time=2)
        dn_int_text.remove_updater(dn_intseg_updater)
        self.play(dn_int_text.animate.next_to(int_segment, RIGHT, buff=.15))
        dn_int_text.remove(dn_intseg_updater)
        self.remove(int_segment)
        self.wait(1)
    
        self.play(Indicate(cur_inc_0_txt), Indicate(lfrt_final_text))
        updn_final_text = MathTex(
            up_int_text.get_tex_string() + " + " + dn_int_text.get_tex_string() + " = 0"
        ).scale(.6).next_to(lfrt_final_text, DOWN, buff=.5)
        self.play(TransformMatchingTex(
            VGroup(up_int_text, dn_int_text), updn_final_text
        ))
        self.wait(1)
        
        # show that we can move the amp_rect around to proove the uniformity of the mag field inside the solenoid
        # first try to move the bottom of the amp_rect
        self.play(amp_rect.animate.stretch_to_fit_height(amp_rect.height / 2, about_edge=UP))
        self.wait(1)
        self.play(amp_rect.animate.shift(DOWN * amp_rect.height).scale(.5))
        # let the rectangle move in eliptical path to show that the magnetic field is uniform everywhere inside the solenoid
        eliptical_path = Ellipse(width = solenoid.zlen * .7, height = solenoid.radius * 2 * .7)
        self.play(amp_rect.animate, amp_rect.animate.move_to(eliptical_path.get_start()))
        self.play(MoveAlongPath(amp_rect, eliptical_path), run_time=4)
        self.wait(1)
        
        # go back to the original perspective and show the solenoid again
        cam_perspective_change_anim = AnimationGroup(
            cam_trackers["phi"].animate.set_value(60 * DEGREES),
            cam_trackers["theta"].animate.set_value((45 + 180)* DEGREES),
            run_time=5
        )
        
        sol_create = Create(solenoid)
        
        twoDscenes_fadeout_anims = AnimationGroup(
            FadeOut(amp_rect),
            FadeOut(updn_final_text),
            FadeOut(lfrt_final_text),
            FadeOut(cur_inc_0_txt),
            FadeOut(inouts),
            run_time=2,
            lag_ratio=.05
        )
        
        self.play(
            AnimationGroup(
                cam_perspective_change_anim,
                twoDscenes_fadeout_anims,
                sol_create,
                electron_moving_anim,
                lag_ratio=.25
            )
        )
        self.remove(*electron_dots)
        self.wait(1)

    
class SolenoidInOutTest(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        xl, yl, zl = axes.get_axis_labels(), axes.get_y_axis_label(
            "y"), axes.get_z_axis_label("z") 
        self.add(axes, xl, yl, zl)
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)
        self.move_camera(zoom=.6)
        solenoid = Solenoid(2, 10, 20, color=WHITE)
        self.play(Create(solenoid))
        self.wait(1)
        inouts = solenoid.get_inout_symbols(.2, PI/2, False, color=WHITE)
        sol_with_sign = VGroup(solenoid, inouts)
        self.play(
            FadeIn(inouts)
        )
        self.wait(1)
        self.play(
            sol_with_sign.animate.rotate(90*DEGREES, axis=UP),
            run_time=2
        )
        self.wait(1)

class RectTest(Scene):
    def construct(self):
        rect = Rectangle(height=2, width=4, color=WHITE)
        self.play(Create(rect))
        self.play(rect.animate.stretch_to_fit_height(4, about_edge=DOWN))
        self.wait(1)

class ArrowCircleTest(Scene):
    def construct(self):
        circ = Circle(radius=2, color=WHITE)
        TIP_CNT = 5
        ang_offset = 2*PI/TIP_CNT
        tips = []
        for i in range(TIP_CNT):
            tip_angle = i*ang_offset + PI/2
            tip_point = vec_by_polar(2, i * ang_offset)
            tip = Triangle()
            tip.rotate(30 * DEGREES)
            tip.rotate(tip_angle).scale(circ.stroke_width / 40).move_to(tip_point)
            tips.append(tip)
        arrow_circ = VGroup(circ, *tips)
        self.play(Create(arrow_circ))
        self.wait(1)
        
        tipcirc = TipedCircle(3, 6, True, WHITE, color=WHITE)
        self.play(Create(tipcirc))
        tip_cw = TipedCircle(4, 7, False, BLUE, color=WHITE)
        self.play(Create(tip_cw))
        self.wait(1)
        
class BugTest(ThreeDScene):
    def construct(self):
        field = ArrowVectorField(lambda p: np.array(
            [0, 0, 1.0]), x_range=[-4, 4], y_range=[-4, 4])
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)
        self.play(SpiralIn(field), run_time=2)
        self.wait(1)

