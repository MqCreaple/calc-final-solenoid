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

        rvec_text = MathTex(r"\vec r").move_to(
            to_pt_arrs[2].get_center() + UP * .5).scale(1.5)
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
        cvec_text = MathTex(r"\vec c").move_to(
            cvec.get_center() + UP * .4).scale(1.5)
        cvec_with_annote = VGroup(cvec, cvec_text)

        orig_coord_objs = VGroup(axes, x_label, y_label, z_label, wire_stack_text,
                                 current_vec_vg_50, current_vec_vg_20, xy_plane, cvec_with_annote)  # objects in the original coordinate system
        trans_vec = np.array([CIRC_R, 0, 0])  # vector of translation
        self.move_camera(phi=0, theta=0, zoom=.4,
                         run_time=2, frame_center=trans_vec)
        # now turn the original stack of wire plus its coordinate, add \vec c and y^\prime

        ROTATW_TOT_TM = 10.0

        ROT_DEMO_START_ANG = PI
        ROT_DEMO_END_ANG = -(PI + 35 * DEGREES)
        ROT_DEMO_ANG_OFFSET = ROT_DEMO_END_ANG - ROT_DEMO_START_ANG

        # # remove the original one since there will be update animation
        # self.play(FadeOut(annotated_rvec))
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
            new_rvec_text = rvec_text.copy().move_to(new_rvec_text_pos)
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
        yprime_text = MathTex(r"y^\prime").move_to(
            yprime.get_center() + get_rt_vir_vec(yprime_vec_val) * .3).scale(1.5)

        self.play(Create(yprime), Write(yprime_text), run_time=1)

        # connecting the end of yprime to the end of rvec
        yprime_proj_val = final_rvec_val - yprime_vec_val
        proj_st = final_circum_coord + yprime_vec_val
        proj_ed = proj_st + yprime_proj_val
        yprime_proj = DashedLine(start=proj_st, end=proj_ed, color=TEAL)

        self.play(Create(yprime_proj), run_time=1)
        yprime_formula_str = r"y^\prime = \frac{\vec c \cdot \vec r}{|\vec c|}"
        yprime_formula = MathTex(yprime_formula_str).rotate(
            PI / 2).move_to(yprime_text.get_right() + RIGHT)
        self.play(Write(yprime_formula), run_time=1)
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
        self.play(
            integration_val.animate.become(MathTex(
                r"\frac{\oint \vec B \cdot \mathrm d\vec l}{\mu_0} = 0").to_corner(UR)),

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

        sol_wire = Wire(solenoid, current=2)
        self.play(Create(solenoid))

        DOT_CNT = 25
        DOT_TRAVEL_MAX_TM = 8  # seconds
        electron_dots = [Dot3D(solenoid.get_start(), color=BLUE)
                         for _ in range(DOT_CNT)]
        electron_moving_anim = LaggedStart(
            *[MoveAlongPath(electron_dots[i], solenoid)
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
        dn_int_text = MathTex(r"\oint \vec{B} \cdot \mathrm d\vec{l} \ (\text{up})").scale(.6).next_to(
            amp_rect_downline.get_start(), DOWN, buff=.15)
        lf_int_text = MathTex(r"\oint \vec{B} \cdot \mathrm d\vec{l} \ (\text{up})").scale(.6).next_to(
            amp_rect_leftline.get_start(), LEFT, buff=.15)
        rt_int_text = MathTex(r"\oint \vec{B} \cdot \mathrm d\vec{l} \ (\text{up})").scale(.6).next_to(
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
            lfrt_final_text_str).scale(.6).next_to(cur_inc_0_txt, DOWN, buff=.15).shift(LEFT * .5)
        # add zero to the left and right line integral
        self.play(Transform(
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
        self.play(mag_circs.animate.shift(RIGHT * move_len), run_time=1.5)
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
