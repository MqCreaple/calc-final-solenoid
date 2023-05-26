from manim import *
import numpy as np


class CoordTransFromStackToCirc(ThreeDScene):
    def rotate_to_face_camera(self, *objs: Mobject):
        for obj in objs:
            yield obj.rotate(self.camera.get_phi(), axis=RIGHT).rotate(90 * DEGREES + self.camera.get_theta(), axis=OUT)

    def get_rt_vir_vec(self, vec: np.array):
        # only for 2D vectos
        return np.array([-vec[1], vec[0], 0])

    def get_lf_vir_vec(self, vec: np.array):
        # only for 2D vectos
        return np.array([vec[1], -vec[0], 0])

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
            repl_anims.append(ReplacementTransform(arr, to_pt_arrs[2]))

        self.play(*repl_anims, run_time=1.5)
        self.wait(1)

        rvec_text = MathTex(r"\vec r").move_to(
            to_pt_arrs[2].get_center() + UP * .5).scale(1.5)
        # change rvec to a 2d version of arrow
        rvec = Arrow(start=ORIGIN, end=pt_locs[-1], color=BLUE)
        annotated_rvec = VGroup(rvec, rvec_text)
        self.play(ReplacementTransform(to_pt_arrs[2], annotated_rvec), run_time=.5)
        self.remove(rvec, rvec_text)

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
            rvec_unit_tan = self.get_rt_vir_vec(
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
            yprime.get_center() + self.get_rt_vir_vec(yprime_vec_val) * .3).scale(1.5)

        self.play(Create(yprime), Write(yprime_text), run_time=1)

        # connecting the end of yprime to the end of rvec
        yprime_proj_val = final_rvec_val - yprime_vec_val
        proj_st = final_circum_coord + yprime_vec_val
        proj_ed = proj_st + yprime_proj_val
        yprime_proj = DashedLine(start=proj_st, end=proj_ed, color=TEAL)

        self.play(Create(yprime_proj), run_time=1)
        yprime_formula_str = r"y^\prime = \frac{\vec c \cdot \vec r}{|\vec c|}"
        yprime_formula = MathTex(yprime_formula_str).rotate(PI / 2).move_to(yprime_text.get_right() + RIGHT)
        self.play(Write(yprime_formula), run_time=1)
        self.wait(1)