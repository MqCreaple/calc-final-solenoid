from manim import *
from manim_physics import *
import numpy as np


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
        for i, arr in enumerate(to_pt_arrs):
            if i != 2:
                self.play(FadeOut(arr), run_time=.3)
        self.wait(1)

        rvec_text = MathTex(r"\vec r").move_to(
            to_pt_arrs[2].get_center() + UP * .5).scale(1.5)
        # change rvec to a 2d version of arrow
        rvec = Arrow(start=ORIGIN, end=pt_locs[-1], color=BLUE)
        self.play(Transform(to_pt_arrs[2], rvec), run_time=1)
        self.remove(to_pt_arrs[2])
        self.add(rvec)
        self.play(Create(rvec_text), run_time=.5)

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
                                 current_vec_vg_50, current_vec_vg_20, xy_plane, cvec_with_annote) # objects in the original coordinate system
        trans_vec = np.array([-1, 0, 0]) # vector of translation
        trans_vg = VGroup(orig_coord_objs, circ, diameter_line, rvec, rvec_text)
        trans_vg.generate_target()
        trans_vg.target.be
        self.move_camera(phi=0, theta=0, zoom=.4, run_time=1.5)

        # now turn the original stack of wire plus its coordinate, add \vec c and y^\prime
        
        ROTATW_TOT_TM = 8.0
        ROT_DIVS = 10
        UNIT_TM = ROTATW_TOT_TM / ROT_DIVS
        UNIT_ANG = (2 * PI) / ROT_DIVS
        for ang in np.linspace(PI - UNIT_ANG, -PI, ROT_DIVS):
            orig_coord_rot_anim = Rotate(
                orig_coord_objs, angle=-UNIT_ANG, about_point=CIRC_CENT, axis=OUT, rate_func=linear)

            # change of rvec
            vecto_circum = CIRC_R * np.array([np.cos(ang), np.sin(ang), 0])
            circum_coord = CIRC_CENT + vecto_circum
            print(circum_coord,  vecto_circum)
            new_rvec = Arrow(start=circum_coord,
                             end=init_pt.get_center(), color=BLUE)
            # move the text to the mid point of that rvec
            
            rvec_val = init_pt.get_center() - circum_coord
            rvec_unit_tan = np.array([-rvec_val[1], rvec_val[0], 0]) / np.linalg.norm(rvec_val)
            new_rvec_text_pos = circum_coord + (init_pt.get_center() - circum_coord) / 2 + rvec_unit_tan * .3
            rvec.generate_target()
            rvec.target.become(new_rvec)
            rvec_text.generate_target()
            rvec_text.target.become(rvec_text.copy().move_to(new_rvec_text_pos))
            self.play(MoveToTarget(rvec, rate_fun=linear), MoveToTarget(rvec_text),
                      orig_coord_rot_anim, run_time=UNIT_TM)
        
        self.wait(1)