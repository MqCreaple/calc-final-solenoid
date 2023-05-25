from manim import *
from manim_physics import *
import numpy as np
class CoordTransFromStackToCirc(ThreeDScene):
    def rotate_to_face_camera(self, *objs: Mobject):
        for obj in objs:
             yield obj.rotate(self.camera.get_phi(), axis = RIGHT).rotate(90 * DEGREES + self.camera.get_theta(), axis = OUT)
    
    def construct(self):
        STACK_HEIGHT = 3.0
        # demonstrate the scene where wire stacking together in the x direction coordinate
        axes = ThreeDAxes(x_range=[-5, 5], y_range=[-5, 5], z_range=[-5, 5])
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")
        z_label = axes.get_z_axis_label("z", rotation=-PI/2)
        self.add(axes)
        self.add(x_label, y_label, z_label)
        self.set_camera_orientation(phi=65 * DEGREES, theta=45 * DEGREES)
        current_vecs_20 = []; current_vecs_50 = [] 
        current_vec_vg_20 = VGroup(); current_vec_vg_50 = VGroup()
        wire_stack_text = Text("stack of wire segments").move_to([0, 0, 4.5]).next_to(axes, OUT).scale(.4)
        self.add(*self.rotate_to_face_camera(wire_stack_text))
        for i in np.linspace(-STACK_HEIGHT, STACK_HEIGHT, 20):
            current_vecs_20.append(Vector(direction=[.3, 0, 0], color=YELLOW).move_to([0, 0, i]))
        for i in np.linspace(-STACK_HEIGHT, STACK_HEIGHT, 50):
            current_vecs_50.append(Vector(direction=[.3, 0, 0], color=YELLOW).move_to([0, 0, i]))
        current_vec_vg_20.add(*current_vecs_20)
        current_vec_vg_50.add(*current_vecs_50)
        self.play(Transform(current_vec_vg_20, current_vec_vg_50), run_time=2)
        self.wait(1)
        
        # put on grids on the xy plane, put some points there and show how this stack of wires generate mag fields
        xy_plane = NumberPlane(x_range=[-5, 5], y_range=[-5, 5], background_line_style={
            "stroke_opacity": .6,
            "stroke_color" : BLUE_D
        })
        self.play(Create(xy_plane), run_time=1)
        self.wait(1)
        
        # put one point and connect vectors from the stack to the point
        pt_locs = np.array([
            [1, 2],
            [2, 1],
            [-1, 2],
            [0, 3]
        ])
        print(pt_locs)
        pt_locs = np.concatenate((pt_locs, np.zeros((4, 1))), axis=1) # since z = 0
        init_pts = Dot3D(pt_locs[0], color=RED)
        self.play(Create(init_pts), run_time=1)
        TO_PT_VEC_CNT = 5
        to_pt_arrs : list[Arrow3D] = [] # arrows from different hights on the stack to the point
        for i in np.linspace(-STACK_HEIGHT, STACK_HEIGHT, TO_PT_VEC_CNT):
            to_pt_arrs.append(Arrow3D(start=[0, 0, i], end=pt_locs[0], color=BLUE, thickness=0.02, base_radius=.05))
        to_pt_arr_vg = VGroup(*to_pt_arrs)
        self.play(Create(to_pt_arr_vg), run_time=1)
        self.wait(1)
        
        
        for i, loc in enumerate(pt_locs[1:]):
            init_pts.generate_target()
            init_pts.target.move_to(loc)
            # create animation that change the tip of the arrows to the new point
            arr_anims : list[Animation] = []
            for j, arr in enumerate(to_pt_arrs):
                new_arr = arr.copy()
                new_arr.put_start_and_end_on(arr.get_start(), loc)
                arr_anims.append(Transform(arr, new_arr))
            # play the animation of when the point moves to the new location, the arrows follow
            self.play(MoveToTarget(init_pts), *arr_anims, run_time=1)
        
            
            
                