from manim import *
import sys
sys.path.append(".\\manim-physics\\")
print(sys.path)
from manim_physics import *


class solenoid_magfield(ThreeDScene):
    def solenoid_wire(radius, zlen, coil_cnt):
        coil_height = zlen / coil_cnt
        return ParametricFunction(
            lambda t: np.array([
                radius * np.cos(2 * PI * t),
                radius * np.sin(2 * PI * t),
                t * coil_height
            ]),
            t_range=[-coil_cnt / 2, coil_cnt / 2],
            color=YELLOW,
        )

    def construct(self):
        self.set_camera_orientation(theta=45 * DEGREES, phi=60 * DEGREES)
        # self.wait(1)
        solenoid = solenoid_magfield.solenoid_wire(
            1, 10, 20).rotate(PI / 2, RIGHT)
        wire = Wire(solenoid, samples=256)
        mag_field = MagneticField(wire,
                                  x_range=[-4, 4],
                                  y_range=[-4, 4],
                                  z_range=[-4, 4],
                                  three_dimensions = True
                                  )

        self.add(wire, mag_field)


class solenoid_cross_section_magfield(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(theta=45 * DEGREES, phi=60 * DEGREES)
        wire_in_lines = []
        wire_out_lines = []
        wire_ins = []
        wire_outs = []
        vfield_funcs = []
        
        WIRE_CNT = 15
        HEI = self.camera.frame_height
        WID = self.camera.frame_width
        # fill the frame with wires from left to right
        OFFSET = WID / WIRE_CNT
        print("OFFSET: ", OFFSET)
        TOP_WIRE_Y = HEI / 3
        print("TOP_WIRE_Y: ", TOP_WIRE_Y)
        BOT_WIRE_Y = -HEI / 3
        print("BOT_WIRE_Y: ", BOT_WIRE_Y)
        for i in range(WIRE_CNT):
            st_up = np.array([-WID / 2 + OFFSET * i, TOP_WIRE_Y, -HEI / 2])
            ed_up = np.array([-WID / 2 + OFFSET * i, TOP_WIRE_Y, HEI / 2])
            st_dn = st_up.copy()
            st_dn[1] = BOT_WIRE_Y
            ed_dn = ed_up.copy()
            ed_dn[1] = BOT_WIRE_Y
            up_line = Line(start=st_up, end=ed_up, color=RED)
            dn_line = Line(start=st_dn, end=ed_dn, color=GREEN)
            wire_in_lines.append(up_line)
            wire_out_lines.append(dn_line)
            wire_ins.append(Wire(up_line, current=1))
            wire_outs.append(Wire(dn_line, current=1))
        self.play(*[Create(wire_in_line) for wire_in_line in wire_in_lines])
        self.play(*[Create(wire_out_line) for wire_out_line in wire_out_lines])
        self.wait(1)
        xrg = [-WID / 2, WID / 2]
        yrg = [-HEI / 2, HEI / 2]
        for wire_in in wire_ins:
            mag_field = MagneticField(wire_in, x_range=xrg, y_range=yrg)
            vfield_funcs.append(mag_field.func)
        for wire_out in wire_outs:
            mag_field = MagneticField(wire_out, x_range=xrg, y_range=yrg)
            vfield_funcs.append(mag_field.func)
        def total_field_func(p):
            total_field = np.array([.0,.0,.0])
            for vfield_func in vfield_funcs:
                total_field += vfield_func(p)
            return total_field

        self.play(Create(ArrowVectorField(total_field_func, x_range=xrg, y_range=yrg)))


sys.path.append(".\\manim-physics\\")
print(sys.path)


class solenoid_cross_section_magfield2(Scene):
    def construct(self):
        wire_into_arr = []
        wire_out_arr = []
        WIRE_CNT = 1
        HEI = self.camera.frame_height
        WID = self.camera.frame_width
        # fill the frame with wires from left to right
        OFFSET = WID / WIRE_CNT
        TOP_WIRE_Y = HEI / 3
        BOT_WIRE_Y = -HEI / 3
        for i in range(WIRE_CNT):
            st_up = np.array([-WID / 2 + OFFSET * i, TOP_WIRE_Y, -HEI / 2])
            ed_up = np.array([-WID / 2 + OFFSET * i, TOP_WIRE_Y, HEI / 2])
            st_dn = st_up.copy()
            st_dn[1] = BOT_WIRE_Y
            ed_dn = ed_up.copy()
            ed_dn[1] = BOT_WIRE_Y
            up_line = Line(st_up, ed_up)
            dn_line = Line(st_dn, ed_dn)
            wire_into_arr.append(Wire(up_line,current=1))
            wire_out_arr.append(Wire(dn_line,current=-1))

        xrg = [-WID / 2, WID / 2]
        yrg = [-HEI / 2, HEI / 2]
        into_mag_field_arr = MagneticField(
            *wire_into_arr, x_range=xrg, y_range=yrg)
        out_mag_field_arr = MagneticField(
            *wire_out_arr, x_range=xrg, y_range=yrg)

        self.add(*wire_into_arr, *wire_out_arr,
                 into_mag_field_arr, out_mag_field_arr)
