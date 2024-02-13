import subprocess as sp

from manim import *

from matrix import MatrixObject
from constants import *

class NaiveGEMMScene(Scene):
    def construct(self):
        self.create()
        self.show_cover()
        self.show_matrix()
        self.show_matmul()

    def create(self):
        # Title
        self.title = Text("Naive GEMM", color=BLACK, font=FONT_TITLE)

        # Matrix
        self.M, self.N, self.K = 4, 4, 4
        self.scaling_factor = 0.5

        # Spacing matrices with 1 unit.
        self.matrix = VDict(
            {
                "A": MatrixObject(self.M, self.K, self.scaling_factor).shift(LEFT * ((self.N + self.K) / 2 + 1) * self.scaling_factor),
                "B": MatrixObject(self.K, self.N, self.scaling_factor).shift(UP * ((self.M + self.K) / 2 + 1) * self.scaling_factor),
                "C": MatrixObject(self.M, self.N, self.scaling_factor),
            }
        ).center().shift(DOWN * 0.5)

        matrix_annotation = VGroup(
            MathTex("A", tex_template=TEX_GENERAL, color=BLACK).move_to(self.matrix["A"]).scale(2.0),
            MathTex("B", tex_template=TEX_GENERAL, color=BLACK).move_to(self.matrix["B"]).scale(2.0),
            MathTex("C", tex_template=TEX_GENERAL, color=BLACK).move_to(self.matrix["C"]).scale(2.0),
        )
        brace = VDict(
            {
                "height_A": Brace(self.matrix["A"], LEFT, SMALL_BUFF, color=BLACK),
                "width_A": Brace(self.matrix["A"], UP, SMALL_BUFF, color=BLACK),
                "height_B": Brace(self.matrix["B"], LEFT, SMALL_BUFF, color=BLACK),
                "width_B": Brace(self.matrix["B"], UP, SMALL_BUFF, color=BLACK),
            }
        )
        brace_annotation = VGroup(
            MathTex("M", tex_template=TEX_GENERAL, color=BLACK).next_to(brace["height_A"], LEFT, SMALL_BUFF),
            MathTex("K", tex_template=TEX_GENERAL, color=BLACK).next_to(brace["width_A"], UP, SMALL_BUFF),
            MathTex("K", tex_template=TEX_GENERAL, color=BLACK).next_to(brace["height_B"], LEFT, SMALL_BUFF),
            MathTex("N", tex_template=TEX_GENERAL, color=BLACK).next_to(brace["width_B"], UP, SMALL_BUFF),
        )
        self.annotation = VGroup(matrix_annotation, brace, brace_annotation)

        # Code
        self.code = Code(
            code=NAIVE_GEMM,
            tab_width=4,
            line_spacing=0.5,
            font_size=12,
            font="Menlo",
            margin=0.2,
            line_no_buff=0.2,
            style=Code.styles_list[42],
            language="cuda"
        )
        self.selected_row, self.selected_col = 2, 2

    def show_cover(self):
        self.play(AddTextLetterByLetter(self.title.scale(1.5), time_per_char=0.1))

    def show_matrix(self):
        self.play(self.title.animate.scale(1.0 / 1.5).to_corner(UL))
        self.play(FadeIn(self.matrix, self.annotation))
        self.wait(1)
        self.play(FadeToColor(self.annotation, GRAY_B))

    def show_matmul(self):
        self.play(
            self.matrix.animate.shift(LEFT * 3),
            self.annotation.animate.shift(LEFT * 3),
            FadeIn(self.code.shift(RIGHT * 3)),
        )

        # Matmul
        self.selected_row_rect = self.matrix["A"].get_row(self.selected_row).set_fill(RED_C, 0.5)
        self.selected_col_rect = self.matrix["B"].get_col(self.selected_col).set_fill(BLUE_C, 0.5)

        self.play(FadeIn(self.selected_row_rect, self.selected_col_rect))

        self.A_MK = self.matrix["A"].get_row_col(self.selected_row, 0).set_fill(RED_C, 1.0)
        self.B_KN = self.matrix["B"].get_row_col(0, self.selected_col).set_fill(BLUE_C, 1.0)
        self.C_MN = self.matrix["C"].get_row_col(self.selected_row, self.selected_col)
        self.play(FadeIn(self.A_MK, self.B_KN, self.C_MN))

        for k in range(self.K):
            self.play(
                self.A_MK.animate.move_to(self.matrix["A"].get_row_col(self.selected_row, k)),
                self.B_KN.animate.move_to(self.matrix["B"].get_row_col(k, self.selected_col)),
                run_time=0.5
            )
            C_MN = self.C_MN.copy().set_fill(PURPLE_C, (k + 1) / self.K)
            self.play(
                Transform(self.A_MK.copy(), C_MN, path_arc=PI / 2),
                Transform(self.B_KN.copy(), C_MN, path_arc=PI / 2),
                run_time=0.5
            )

        self.wait(1)


class OptimizedGEMMV0Scene(Scene):
    def construct(self):
        self.create()
        self.show_cover()
        self.show_matrix()
        self.show_matmul()

    def create(self):
        # Title
        self.title = Text("Optimized GEMM v0", color=BLACK, font=FONT_TITLE)
        self.subtitle = Text("Shared Memory & Tiling", color=BLACK, font=FONT_TITLE).next_to(self.title, DOWN)

        # Matrix
        self.M, self.N, self.K = 4, 4, 4
        self.scaling_factor = 0.5

        # Spacing matrices with 1 unit.
        self.matrix = VDict(
            {
                "A": MatrixObject(self.M, self.K, self.scaling_factor).shift(LEFT * ((self.N + self.K) / 2 + 1) * self.scaling_factor),
                "B": MatrixObject(self.K, self.N, self.scaling_factor).shift(UP * ((self.M + self.K) / 2 + 1) * self.scaling_factor),
                "C": MatrixObject(self.M, self.N, self.scaling_factor),
            }
        ).center().shift(DOWN * 0.5)

        matrix_annotation = VGroup(
            MathTex("A", tex_template=TEX_GENERAL, color=BLACK).move_to(self.matrix["A"]).scale(2.0),
            MathTex("B", tex_template=TEX_GENERAL, color=BLACK).move_to(self.matrix["B"]).scale(2.0),
            MathTex("C", tex_template=TEX_GENERAL, color=BLACK).move_to(self.matrix["C"]).scale(2.0),
        )
        brace = VDict(
            {
                "height_A": Brace(self.matrix["A"], LEFT, SMALL_BUFF, color=BLACK),
                "width_A": Brace(self.matrix["A"], UP, SMALL_BUFF, color=BLACK),
                "height_B": Brace(self.matrix["B"], LEFT, SMALL_BUFF, color=BLACK),
                "width_B": Brace(self.matrix["B"], UP, SMALL_BUFF, color=BLACK),
            }
        )
        brace_annotation = VGroup(
            MathTex("M", tex_template=TEX_GENERAL, color=BLACK).next_to(brace["height_A"], LEFT, SMALL_BUFF),
            MathTex("K", tex_template=TEX_GENERAL, color=BLACK).next_to(brace["width_A"], UP, SMALL_BUFF),
            MathTex("K", tex_template=TEX_GENERAL, color=BLACK).next_to(brace["height_B"], LEFT, SMALL_BUFF),
            MathTex("N", tex_template=TEX_GENERAL, color=BLACK).next_to(brace["width_B"], UP, SMALL_BUFF),
        )
        self.annotation = VGroup(matrix_annotation, brace, brace_annotation)

        # Code
        self.code = Code(
            code=OPTIMIZED_GEMM_0,
            tab_width=4,
            line_spacing=0.5,
            font_size=12,
            font="Menlo",
            margin=0.2,
            line_no_buff=0.2,
            style=Code.styles_list[42],
            language="cuda"
        )
        self.selected_row, self.selected_col = 2, 2

    def show_cover(self):
        self.play(AddTextLetterByLetter(self.title.scale(1.5), time_per_char=0.1))
        self.play(AddTextLetterByLetter(self.subtitle, time_per_char=0.1))

    def show_matrix(self):
        self.play(self.title.animate.scale(1.0 / 1.5).to_corner(UL), Uncreate(self.subtitle), rum_time=0.5)
        self.play(FadeIn(self.matrix, self.annotation))
        self.wait(1)
        self.play(FadeToColor(self.annotation, GRAY_B))

    def show_matmul(self):
        self.play(
            self.matrix.animate.shift(LEFT * 3),
            self.annotation.animate.shift(LEFT * 3),
            FadeIn(self.code.shift(RIGHT * 3)),
        )

        # Matmul
        self.selected_row_rect = self.matrix["A"].get_row(self.selected_row).set_fill(RED_C, 0.5)
        self.selected_col_rect = self.matrix["B"].get_col(self.selected_col).set_fill(BLUE_C, 0.5)

        self.play(FadeIn(self.selected_row_rect, self.selected_col_rect))

        self.A_MK = self.matrix["A"].get_row_col(self.selected_row, 0).set_fill(RED_C, 1.0)
        self.B_KN = self.matrix["B"].get_row_col(0, self.selected_col).set_fill(BLUE_C, 1.0)
        self.C_MN = self.matrix["C"].get_row_col(self.selected_row, self.selected_col)
        self.play(FadeIn(self.A_MK, self.B_KN, self.C_MN))

        for k in range(self.K):
            self.play(
                self.A_MK.animate.move_to(self.matrix["A"].get_row_col(self.selected_row, k)),
                self.B_KN.animate.move_to(self.matrix["B"].get_row_col(k, self.selected_col)),
                run_time=0.5
            )
            C_MN = self.C_MN.copy().set_fill(PURPLE_C, (k + 1) / self.K)
            self.play(
                Transform(self.A_MK.copy(), C_MN, path_arc=PI / 2),
                Transform(self.B_KN.copy(), C_MN, path_arc=PI / 2),
                run_time=0.5
            )

        self.wait(1)


if __name__ == "__main__":
    sp.run(["manim", "-qk", "matmul.py", "NaiveGEMMScene"])
    # sp.run(["manim", "-ql", "matmul.py", "OptimizedGEMMV0Scene"])
