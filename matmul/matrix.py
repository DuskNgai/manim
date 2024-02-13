from manim import *

class MatrixObject(Rectangle):
    def __init__(self,
        num_rows: int,
        num_cols: int,
        scaling_factor: float = 0.5,
        use_grid: bool = True
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.scaling_factor = scaling_factor

        super().__init__(
            color=BLACK,
            height=num_rows,
            width=num_cols,
            grid_xstep=1 if use_grid else None,
            grid_ystep=1 if use_grid else None
        )

        self.scale(scale_factor=self.scaling_factor)
        self.set_fill(ManimColor("#F8F8F8"), 1.0)
        self.set_stroke(BLACK, 1, family=True)
        self.set_stroke(BLACK, 6, family=False)

    def get_unit_length(self) -> float:
        return self.get_width() / self.num_cols

    def get_row_center(self, row: int) -> np.ndarray:
        return self.get_center() + (row - (self.num_rows - 1) / 2) * DOWN * self.get_unit_length()

    def get_col_center(self, col: int) -> np.ndarray:
        return self.get_center() + (col - (self.num_cols - 1) / 2) * RIGHT * self.get_unit_length()

    def get_row_col_center(self, row: int, col: int) -> np.ndarray:
        return self.get_center() + ((row - (self.num_rows - 1) / 2) * DOWN + (col - (self.num_cols - 1) / 2) * RIGHT) * self.get_unit_length()

    def get_row(self, row: int) -> "MatrixObject":
        return MatrixObject(1, self.num_cols, self.scaling_factor, False).move_to(self.get_row_center(row))

    def get_col(self, col: int) -> "MatrixObject":
        return MatrixObject(self.num_rows, 1, self.scaling_factor, False).move_to(self.get_col_center(col))

    def get_row_col(self, row: int, col: int) -> "MatrixObject":
        return MatrixObject(1, 1, self.scaling_factor, False).move_to(self.get_row_col_center(row, col))
