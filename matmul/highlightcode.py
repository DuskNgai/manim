from manim import *

class HighLightCode(VGroup):
    def __init__(self,
        code,
        highlight_color=GREEN,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.code = Code(
            code=code,
            tab_width=4,
            line_spacing=0.5,
            font_size=12,
            font="Menlo",
            margin=0.2,
            line_no_buff=0.2,
            style=Code.styles_list[42],
            language="cuda"
        )
        self.add(self.code)

        self.box = None
        self.box_buff = 0.05
        self.box_width = self.code.width - 2.0 * (self.code.margin - self.box_buff)
        self.box_height = (self.code.height - 2.0 * (self.code.margin - self.box_buff)) / len(self.code.code_json)

        self.highlight_color = highlight_color

    def create_box(self, start: int, end: int) -> Rectangle:
        return RoundedRectangle(
            color=self.highlight_color,
            width=self.box_width,
            height=self.box_height * (end - start),
            corner_radius=0.05
        ).move_to(self.code.get_center()).move_to(self.code.line_numbers[start:end], coor_mask=[0, 1, 0])

    def highlight(self, start: int, end: int | None = None) -> Animation:
        if end is None:
            end = start + 1

        box = self.create_box(start, end)
        if self.box is None:
            self.box = box
            return Create(box)
        else:
            return Transform(self.box, box)

    def lowlight(self) -> Animation:
        if self.box is not None:
            box = self.box
            self.box = None
            return Uncreate(box)
