from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class code_reveal(VoiceoverScene):
    def construct(self):
        # Set up speech service
        self.set_speech_service(GTTSService(lang="en", tld="com"))

        # Title
        title = Text("Code Reveal", font_size=48).to_edge(UP)
        with self.voiceover(text="Code Reveal") as tracker:
            self.play(Write(title), run_time=tracker.duration)

        # Description
        description = Text(
            "Here is the code used to generate the Mandelbrot Set and Julia Sets.",
            font_size=30,
        ).next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="Here is the code used to generate the Mandelbrot Set and Julia Sets.") as tracker:
            self.play(Write(description), run_time=tracker.duration)

        self.wait(3)

        # Code
        Main_py = Code(
            code_file= "Main.py",
            tab_width=4,
            formatter_style="monokai",
            background="rectangle",
            language="python",
            background_config={"stroke_color": WHITE}
        )
        
        Mandelbrot_py = Code(
            code_file= "Mandelbrot.py",
            tab_width=4,
            formatter_style="monokai",
            background="rectangle",
            language="python",
            background_config={"stroke_color": WHITE}
        )
        self.play(Write(Mandelbrot_py), run_time=3)

        '''backgroud_code = Code(
            code_string= " ",
            background="window"
        )'''

        self.play(Write(Main_py), run_time=3)