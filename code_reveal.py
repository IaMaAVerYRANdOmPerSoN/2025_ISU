# type: ignore
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

config.max_files_cached = 500

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
            self.play(Unwrite(title), run_time=2)
            self.play(Unwrite(description), run_time=2)

        self.wait(3)

        with open("Main.py", 'r') as myfile:
            chunks = []
            lines = myfile.readlines()
            chunk_size = 40
            for i in range(0, len(lines), chunk_size):
                chunks.append("".join(lines[i:i + chunk_size]))

            code_mobject = Code(
                code_string= "",
                tab_width=4,
                formatter_style="monokai",
                background="rectangle",
                language="python",
                background_config={"stroke_color": WHITE}
            ).scale(0.25)

            file_name = Text("Main.py", font_size=36).to_edge(UP, buff=0.5).shift(LEFT*3.5)
            self.play(Write(file_name), run_time=2)
            self.play(Unwrite(file_name), run_time=2)

            self.add(code_mobject)

            for ind, i in enumerate(chunks):
                self.play(code_mobject.animate.become(
                    Code(
                        code_string= i,
                        tab_width=4,
                        formatter_style="monokai",
                        line_numbers_from= ind*40 +1,
                        background="window",
                        language="python",
                        background_config={"stroke_color": WHITE}
                    ).scale(0.4)
                ), run_time=3)
                self.wait(10)

            self.play(Unwrite(code_mobject), run_time=2)
            

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> part 2 >>>>>>>>
        with open("Mandelbrot.py", 'r') as myfile:
            chunks = []
            lines = myfile.readlines()
            chunk_size = 40
            for i in range(0, len(lines), chunk_size):
                    chunks.append("".join(lines[i:i + chunk_size]))

            code_mobject_2 = Code(
                code_string= "",
                tab_width=4,
                formatter_style="monokai",
                background="rectangle",
                language="python",
                background_config={"stroke_color": WHITE}
            ).move_to(ORIGIN).scale_to_fit_height(ScreenRectangle().height + 1).scale_to_fit_width(
                 ScreenRectangle().width + 1)
            
            file_name = Text("Mandelbrot.py", font_size=36).to_edge(UP, buff=0.5).shift(LEFT*3.5)
            self.play(Write(file_name), run_time=2)
            self.play(Unwrite(file_name), run_time=2)

            self.add(code_mobject_2)


            for ind, i in enumerate(chunks):
                self.play(code_mobject_2.animate.become(
                    Code(
                        code_string= i,
                        tab_width=4,
                        formatter_style="monokai",
                        line_numbers_from= ind*40 +1,
                        background="window",
                        language="python",
                        background_config={"stroke_color": WHITE}
                    ).move_to(ORIGIN).scale_to_fit_height(ScreenRectangle().height + 1).scale_to_fit_width(
                         ScreenRectangle().width + 1)
                ), run_time=3)
                self.wait(10)

            self.play(Unwrite(code_mobject_2), run_time=2)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> part 3 >>>>>>>>>>>
        self.play(Write(bragging := Text("Just the code reveal program itself is 190 lines",
                                          font_size = 36).to_edge(UP, buff = 0.5)), run_time=2)
        self.play(Unwrite(bragging), run_time=2)

        with open(__file__, 'r') as myfile:
            chunks = []
            lines = myfile.readlines()
            chunk_size = 40
            for i in range(0, len(lines), chunk_size):
                    chunks.append("".join(lines[i:i + chunk_size]))

            code_mobject_3 = Code(
                code_string= "",
                tab_width=4,
                formatter_style="monokai",
                background="rectangle",
                language="python",
                background_config={"stroke_color": WHITE}
            ).move_to(ORIGIN).scale_to_fit_height(ScreenRectangle().height + 1).scale_to_fit_width(
                 ScreenRectangle().width + 1)

            self.add(code_mobject_3)

            for ind, i in enumerate(chunks):
                self.play(code_mobject_3.animate.become(
                    Code(
                        code_string= i,
                        tab_width=4,
                        formatter_style="monokai",
                        line_numbers_from= ind*40 +1,
                        background="window",
                        language="python",
                        background_config={"stroke_color": WHITE}
                    ).move_to(ORIGIN).scale_to_fit_height(
                         ScreenRectangle().height + 1).scale_to_fit_width(ScreenRectangle().width + 1)
                ), run_time=3)
                self.wait(10)

            self.play(Unwrite(code_mobject_3), run_time=2)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> part 4 >>>>>>>>>>>>>>

        with open(r"C:\Users\wenmi\OneDrive\Documents\GitHub\2025_ISU\GitHub-Logos\GitHub-Logos\GitHub_Logo_White.png", 'r') as logo:
            with open(r"C:\Users\wenmi\OneDrive\Documents\GitHub\2025_ISU\github-mark\github-mark-white.png", 'r') as logo_2:
                logo_mobject = ImageMobject(logo.name).scale(0.5)
                logo_mobject_2 = ImageMobject(logo_2.name).scale(0.5).next_to(logo_mobject, UP, buff=0.1)
                link = MarkupText("<i>github.com/IaMaAVerYRANdOmPerSoN/2025_ISU</i>", font_size=30
                                  ).set_color(BLUE).next_to(logo_mobject, DOWN, buff=0.1)
                link_Underline = Underline(link, buff=0.05).set_color(BLUE)

                self.play(FadeIn(logo_mobject, logo_mobject_2), run_time=2)
                self.play(Write(link),  Write(link_Underline), run_time=2)
                self.wait(3)

                self.play(FadeOut(logo_mobject, logo_mobject_2), Unwrite(link), Unwrite(link_Underline), run_time=2)

        self.play(Write(creds := Text("Made with:", font_size=48).to_edge(UP, buff=0.5)))
        self.play(Unwrite(creds))

        banner = ManimBanner()
        self.play(banner.create())
        self.play(banner.expand())
        self.wait(2)
        self.play(Unwrite(banner))

        numpy_logo = ImageMobject(r"C:\Users\wenmi\OneDrive\Documents\GitHub\2025_ISU\numpylogo.png")

        self.play(FadeIn(numpy_logo), run_time=2)
        self.play(FadeOut(numpy_logo), run_time=2)

        matplotlib_logo = ImageMobject(r"C:\Users\wenmi\OneDrive\Documents\GitHub\2025_ISU\sphx_glr_logos2_003_2_00x.webp")

        self.play(FadeIn(matplotlib_logo), run_time=2)
        self.play(FadeOut(matplotlib_logo), run_time=2)

        vscode_logo = ImageMobject(r"C:\Users\wenmi\OneDrive\Documents\GitHub\2025_ISU\Visual_Studio_Code_1.35_icon.svg.png").scale(0.5)

        self.play(FadeIn(vscode_logo), run_time=2)
        self.play(FadeOut(vscode_logo), run_time=2)