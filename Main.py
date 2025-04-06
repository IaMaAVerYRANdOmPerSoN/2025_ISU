from Mandelbrot import *

import numpy as np
import matplotlib.cm as cm

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

# Configure Manim settings
config.frame_rate = 15
config.max_files_cached = 500

# Scene 1: Introduction to the Mandelbrot Set
class TextScene(VoiceoverScene):
    def construct(self):
        # Set up speach service
        self.set_speech_service(GTTSService(lang="en", tld="com"))

        # Title
        title = Text("What is the Mandelbrot Set?", font_size=48).to_edge(UP)
        with self.voiceover(text="What is the Mandelbrot Set?"):
            self.play(Write(title), run_time=2)

        # Description
        description = Text(
            "The Mandelbrot Set is a set of complex numbers c\n"
            "for which the function z = z^2 + c does not diverge\n"
            "when iterated from z = 0.",
            font_size=36,
            line_spacing=1.5
        ).next_to(title, DOWN, buff=0.5)
        with self.voiceover(text="The Mandelbrot Set is a set of complex numbers c for which the function z equals z squared plus c does not diverge when iterated from z equals zero."):
            self.play(Write(description), run_time=4)
        self.wait(5)

        # Transition to example iterations
        with self.voiceover(text="Now, let's look at some example iterations."):
            self.play(FadeOut(title), FadeOut(description))
        example_title = Text("Example Iterations", font_size=42).to_edge(UP)
        self.play(Write(example_title), run_time=2)

        # Example iterations for a specific complex number
        c = 0.355 + 0.355j
        text_arr = [i for i in mandelbrot_point_cloud(c, 10)]
        example_text = [
            MathTex(
            f"z_{{{i+1}}}", "=", f"({text_arr[i][0]:.2f}", 
            f"{'+' if text_arr[i][1] >= 0 else '-'}", 
            f"{abs(text_arr[i][1].imag):.2f}i)", f"{'+' if c.real >= 0 else '-'}", 
            f"({abs(c.real):.2f}", f"{'+' if c.imag >= 0 else '-'}", 
            f"{abs(c.imag):.2f}i)"
            ).set_color_by_tex_to_color_map({
            f"z_{{{i+1}}}": BLUE,
            f"({text_arr[i][0]:.2f}": YELLOW,
            f"{abs(text_arr[i][1].imag):.2f}i)": YELLOW,
            f"({abs(c.real):.2f}": RED,
            f"{abs(c.imag):.2f}i)": RED
            })
            for i in range(len(text_arr) - 1)
        ]

        # Display example iterations
        example_text_mobjects = VGroup(example_text
        ).arrange(DOWN, buff=0.5).next_to(example_title, DOWN, buff=0.5)

        with self.voiceover(text="Here are the first few iterations for a specific complex number."):
            self.play(Write(example_text_mobjects[0:5]), run_time=3)
        self.wait()

        # Animate the remaining iterations
        for i in range(5, len(example_text_mobjects)):
            self.play(FadeOut(example_text_mobjects[i-5]))
            example_text_mobjects.shift(UP)
            self.play(Write(example_text_mobjects[i]), run_time=2)
            self.wait()

        self.wait(2)
        with self.voiceover(text="If the sequence remains bounded, c is in the Mandelbrot Set."):
            self.play(FadeOut(example_text_mobjects), FadeOut(example_title))
        self.wait(2)


class MandelbrotImageExample(VoiceoverScene):
    def construct(self):
        # Set up speach service
        self.set_speech_service(GTTSService(lang="en", tld="com"))

        # Title
        title = Text("How?", font_size=48).to_edge(UP)
        with self.voiceover(text="How do you get the images?"):
            self.play(Write(title), run_time=2)

        # black-and-white colormap
        colormap = cm.get_cmap("grey")

        # Generate black-and-white Mandelbrot set image
        mandelbrot_set = mandelbrot(-2, 1, 1.5 + 1j, 3840, 2160, 256)
        normalized_set = np.log(1 + mandelbrot_set) / np.log(3) / np.log(256)
        colored_image = (colormap(normalized_set) * 255).astype(np.uint8)
        mandelbrot_mobject_mono = ImageMobject(colored_image, scale_to_resolution=2160)

        # Modern colormap
        colormap = cm.get_cmap("inferno")
        mandelbrot_set = mandelbrot(-2, 1, 1.5 + 1j, 3840, 2160, 256)
        normalized_set = np.log(1 + mandelbrot_set) / np.log(3) / np.log(256)
        colored_image = (colormap(normalized_set)[:, :, :3] * 255).astype(np.uint8)
        mandelbrot_mobject = ImageMobject(colored_image, scale_to_resolution=2160)

        # Display the descriptions and images

        with self.voiceover(text="In older images, members of the Mandelbrot set are colored black, and those not in the set are colored white, which looks like this."):
            pass

        self.play(FadeIn(mandelbrot_mobject_mono), FadeOut(title), run_time=2)

        with self.voiceover(text="Modern images use a color map to show the number of iterations before divergence, assigning a color to each point based on iterations to divergence, and coloring them black if they never diverge."):
            self.play(FadeOut(mandelbrot_mobject_mono), run_time=3)


        with self.voiceover(text="Here is an example of a modern Mandelbrot set image."):
            self.play(FadeIn(mandelbrot_mobject), run_time=3)


        self.play(FadeOut(mandelbrot_mobject), run_time=2)

# Scene 2: Visualizing a single point's orbit
class OnePointExample(VoiceoverScene):
    def construct(self):
        # Set up speech service
        self.set_speech_service(GTTSService(lang="en", tld="com"))

        # Display formula in the upper-left corner
        text = MathTex(r"{z_0}^2 + c =", r"z_1").to_corner(UL)
        self.play(Write(ComplexPlane().add_coordinates()), run_time=2)
        self.play(Write(text), run_time=2)

        # Define orbit points for a specific complex number
        orbit_points = [
            -0.1 + 0.75j,
            (-0.1 + 0.75j) ** 2 + (-0.1 + 0.75j),
            ((-0.1 + 0.75j) ** 2 + (-0.1 + 0.75j)) ** 2 + (-0.1 + 0.75j),
        ]

        # Animate the orbit points
        point = ComplexValueTracker(orbit_points[0])
        dot = Dot(
            np.array([point.get_value().real, point.get_value().imag, 0]),
            radius=0.05,
            color=WHITE,
        )
        dot.add_updater(lambda x: x.move_to(np.array([point.get_value().real, point.get_value().imag, 0])))
        self.play(FadeIn(dot), run_time=0.3)

        # Add voiceover for the explanation
        with self.voiceover(text="Let's visualize the orbit of a single point under the iteration of our function."):
            self.wait(2)

        # Iterate through the orbit points
        for _ in range(2):  # Repeat the cycle twice
            for i in range(len(orbit_points)):
                if i > 0:
                    # Draw an arrow between points
                    arrow = CurvedArrow(
                        np.array([orbit_points[i - 1].real, orbit_points[i - 1].imag, 0]),
                        np.array([orbit_points[i].real, orbit_points[i].imag, 0]),
                        color=BLUE,
                        tip_length=0.2
                    )
                    self.play(Write(arrow), run_time=0.5)
                    self.wait(0.75)
                    self.play(FadeOut(arrow), run_time=0.5)

                # Update the dot and formula text
                self.play(point.animate.set_value(orbit_points[i]))
                new_text = MathTex(
                    f"({orbit_points[i]:.2f})^2 + c = {orbit_points[i]**2 + orbit_points[i]:.2f}"
                ).to_corner(UL)
                self.play(Transform(text, new_text))

            # Complete the cycle with an arrow back to the first point
            arrow = CurvedArrow(
                np.array([orbit_points[-1].real, orbit_points[-1].imag, 0]),
                np.array([orbit_points[0].real, orbit_points[0].imag, 0]),
                color=BLUE,
                tip_length=0.2
            )
            self.play(Write(arrow), run_time=0.5)
            self.wait(0.75)
            self.play(FadeOut(arrow), run_time=0.5)
            self.play(point.animate.set_value(orbit_points[0]))

        # Cleanup
        self.play(FadeOut(text), FadeOut(dot), run_time=0.5)
        dot.clear_updaters()
        self.wait(2)

# Scene 3: Mandelbrot Set visualization
class MandelbrotScene(VoiceoverScene):
    def construct(self):
        # Set up speech service
        self.set_speech_service(GTTSService(lang="en", tld="com"))

        # Mandelbrot set parameters
        rmin, rmax = -2, 1
        cmax = 1.5 + 1j
        width, height = 3840, 2160
        maxiter = 256
        colormap = cm.get_cmap("inferno")

        # Generate Mandelbrot set image
        mandelbrot_set = mandelbrot(rmin, rmax, cmax, width, height, maxiter)
        normalized_set = np.log(1 + mandelbrot_set) / np.log(3) / np.log(maxiter)
        colored_image = (colormap(normalized_set)[:, :, :3] * 255).astype(np.uint8)
        mandelbrot_mobject = ImageMobject(colored_image, scale_to_resolution=2160)

        # Display Mandelbrot set
        with self.voiceover(text="Lets see how the orbits of a couple points appear on the full set."):
            self.add(mandelbrot_mobject)
            self.wait(5)

        # Animate points on the Mandelbrot set
        c = ComplexValueTracker(0 + 0j)
        dots = VGroup()
        lines = VGroup()
        text = MathTex(r"f_c(z)= z^2+c").to_corner(UL)

        # Update dots to represent orbit points
        dots.add_updater(lambda x: x.become(
            VGroup(*[Dot(np.array([point[0], point[1], 0]), radius=0.05, color=WHITE)
                     for point in mandelbrot_point_cloud(c.get_value(), maxiter)])))

        # Update lines to connect orbit points
        lines.add_updater(lambda x: x.become(
            VGroup(*[
                Line(
                    np.array([point[0][0], point[0][1], 0]),
                    np.array([point[1][0], point[1][1], 0]),
                    color=BLUE,
                    stroke_width=1
                )
                for point in zip(mandelbrot_point_cloud(c.get_value(), maxiter)[:-1],
                                 mandelbrot_point_cloud(c.get_value(), maxiter)[1:])
            ])
        ))

        # Update the formula text
        text.add_updater(lambda x: x.become(
            MathTex(f"f_{{{c.get_value().real:.2f} {'+' if c.get_value().imag >= 0 else '-'} {abs(c.get_value().imag):.2f}i}}(z) = z^2 {'+' if c.get_value().real >= 0 else '-'} {c.get_value().real:.2f} {'+' if c.get_value().imag >= 0 else '-'} {abs(c.get_value().imag):.2f}").to_corner(UL)
        ))

        self.add(dots, lines, text)

        # Animate transitions between different points
        points = [
            0.355 + 0.355j,
            -0.10109636384562 + 0.95628651080914j,
            0 + 1j,
            -0.1 + 0.75j,
            0 + 0j,
            -0.335 + 0.335j
        ]
        for point in points:
                self.play(c.animate.set_value(point), run_time=3)
                self.wait(2)

        # Cleanup
        self.play(FadeOut(dots, lines, text))
        self.wait(2)

class MultiplePointsExample(VoiceoverScene):
    def construct(self):
        # Set up speech service
        self.set_speech_service(GTTSService(lang="en", tld="com"))

        self.play(Write(ComplexPlane()))

        Dots = VGroup()
        lines = VGroup()
        text = MathTex(r"f_c(z) = z^2 + c")
        c = ComplexValueTracker(0 + 0j)
        
        Dots.add_updater(lambda x: x.become(VGroup(*[Dot(np.array([i[0], i[1], 0]), radius=0.02) for i in mandelbrot_point_cloud(c.get_value(), 25)])))
        lines.add_updater(lambda x: x.become(
            VGroup(*[
                Line(
                    np.array([point[0][0], point[0][1], 0]),
                    np.array([point[1][0], point[1][1], 0]),
                    color=BLUE,
                    stroke_width=1
                )
                for point in zip(mandelbrot_point_cloud(c.get_value(), 25)[:-1],
                                 mandelbrot_point_cloud(c.get_value(), 25)[1:])
            ])
        ))
        text.add_updater(lambda x: x.become(MathTex(f"f_{{c.get_value():.2f}}(z) = z^2 {'+' if c.get_value().real >= 0 else '-'} {abs(c.get_value()).real:.2f} {'+' if c.get_value().imag >= 0 else '-'} {abs(c.get_value().imag):.2f}i").to_corner(UL)))

        self.add(Dots, lines)
        with self.voiceover(text="Here is the visualization of multiple points iterating under our dynamic system, with thier orbits traced."):
            self.play(Write(text))

        points = [
            (0.335 - 0.335j, "Some points fall into stable cycles;"),
            (-0.2 - 0.3j, "Other points aproach a attracting limit point;"),
            (0.33 + 0.06j, " Some points are attracted to a repelling limit point;"),
            (0.5 + 0.5j, "And some points just shoot of to infinity;"),
            (0, "Or at some points, like the origin, the function is it's own inverse!")
        ]

        for point, narration in points:
            with self.voiceover(text=narration) as tracker:
                self.play(c.animate.set_value(point), run_time=tracker.duration)

        self.play(FadeOut(Dots, lines), run_time=3)


# Scene 4: Introduction to Julia Sets
class JuliaSetScene(VoiceoverScene):
    def construct(self):
        # Set up speech service
        self.set_speech_service(GTTSService(lang="en", tld="com"))

        # Title
        title = Text("What are Julia Sets?", font_size=48).to_edge(UP)
        with self.voiceover(text="What are Julia Sets?"):
            self.play(Write(title), run_time=2)

        # Description
        description = Tex(
            "Julia Sets are fractals generated by iterating the function\n"
            "$z = z^2 + c$ for a fixed complex number $c$, starting from\n"
            "different initial values of $z$.",
            font_size=36,
        ).next_to(title, DOWN, buff=0.5)
        with self.voiceover(text="Julia Sets are fractals generated by iterating the function z equals z squared plus c for a fixed complex number c, starting from different initial values of z."):
            self.play(Write(description), run_time=4)
        self.wait(5)

        relationship = Text(
            "The Mandelbrot Set acts as a map for Julia Sets:\n"
            "each point in the Mandelbrot Set corresponds to a\n"
            "unique Julia Set. Points inside the Mandelbrot Set\n"
            "produce connected Julia Sets, while points outside\n"
            "produce infinitely disconnected (dust-like) Julia Sets.",
            font_size=36,
            color=BLUE
        ).next_to(description, DOWN, buff=0.5)
        with self.voiceover(text="The Mandelbrot Set acts as a map for Julia Sets. Each point in the Mandelbrot Set corresponds to a unique Julia Set. Points inside the Mandelbrot Set produce connected Julia Sets, while points outside produce infinitely disconnected, dust-like Julia Sets."):
            self.play(Write(relationship), run_time=6)
        self.wait(5)

        # Transition to example
        with self.voiceover(text="Now, let's look at an example of a Julia Set."):
            self.play(FadeOut(title), FadeOut(description))
        example_title = Text("Example Julia Set", font_size=42).to_edge(UP)
        self.play(Write(example_title), run_time=2)

        # Generate initial Julia set image
        c = ComplexValueTracker(-0.8 + 0.156j)  # Example constant for Julia set
        colormap = cm.get_cmap("inferno")
        julia_set = julia(c.get_value(), 2160, 2160, 256)
        normalized_set = np.log(1 + julia_set) / np.log(3) / np.log(256)
        normalized_set = (normalized_set - normalized_set.min()) / (normalized_set.max() - normalized_set.min())
        colored_image = (colormap(normalized_set)[:, :, :3] * 255).astype(np.uint8)
        julia_mobject = ImageMobject(colored_image, scale_to_resolution=1080)

        # Add updater to update the Julia set dynamically
        def update_julia(mobject):
            julia_set = julia(c.get_value(), 2160, 2160, 256)
            normalized_set = np.log(1 + julia_set) / np.log(3) / np.log(256)
            colored_image = (colormap(normalized_set)[:, :, :3] * 255).astype(np.uint8)
            mobject.become(colored_image)

        julia_mobject.add_updater(update_julia)

        # Display Julia set
        with self.voiceover(text="Here is an example of a Julia Set for a specific value of c."):
            self.play(FadeIn(julia_mobject), run_time=2)
        with self.voiceover(text="Now, let's explore how the Julia Set changes as we vary the value of c."):
            self.play(c.animate.set_value(0.355 - 0.335j), run_time=10)
            self.play(c.animate.set_value(0.36 + 1j), run_time=10)
            self.play(c.animate.set_value(0.33 + 0.06j), run_time=10)
            self.play(c.animate.set_value(0 + 0j), run_time=10)

        # Cleanup
        julia_mobject.clear_updaters()
        self.play(FadeOut(julia_mobject), FadeOut(example_title))

class AlternateDefintionWithJulia(VoiceoverScene):
    def construct(self):
        # Set up speach service
        self.set_speech_service(GTTSService(lang="en", tld="com"))

        colormap = cm.get_cmap('inferno')

        julia_wdith, julia_height = ValueTracker(80), ValueTracker(80)

        def normalize_color(c):
            julia_set = julia(c, int(julia_wdith.get_value() // 1), int(julia_height.get_value() // 1), 256)
            normalized_set = np.log(1 + julia_set) / np.log(3) / np.log(256)
            return (colormap(normalized_set)[:, :, :3] * 255).astype(np.uint8)

        # Correctly cast width and height to integers
        width = int(40 * 80 // julia_wdith.get_value())
        height = int(30 * 80 // julia_height.get_value())

        # Create a grid of complex numbers
        real = np.linspace(-2, 1, width)
        imag = np.linspace(-1.5, 1.5, height)
        real_grid, imag_grid = np.meshgrid(real, imag)

        # Initialize the grid of ImageMobject instances
        julia_images = [
            ImageMobject(normalize_color(complex(real_grid[i, j], imag_grid[i, j])), scale_to_resolution=1080).move_to(
                np.array([real_grid[i, j], imag_grid[i, j], 0])
            )
            for i in range(real_grid.shape[0]) for j in range(real_grid.shape[1])
        ]

        # Add all images to the scene
        for image in julia_images:
            self.add(image)

        # Define an updater to update the images dynamically
        def update_julia_images():
            for i in range(real_grid.shape[0]):
                for j in range(real_grid.shape[1]):
                    index = i * real_grid.shape[1] + j
                    julia_images[index].set_image(normalize_color(complex(real_grid[i, j], imag_grid[i, j])))

        # Add the updater to the scene
        self.add_updater(lambda _: update_julia_images())

        # Display the combined image
        with self.voiceover(text="Here, we see a grid of Julia Sets, each corresponding to a different point in the complex plane."):
            pass
        with self.voiceover(text="Observe how the Image changes as we increase the resolution of the grid."):
            self.play(julia_wdith.animate.set_value(5), julia_wdith.animate.set_value(5), run_time=30, rate_func=double_smooth)
        with self.voiceover(text="Now, if we zoom in on only the center pixel of the grid, we can see how the mandelbrot set acts as a map for the Julia Sets."):
            self.play(julia_wdith.animate.set_value(1), julia_wdith.animate.set_value(1), run_time=30, rate_func=double_smooth)

        with self.voiceover(text="for a point p, if the julia set of z^2 + p is connected, then p is in the mandelbrot set."):
            self.play(FadeOut(*julia_images), run_time=2)


class Generalizations(VoiceoverScene):
    def construct(self):
        # Set up speach service
        self.set_speech_service(GTTSService(lang="en", tld="com"))
    
        # Title
        Title = Tex("Generalizations of the Mandelbrot Set", font_size=48).to_edge(UP)

        contents = Tex(
            r"\begin{enumerate}"
            r"\item \textbf{Multibrot Sets:} A generalization of the Mandelbrot set where the exponent in the iteration formula $z = z^d + c$ is replaced with a higher degree $d$."
            r"\item \textbf{Tricorn:} A fractal similar to the Mandelbrot set, but generated using the iteration formula $z = \overline{z}^2 + c$, where $\overline{z}$ is the complex conjugate of $z$."
            r"\item \textbf{4D Mandelbrot:} A higher-dimensional generalization of the Mandelbrot set, visualized by projecting 4D fractals into 3D or 2D spaces."
            r"\end{enumerate}",
            font_size=36
        ).next_to(Title, DOWN, buff=0.5)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multibrot set <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        Multibrot_title = Tex(r"1. \quad Multibrot Sets", font_size = 48).to_edge(UP)

        rmin, rmax = -2, 1
        cmax = 1.5 + 1j
        width, height = 300, 225
        maxiter = 256
        colormap = cm.get_cmap("inferno")
        exp = ValueTracker(0)
        exp_indicator = MathTex("d =", "0.00", substrings_to_isolate='d').set_color_by_tex('d', RED).to_corner(UL)
        exp_indicator.add_updater(lambda x: x[1].become(MathTex(f"d = {ValueTracker.get_value()}", color = BLUE).to_corner(UL)))
        
        # Create a blank image with dimensions 3840x2160
        from PIL import Image
        blank_image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        multibrot = ImageMobject(Image.fromarray(blank_image))

        self.add(multibrot)

        def Update_multibrot(old_multibrot):
            multibrot_set = multibrot(rmin, rmax, cmax, width, height, maxiter, exp)
            normalized_set = np.log(1 + multibrot_set) / np.log(3) / np.log(maxiter)
            colored_image = (colormap(normalized_set)[:, :, :3] * 255).astype(np.uint8)
            multibrot_mobject = ImageMobject(colored_image, scale_to_resolution=2160)
            old_multibrot.become(multibrot_mobject)

        multibrot.add_updater(Update_multibrot)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Tricorn <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        Tricorn_title = Tex(r"2. \quad Tricorn sets", font_size = 48).to_edge(UP)

        tricorn = multibrot

        def Update_tricorn(old_tricorn):
            tricorn_set = multicorn(rmin, rmax, cmax, width, height, maxiter, exp)
            normalized_set = np.log(1 + tricorn_set) / np.log(3) / np.log(maxiter)
            colored_image = (colormap(normalized_set)[:, :, :3] * 255).astype(np.uint8)
            tricorn_mobject = ImageMobject(colored_image, scale_to_resolution=2160)
            old_tricorn.become(tricorn_mobject)

        tricorn.add_updater(Update_tricorn)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 4d Mandelbrot <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        buhdabrot_projections = ['zr_zi',
                                'zr_cr',
                                'zr_ci',
                                'zi_cr',
                                'zi_ci',
                                'cr_ci']
        
        buhdabrot_R = project_histogram_4d_to_2d(histogram = buhdabrot(rmin, rmax, cmax, width, height, maxiter = 100, sample_size = int(5e8)),
        projection_plane= "zi_ci",
        width = 3840,
        height = 2160,
        )

        buhdabrot_G = project_histogram_4d_to_2d(histogram = buhdabrot(rmin, rmax, cmax, width, height, maxiter = 500, sample_size = int(5e8)),
        projection_plane= "zi_ci",
        width = 3840,
        height = 2160,
        )

        buhdabrot_B = project_histogram_4d_to_2d(histogram = buhdabrot(rmin, rmax, cmax, width, height, maxiter = 1000, sample_size = int(5e8)),
        projection_plane= "zi_ci",
        width = 3840,
        height = 2160,
        )

        normalized_R = np.log(1 + buhdabrot_R) / np.log(3) / np.log(100)
        normalized_G = np.log(1 + buhdabrot_G) / np.log(3) / np.log(500)
        normalized_B = np.log(1 + buhdabrot_B) / np.log(3) / np.log(1000)

        # Combine RGB channels into one image
        combined_image = np.stack([normalized_R, normalized_G, normalized_B], axis=-1)
        combined_image = (combined_image - combined_image.min()) / (combined_image.max() - combined_image.min())
        combined_image = (combined_image * 255).astype(np.uint8)
        buhdabrot_mobject = ImageMobject(combined_image, scale_to_resolution=2160)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> animations <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        with self.voiceover(text="In this scene, we will explore some fascinating generalizations of the Mandelbrot Set."):
            self.play(Write(Title), run_time=2)
        with self.voiceover(text="These include the Multibrot Sets, the Tricorn, and the 4D Mandelbrot Set."):
            self.play(Write(contents), run_time=4)
        self.wait(5)
        self.play(FadeOut(Title, contents))
        
        with self.voiceover(text="First, let's look at the Multibrot Sets."):
            self.play(Write(Multibrot_title))
        self.wait()
        with self.voiceover(text="The Multibrot Sets are a generalization of the Mandelbrot Set, where the exponent in the iteration formula is replaced with a diffrent degree."):
            self.play(Write(exp_indicator), FadeOut(Multibrot_title))
        with self.voiceover(text="Observe how the fractal changes as we vary the exponent."):
            self.play(exp.animate.set_value(6), runtime=15, rate_func=double_smooth)
            self.play(exp.animate.set_value(-6), runtime=20, rate_func=double_smooth)
        self.play(FadeOut(multibrot, exp_indicator))
        multibrot.clear_updaters()
        exp.set_value(0)

        with self.voiceover(text="Next, we explore the Tricorn sets."):
            self.add(tricorn)
            self.play(Write(Tricorn_title))
        self.wait()
        with self.voiceover(text="The Tricorn is generated using the iteration formula where the complex conjugate of z is squared and added to c, istead of just z squared."):
            self.play(Write(exp_indicator), FadeOut(Tricorn_title))
        with self.voiceover(text="Let's observe the changes as we vary the exponent here as well."):
            self.play(exp.animate.set_value(6), runtime=15, rate_func=double_smooth)
            self.play(exp.animate.set_value(-6), runtime=20, rate_func=double_smooth)
        self.play(FadeOut(exp_indicator, tricorn))