import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_point_cloud(c, maxiter):
    """
    Generates a point cloud representing the iterations of the Mandelbrot set for a given complex number.
    Parameters:
    c (complex): The complex number to evaluate in the Mandelbrot set.
    maxiter (int): The maximum number of iterations to perform.
    Returns:
    numpy.ndarray: A 2D array of shape (maxiter, 2), where each row contains the real and imaginary 
                   parts of the complex number at each iteration.
    """
    point_cloud = np.zeros((maxiter, 2))  # Preallocate array with zeros
    z = complex(0, 0)
    
    for i in range(maxiter):
        point_cloud[i] = [z.real, z.imag]
        z = z * z + c
    return point_cloud

def mandelbrot(rmin, rmax, cmax, width, height, maxiter):
    """
    Generate a Mandelbrot set image.
    This function computes the Mandelbrot set for a given range of real and imaginary values,
    and returns a 2D array representing the fractal image. The computation is optimized by
    skipping points inside the main cardioid and the period-2 bulb.
    Parameters:
        rmin (float): The minimum value of the real axis.
        rmax (float): The maximum value of the real axis.
        cmax (complex): The maximum value of the imaginary axis (only the imaginary part is used).
        width (int): The width of the output image in pixels.
        height (int): The height of the output image in pixels. If odd, it will be adjusted to the next even number.
        maxiter (int): The maximum number of iterations to determine divergence.
    Returns:
        numpy.ndarray: A 2D array of shape (height, width) containing the Mandelbrot set.
                       Each value represents the iteration count at which the point diverged,
                       or 0 if the point is in the set.
    """
    
    # Ensure height is even for symmetry
    if height % 2 != 0:
        height += 1

    real = np.linspace(rmin, rmax, width, dtype=np.float32)
    imag = np.linspace(0, cmax.imag, height // 2, dtype=np.float32)
    c_real, c_imag = np.meshgrid(real, imag)

    output = np.zeros((height // 2, width), dtype=np.uint16)
    z_real = np.zeros_like(c_real)
    z_imag = np.zeros_like(c_imag)
    mask = np.ones_like(c_real, dtype=bool)

    # Bulb checking
    p = np.sqrt((c_real - 0.25) ** 2 + c_imag ** 2)
    cardioid = c_real < p - 2 * p ** 2 + 0.25
    period2_bulb = (c_real + 1) ** 2 + c_imag ** 2 < 0.0625
    mask[cardioid | period2_bulb] = False
    output[~mask] = maxiter - 1

    for i in range(maxiter):
        zr2 = z_real[mask] ** 2
        zi2 = z_imag[mask] ** 2

        z_imag_new = 2 * z_real[mask] * z_imag[mask] + c_imag[mask]
        z_real[mask] = zr2 - zi2 + c_real[mask]
        z_imag[mask] = z_imag_new

        diverged = zr2 + zi2 >= 4.0
        output[mask] = i
        mask[mask] = ~diverged

    output[output == maxiter - 1] = 0

    # Mirror the top half to the bottom half
    full_output = np.vstack([np.flipud(output), np.flipud(output[::-1, :])])
    return full_output


"""width, height, maxiter = 12000, 9000, 100
cmax = complex(0, 1)
mandelbrot_set = mandelbrot(cmax, width, height, maxiter)

plt.figure(figsize=(16, 12))
plt.imshow(mandelbrot_set, extent=[-2, 0.5, -1, 1], cmap="turbo", interpolation="bilinear")
plt.colorbar(label="Iterations to Divergence")
plt.title("Mandelbrot Set")
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.show()"""

def mandelbrot_from_center(centerpoint, zoom, width, height, maxiter):
    """
    Generate the Mandelbrot set with a given centerpoint and zoom level.

    Parameters:
        centerpoint: Complex number representing the center of the viewing field.
        zoom: Float representing the zoom level (higher values zoom in).
        width: Integer, width of the output image.
        height: Integer, height of the output image.
        maxiter: Integer, maximum number of iterations.

    Returns:
        A 2D array representing the Mandelbrot set.
    """
    # Ensure height is even for symmetry
    if height % 2 != 0:
        height += 1

    # Calculate the viewing field based on the centerpoint and zoom
    aspect_ratio = width / height
    view_height = 4 / zoom  # Default height of the viewing field is 4, scaled by zoom
    view_width = view_height * aspect_ratio

    rmin = centerpoint.real - view_width / 2
    rmax = centerpoint.real + view_width / 2
    cmin = centerpoint.imag
    cmax = centerpoint.imag + view_height / 2

    # Generate the grid of complex numbers
    real = np.linspace(rmin, rmax, width, dtype=np.float32)
    imag = np.linspace(cmin, cmax, height // 2, dtype=np.float32)
    c_real, c_imag = np.meshgrid(real, imag)

    # Initialize arrays for the Mandelbrot computation
    output = np.zeros((height // 2, width), dtype=np.uint16)
    z_real = np.zeros_like(c_real)
    z_imag = np.zeros_like(c_imag)
    mask = np.ones_like(c_real, dtype=bool)

    # Bulb checking for optimization
    p = np.sqrt((c_real - 0.25) ** 2 + c_imag ** 2)
    cardioid = c_real < p - 2 * p ** 2 + 0.25
    period2_bulb = (c_real + 1) ** 2 + c_imag ** 2 < 0.0625
    mask[cardioid | period2_bulb] = False
    output[~mask] = maxiter - 1

    # Mandelbrot iteration
    for i in range(maxiter):
        zr2 = z_real[mask] ** 2
        zi2 = z_imag[mask] ** 2

        z_imag_new = 2 * z_real[mask] * z_imag[mask] + c_imag[mask]
        z_real[mask] = zr2 - zi2 + c_real[mask]
        z_imag[mask] = z_imag_new

        diverged = zr2 + zi2 >= 4.0
        output[mask] = i
        mask[mask] = ~diverged

    output[output == maxiter - 1] = 0

    # Mirror the top half to the bottom half
    full_output = np.vstack([output, np.flipud(output)])
    return full_output

def multibrot(rmin, rmax, cmax, width, height, maxiter, exponent):
    # Ensure height is even for symmetry
    if height % 2 != 0:
        height += 1

    real = np.linspace(rmin, rmax, width, dtype=np.float32)
    imag = np.linspace(0, cmax.imag, height // 2, dtype=np.float32)
    c_real, c_imag = np.meshgrid(real, imag)

    output = np.zeros((height // 2, width), dtype=np.uint16)
    z_real = np.zeros_like(c_real)
    z_imag = np.zeros_like(c_imag)
    mask = np.ones_like(c_real, dtype=bool)

    for i in range(maxiter):
        zr2 = z_real[mask] ** 2
        zi2 = z_imag[mask] ** 2

        z_imag_new = (z_real[mask] + 1j * z_imag[mask]) ** exponent
        z_real[mask] = z_imag_new.real + c_real[mask]
        z_imag[mask] = z_imag_new.imag + c_imag[mask]

        diverged = zr2 + zi2 >= 4.0
        output[mask] = i
        mask[mask] = ~diverged

    output[output == maxiter - 1] = 0

    # Mirror the top half to the bottom half
    full_output = np.vstack([np.flipud(output), np.flipud(output[::-1, :])])
    return full_output

def julia(c, width, height, maxiter):

    real = np.linspace(-2, 2, width, dtype=np.float32)
    imag = np.linspace(-2, 2, height, dtype=np.float32)
    
    top_height = height // 2
    top_imag = imag[:top_height]
    z_real, z_imag = np.meshgrid(real, top_imag)
    
    output_top = np.zeros((top_height, width), dtype=np.uint16)
    mask = np.ones_like(z_real, dtype=bool)

    for i in range(maxiter):
        # Compute squares only where not yet diverged.
        zr2 = z_real[mask] ** 2
        zi2 = z_imag[mask] ** 2
        # Calculate new imaginary values.
        z_imag_new = 2 * z_real[mask] * z_imag[mask] + c.imag
        # Update real and imaginary parts.
        z_real[mask] = zr2 - zi2 + c.real
        z_imag[mask] = z_imag_new
        # Check for divergence.
        diverged = zr2 + zi2 >= 4.0
        output_top[mask] = i
        mask[mask] = ~diverged

    # Allocate full output and fill top half.
    full_output = np.empty((height, width), dtype=np.uint16)
    full_output[:top_height, :] = output_top

    # Fill bottom half by 180° rotation of the top half.
    full_output[top_height:, :] = np.flipud(np.fliplr(output_top))
    
    return full_output

def multicorn(rmin, rmax, cmax, width, height, maxiter, exponent):
    # Ensure height is even for symmetry
    if height % 2 != 0:
        height += 1

    real = np.linspace(rmin, rmax, width, dtype=np.float32)
    imag = np.linspace(0, cmax.imag, height // 2, dtype=np.float32)
    c_real, c_imag = np.meshgrid(real, imag)

    output = np.zeros((height // 2, width), dtype=np.uint16)
    z_real = np.zeros_like(c_real)
    z_imag = np.zeros_like(c_imag)
    mask = np.ones_like(c_real, dtype=bool)

    for i in range(maxiter):
        zr2 = z_real[mask] ** 2
        zi2 = z_imag[mask] ** 2

        z_imag_new = np.conj((z_real[mask] + 1j * z_imag[mask]) ** exponent)
        z_real[mask] = z_imag_new.real + c_real[mask]
        z_imag[mask] = z_imag_new.imag + c_imag[mask]

        diverged = zr2 + zi2 >= 4.0
        output[mask] = i
        mask[mask] = ~diverged

    output[output == maxiter - 1] = 0

    # Mirror the top half to the bottom half
    full_output = np.vstack([np.flipud(output), np.flipud(output[::-1, :])])
    return full_output

import numpy as np

def buhdabrot(rmin, rmax, cmax, width, height, maxiter, sample_size):
    """
    Buhdabrot fractal generator with 4D histogram

    Parameters:
        rmin, rmax: Real axis range.
        cmax: Imaginary axis range.
        width, height: Resolution of the output image.
        maxiter: Maximum number of iterations.
        sample_size: Number of random samples.

    Returns:
        4D histogram of the Buhdabrot fractal.
    """
    # Define the number of bins for each dimension
    zr_bins = width
    zi_bins = height
    cr_bins = width
    ci_bins = height

    # Create a 4D histogram to track (zr, zi, cr, ci)
    histogram = np.zeros((zr_bins, zi_bins, cr_bins, ci_bins), dtype=np.uint32)

    for _ in range(sample_size):
        c = complex(
            np.random.uniform(rmin, rmax),
            np.random.uniform(-cmax.imag, cmax.imag)
        )
        z = complex(
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2)
        )
        trajectory = []

        for i in range(maxiter):
            if abs(z) > 2:
                # Update the 4D histogram for all points in the trajectory
                for zr, zi, cr, ci in trajectory:
                    zr_idx = int((zr - rmin) / (rmax - rmin) * zr_bins)
                    zi_idx = int((zi + cmax.imag) / (2 * cmax.imag) * zi_bins)
                    cr_idx = int((cr - rmin) / (rmax - rmin) * cr_bins)
                    ci_idx = int((ci + cmax.imag) / (2 * cmax.imag) * ci_bins)

                    if 0 <= zr_idx < zr_bins and 0 <= zi_idx < zi_bins and 0 <= cr_idx < cr_bins and 0 <= ci_idx < ci_bins:
                        histogram[zr_idx, zi_idx, cr_idx, ci_idx] += 1
                break

            # Store the current state in the trajectory
            trajectory.append((z.real, z.imag, c.real, c.imag))
            z = z * z + c

    return histogram


def project_histogram_4d_to_2d(histogram, projection_plane, width, height):
    """
    Projects a 4D histogram onto a 2D plane.

    Parameters:
        histogram: 4D histogram to project.
        projection_plane: Plane to project onto 
        ("zr_zi", "zr_cr", "zr_ci", "zi_cr", "zi_ci", "cr_ci").
        width:  Resolution of the output image.
        height: Resolution of the output image.

    Returns:
        2D histogram of the projected data.
    """
    # Initialize a 2D histogram
    projection = np.zeros((height, width), dtype=np.uint32)

    zr_bins, zi_bins, cr_bins, ci_bins = histogram.shape

    for zr_idx in range(zr_bins):
        for zi_idx in range(zi_bins):
            for cr_idx in range(cr_bins):
                for ci_idx in range(ci_bins):
                    count = histogram[zr_idx, zi_idx, cr_idx, ci_idx]
                    if count == 0:
                        continue

                    # Map 4D indices to 2D based on the projection plane
                    if projection_plane == "zr_zi":
                        x = zr_idx
                        y = zi_idx
                    elif projection_plane == "zr_cr":
                        x = zr_idx
                        y = cr_idx
                    elif projection_plane == "zr_ci":
                        x = zr_idx
                        y = ci_idx
                    elif projection_plane == "zi_cr":
                        x = zi_idx
                        y = cr_idx
                    elif projection_plane == "zi_ci":
                        x = zi_idx
                        y = ci_idx
                    elif projection_plane == "cr_ci":
                        x = cr_idx
                        y = ci_idx
                    else:
                        raise ValueError("Invalid projection_plane. Choose from" \
                        " 'zr_zi', 'zr_cr', 'zr_ci', 'zi_cr', 'zi_ci', or 'cr_ci'.")

                    # Accumulate the count into the 2D projection
                    if 0 <= x < width and 0 <= y < height:
                        projection[y, x] += count

    # Normalize the projection by dividing by the total count
    total_count = np.sum(histogram)
    if total_count > 0:
        projection = projection / total_count

    return projection

# Noway the code reveal was longer than the actaul video lol
# it might be a crapy video, but it was a fun project to make, and I learned a lot about the mandelbrot set and how to use manim
# and how to make a video with it, so I am happy with it.