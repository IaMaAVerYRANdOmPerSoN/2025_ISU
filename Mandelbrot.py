import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_point_cloud(c, maxiter):
    point_cloud = np.zeros((maxiter, 2))  # Preallocate array with zeros
    z = complex(0, 0)
    
    for i in range(maxiter):
        point_cloud[i] = [z.real, z.imag]
        z = z * z + c
    return point_cloud

def mandelbrot(rmin, rmax, cmax, width, height, maxiter):
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