from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils import detect_threshold
from photutils import detect_sources
from libtiff import TIFF
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import deblend_sources
from photutils import source_properties, EllipticalAperture

# green_path = 'images/raw/02 FITC-before.tif'
green_path = 'images/raw/04 FITC-after.tif'
# red_path = 'images/raw/20190514/20190514 kb fusion 01bt1.tif'
green = TIFF.open(green_path).read_image() / 65535.
# red = TIFF.open(red_path).read_image() / 65535.

green_avg = green.mean(axis=2)
# red_avg = red.mean(axis=2)

# TODO: try adaptive threshold...
threshold = detect_threshold(green_avg, snr=2.)

sigma = 3.0 * gaussian_fwhm_to_sigma # FWHM = 3.
kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
kernel.normalize()
segm = detect_sources(green_avg, threshold, npixels=5, filter_kernel=kernel)
npixels = 5

segm_deblend = deblend_sources(green_avg, segm, npixels=npixels,
                                filter_kernel=kernel, nlevels=32,
                                contrast=0.001)
cat = source_properties(green_avg, segm)
r=3
apertures = []

major = []
minor = []

for obj in cat:
    position = (obj.xcentroid.value, obj.ycentroid.value)
    a = obj.semimajor_axis_sigma.value * r
    b = obj.semiminor_axis_sigma.value * r
    theta = obj.orientation.value
    major.append(a)
    minor.append(b)
    print("({0},{1})".format(obj.xcentroid.value, obj.ycentroid.value))

    apertures.append(EllipticalAperture(position, a, b, theta=theta))


norm = ImageNormalize(stretch=SqrtStretch())
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
# ax1.imshow(green, origin='lower', cmap='Greys_r', norm=norm)
# ax1.set_title('Before Fusion')
# ax2.imshow(segm, origin='lower', cmap=segm.cmap(random_state=12345))


# for aperture in apertures:
#     aperture.plot(color='white', lw=1.5, ax=ax2)

#plt.show()
import os
# os.makedirs('images/output')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))
ax1.set_title('Major Axis')
n, bins, patches = ax1.hist(major, 50, normed=1, facecolor='green', alpha=0.75)
ax2.set_title('Minor Axis')
n, bins, patches = ax2.hist(minor, 50, normed=1, facecolor='blue', alpha=0.75)



plt.savefig('images/output/statsafter.png')