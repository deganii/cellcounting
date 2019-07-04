from optparse import OptionParser
import time

import numpy as np
import nibabel as nb

from
from nireg import HistogramRegistration, resample
from nipy.utils import example_data

tol = 1e-2


