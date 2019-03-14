# pip install fbm
import math
import matplotlib.pyplot as plt
from fbm import MBM
from matplotlib.widgets import Slider  # To add interactivity on plot (maybe)


plt.close('all')

# Example Hurst function with respect to time.
def h(t):
    return -0.30 * math.exp(-(50*(t-0.5)**2)) + 0.55

m = MBM(1024, h)

fig = plt.plot(m.times(),m.mbm())