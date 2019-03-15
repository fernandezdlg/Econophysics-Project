# pip install fbm
import numpy as np
import matplotlib.pyplot as plt
from fbm import MBM
from matplotlib.widgets import Slider  # To add interactivity on plot (maybe)
import matplotlib.pylab as pylab
#from matplotlib import rc
##rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
### for Palatino and other serif fonts use:
##rc('font',**{'family':'serif','serif':['Palatino']})
##rc('text', usetex=True)


params = {'legend.fontsize': 'medium',
          'figure.figsize': (8, 5),
         'axes.labelsize': 'medium',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large',
         'font.family':'sans-serif'}
pylab.rcParams.update(params)


plt.close('all')

# Example Hurst function with respect to time.
def h0(t):
    return -0.30 * np.exp(-(50*(t-0.5)**2)) + 0.55

def h1(t):
    return 0.4/(1+np.exp(5-10*t)) + 0.3

def h2(t):
    return 0.2*np.sin(20*t) + 0.5

m0 = MBM(1024, h0)
m1 = MBM(1024, h1)
m2 = MBM(1024, h2)




#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')


fig0 = plt.figure(0)


ax01 = fig0.add_subplot(2,1,1)
ax02 = fig0.add_subplot(2,1,2)

ax01.set_title(r'MBM for centered negative gaussian Hurst function',)
ax01.plot(m0.times(),m0.mbm(),'r')
ax02.set_title(r'Hurst function')
ax02.plot(m0.times(),h0(m0.times()),'r')

fig0.tight_layout()
fig0.show()

fig0.savefig('MBM_gaussian.png', format='png', dpi=1000)




fig1 = plt.figure(1)

ax11 = fig1.add_subplot(2,1,1)
ax12 = fig1.add_subplot(2,1,2)

ax11.set_title('MBM for logistic Hurst function')
ax11.plot(m1.times(),m1.mbm(),'r')
ax12.set_title('Hurst function')
ax12.plot(m1.times(),h1(m1.times()),'r')

fig1.tight_layout()
fig1.show()

fig1.savefig('MBM_logistic.png', format='png', dpi=1000)




fig2 = plt.figure(2)

ax21 = fig2.add_subplot(2,1,1)
ax22 = fig2.add_subplot(2,1,2)

ax21.set_title('MBM for harmonic oscillating Hurst function')
ax21.plot(m2.times(),m2.mbm(),'r')
ax22.set_title('Hurst function')
ax22.plot(m2.times(),h2(m2.times()),'r')

fig2.tight_layout()
fig2.show()

fig2.savefig('MBM_sin.png', format='png', dpi=1000)


