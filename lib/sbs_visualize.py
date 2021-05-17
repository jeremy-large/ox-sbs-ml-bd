import numpy as np
import pylab as plt

HOUR = 12
MONTH = 201007
AX_MIN = 0
AX_MAX = 4
JUMPS = 20j
GRID_X, GRID_Y = np.mgrid[AX_MIN: AX_MAX: JUMPS, AX_MIN: AX_MAX: JUMPS]


def threeDplot(a, b, c, model_name='Decision Tree'):
    ax = plt.figure().add_subplot(111, projection="3d")
    ax.set_box_aspect([4, 3, 2])
    ax.plot_surface(a, b, c, cmap="coolwarm", rstride=1, cstride=1)
    ax.set_title(f"Predictions of the {model_name}, setting hour to {HOUR} and month to {MONTH}")
    return ax


def plot_coeffs(mod, mod_name, comment):
    plt.plot(mod.coef_, marker='o')
    plt.grid()
    plt.title(f"The betas of the {mod_name} - {comment}")
    plt.axhline(color='k')

