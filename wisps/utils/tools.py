#my colormap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def my_color_map():
        colors1 = plt.cm.BuGn(np.linspace(0., 1, 256))
        colors2 = plt.cm.Purples(np.linspace(0., 1, 256))
        colors = np.vstack((colors1+colors2)/2)
        return mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
MYCOLORMAP=my_color_map()


