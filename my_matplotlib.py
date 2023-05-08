from matplotlib import pyplot
import math
import numpy as np

x = np.linspace(-10,10,100)
y= x**2
g1 = pyplot.plot(x/2,y/2, linestyle="dashed", color = "red")
g2 = pyplot.plot(x,y)
pyplot.legend((g1[0], g2[0]), ("red", "blue"), loc=4)
pyplot.show()