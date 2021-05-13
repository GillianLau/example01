import missingno as msno
import pandas as pd
import numpy as np
collision = pd.read_csv("Motor_Vehicle_Collisions_-_Crashes.csv")
collision = collision.replace("nan",np.nan)
import matplotlib as mpl
mpl.use("TkAgg")
# matplotlib inline
import matplotlib.pyplot as plt
#fig = plt.figure()

#msno.matrix(collision.sample(250))


msno.bar(collision.sample(1000))


#msno.heatmap(collision)

#msno.dendrogram(collision)
plt.show()
