from Demosaic import Demosaic
import numpy as np

arr = np.ones((100, 100), dtype=np.float32)
obj = Demosaic(arr)
obj.demosaic()

rgb_img = obj.getNIR()