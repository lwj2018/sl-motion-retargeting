import sys
sys.path.append("../utils")
import numpy as np
from Quaternions import Quaternions

x = np.random.randn(1,6,4)
x = Quaternions(x)
y = Quaternions.id_like(x)
z = Quaternions.slerp(x,y,0.6)
print(z.shape)