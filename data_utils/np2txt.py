import numpy as np
import sys

test = sys.argv[1]
out = sys.argv[2]

tmp = np.load(test)
#np.savetxt(out, tmp, fmt="%.2f %.2f %.2f %d", newline='\n')
np.savetxt(out, tmp, fmt="%.2f %.2f %.2f %d %d %d %d", newline='\n')
