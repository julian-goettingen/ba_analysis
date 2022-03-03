import matplotlib.pyplot as plt
import numpy as np
import sys

#from PyQt5.Widgets import QWidget
#from PyQt5.QtCore import QApplication

sys.path.append("/home/julian/astaroth/analysis/python/astar/data")
import read


time = int(sys.argv[1])
snapshotdir = f"{sys.argv[2]}/snapshots/final/"

mi = read.MeshInfo(snapshotdir)

# files
uux, xphystime, xok = read.read_bin("VTXBUF_UUX", snapshotdir, str(time), mi)
uuy, yphystime, yok = read.read_bin("VTXBUF_UUY", snapshotdir, str(time), mi)
uuz, zphystime, zok = read.read_bin("VTXBUF_UUZ", snapshotdir, str(time), mi)
lnrho, rphystime, rok = read.read_bin("VTXBUF_LNRHO", snapshotdir, str(time), mi)

def unite(*args):
    if all(map(lambda x : x == args[0], args)):
        return args[0]
    else:
        raise ValueError(f"not all elements equal: {args}")

uux = uux[3:-3,3:-3,3:-3]
uuy = uuy[3:-3,3:-3,3:-3]
uuz = uuz[3:-3,3:-3,3:-3]
lnrho = lnrho[3:-3,3:-3,3:-3]


ok = unite(xok, yok, zok, rok)
phystime = unite(xphystime, yphystime, zphystime, rphystime)
assert(ok)

print(phystime)

print(uux.shape)
print(uuy.shape)
print(uuz.shape)
print(lnrho.shape)

plt.imshow(lnrho[:,:,0])
plt.colorbar()
plt.show()

z = 20

s = 3 # sampling
plt.quiver(uux[::s,::s,z], uuy[::s,::s,z], scale=10)
plt.show()

x = np.arange(0,128)/(2*np.pi)
plt.streamplot(x,x,uux[:,:,z],uuy[:,:,z])
plt.show()




