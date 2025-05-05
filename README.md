# PySim
PySim is a package for simply interacting with simulations in a pythonic, object oriented way.

```python
from pysim.dhybridr import TurbSim
from pysim.plotting import show, show_video

s = TurbSim("path/to/simulation")
# examine initial conditions
show(s.B.z[0])
show(s.u.x[0])
# make video of density evolution over simulation
@show_video(name='energy_flux', latex=r'$\rho \mathcal{u}_\perp^2$')
def energy_flux(s, **kwargs) -> np.ndarray:
    return np.array([p*(ux**2+uy**2) for p, ux, uy in zip(s.density, s.u.x, s.u.y)])

energy_flux(s)
```

