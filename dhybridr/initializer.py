#pysim imports
from pysim.parsing import File, Folder
from pysim.fields import ScalarField, VectorField
from pysim.dhybridr.input import dHybridRinput
#nonpysim imports
from scipy.io import FortranFile
import numpy as np
from numpy import pi

def field_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray: return np.sum(A * B, axis=0)

class dHybridRconfig(File):
    def __init__(self, parent, mode: str|None = None) -> None:
        self.parent = parent 
        path = f"{self.parent.path}/config"
        File.__init__(self, path)
        if mode: self.mode = mode
        self.read()

    def read(self) -> None:
        self.params = []
        with open(self.path, 'r') as file:
            self.lines = file.readlines()
            for line in self.lines:
                line = line.strip()
                #skip comments and blank lines
                if any([
                    len(line)==0,
                    line.startswith("#"),
                    line.startswith("!")
                ]): continue
                name, value = (x.strip() for x in line.split("="))
                self.params.append(name)
                setattr(self, name, value)
        if "mode" not in self.params: self.set_interactive()
    
    def write(self) -> None:
        self.lines = [
            f"mode={self.mode}"
        ] + [
            f"{n}={getattr(self,n)}" for n in self.params if n!='mode'
        ]
        with open(self.path, 'w') as file: file.write("\n".join(self.lines))

    def set_interactive(self) -> None:
        print("this simulation hasn't been configured yet.")
        self.mode = input("What mode? ").lower()
        while len(new_param:=input("param: ").strip())>0:
            setattr(self, new_param, input("value: "))
        self.write()

class dHybridRSnapshot:
    def __init__(
        self, 
        parent,
        i: int,
        caching: bool = False,
        verbose: bool = False
    ):
        i = i if i<len(parent.B) else len(parent.B)-1
        kwargs = {'caching':caching, 'verbose':verbose, 'parent':parent}
        self.B = VectorField(parent.B.x[i], parent.B.y[i], parent.B.z[i], name=parent.B.name, latex=parent.B.latex, **kwargs)
        self.u = VectorField(parent.u.x[i], parent.u.y[i], parent.u.z[i], name=parent.u.name, latex=parent.u.latex, **kwargs)
        self.E = VectorField(parent.E.x[i], parent.E.y[i], parent.E.z[i], name=parent.E.name, latex=parent.E.latex, **kwargs)
        self.density = ScalarField(parent.density[i], name=parent.density.name, latex=parent.density.latex, **kwargs)
        self.energy_grid = parent.energy_grid[i]
        self.energy_pdf = parent.energy_pdf[i]
        self.dlne = parent.dlne[i]
        self.T = sum(self.energy_grid*self.energy_pdf*self.dlne)
        self.tau = parent.tau[i]
        self.time = parent.time[i]

class dHybridRinitializer:
    def __init__(
        self,
        simulation
    ):
        self.simulation = simulation
        self.input = self.simulation.input
        #parse grid size and shape from input
        self.L: list[float, float] = self.input.boxsize
        self.dx: float = self.L[0] / self.input.ncells[0]
        self.dy: float = self.L[1] / self.input.ncells[1]
        self.Nx: int = int(self.L[0] / self.dx)
        self.Ny: int = int(self.L[1] / self.dy)
        self.shape: tuple[int, int] = (self.Ny, self.Nx)

    def build_B_field(self): 
        self.B = np.array([np.zeros(self.input.ncells) for i in range(2)])
    def build_u_field(self):
        self.u = np.array([np.zeros(self.input.ncells) for i in range(2)])
    def save_init_field(self, field: np.ndarray, path: str): 
        FortranFile(path, 'w').write_record(field.T)
    def prepare_simulation(self):
        self.build_B_field()
        self.save_init_field(self.B, self.simulation.path+"/input/Bfld_init.unf")
        self.build_u_field()
        self.save_init_field(self.u, self.simulation.path+"/input/vfld_init.unf")

class TurbInit(dHybridRinitializer):
    def __init__(
        self,
        simulation
    ):
        self.config = dHybridRconfig(simulation)
        self.mach = self.config.mach
        self.dB = self.config.dB
        self.amplitude: tuple = (self.dB, self.mach)
        self.kinit = (1, np.pi) if 'kinit' not in self.config.params else (float(x) for x in self.config.kinit.split(','))
        dHybridRinitializer.__init__(self, simulation)
        self.kmin: float = 2 * pi / max(self.L)
        self.kmax: float = pi / min([self.dx, self.dy])

        # Set initial k vector
        self.k: np.ndarray = np.mgrid[
                             -self.Ny // 2: self.Ny // 2,
                             -self.Nx // 2: self.Nx // 2
                             ][::-1] * self.kmin
        self.kmag: np.ndarray = np.hypot(*self.k)
        self.kmag[self.kmag == 0] = np.nan

        self.simulation.mach = self.mach 
        self.simulation.dB = self.dB 
        self.simulation.kinit = self.kinit
        if not self.simulation.compressed:
            l = self.input.niter if not self.simulation.outputDir.exists() else len(self.simulation.B)*self.input.ndump
            self.simulation.time = np.arange(0, l, self.input.ndump) * self.input.dt
            self.simulation.tau = self.simulation.time * self.mach / (max(self.input.boxsize))
        elif self.simulation.compressed:
            self.simulation.time = np.array([int(x[-11:-3]) for x in self.simulation.density.file_names]) * self.simulation.input.dt
            self.simulation.tau = self.simulation.time * self.mach / (max(self.input.boxsize))
        if "peak_jz" in self.config.params: 
            self.simulation.peak_jz_ind = int(self.config.peak_jz)
            self.simulation.initial = dHybridRSnapshot(self.simulation,0)
            self.simulation.peak = dHybridRSnapshot(self.simulation, self.simulation.peak_jz_ind)
            self.simulation.snapshots = [
                self.simulation.initial, self.simulation.peak
            ]+[
                dHybridRSnapshot(self.simulation, np.argmin(abs(self.simulation.tau - n))) for n in range(1, int(self.simulation.tau[-1]//1))
            ]

    def fluctuate3D(self, field, amp, no_div=True):
        init_mask = np.where((self.kinit[0] * self.kmin < self.kmag)&(self.kmag < self.kinit[1]*self.kmin))
        M = np.array(init_mask).flatten().size
        phases = np.exp(2j * np.pi * np.random.random(field.shape))

        FT = np.zeros(field.shape, dtype=complex)
        FT[0][init_mask] = amp[0] * np.pi / 2
        FT[1][init_mask] = amp[1] * np.pi
        FT[2][init_mask] = amp[2] * np.pi / 2
        FT *= phases

        # subtract off the parallel x/y components
        FT -= field_dot(FT, self.k / self.kmag) * self.k / self.kmag
        FT[np.isnan(FT)] = 0

        # apply the condition to make this real
        _fx = np.roll(FT[1, ::-1, ::-1, ::-1], 1, axis=(0, 1, 2))
        FT[1, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = np.conj(_fx[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[1, self.Ny // 2:, :self.Nx // 2, :self.Nz // 2] = np.conj(_fx[self.Ny // 2:, :self.Nx // 2, :self.Nz // 2])
        FT[1, :self.Ny // 2, :self.Nx // 2, self.Nz // 2:] = np.conj(_fx[:self.Ny // 2, :self.Nx // 2, self.Nz // 2:])
        FT[1, self.Ny // 2:, :self.Nx // 2, self.Nz // 2:] = np.conj(_fx[self.Ny // 2:, :self.Nx // 2, self.Nz // 2:])

        _fy = np.roll(FT[0, ::-1, ::-1, ::-1], 1, axis=(0, 1, 2))
        FT[0, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = np.conj(_fy[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[0, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = np.conj(_fy[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[0, self.Ny // 2:, :self.Nx // 2, :self.Nz // 2] = np.conj(_fy[self.Ny // 2:, :self.Nx // 2, :self.Nz // 2])
        FT[0, :self.Ny // 2, :self.Nx // 2, self.Nz // 2:] = np.conj(_fy[:self.Ny // 2, :self.Nx // 2, self.Nz // 2:])
        FT[0, self.Ny // 2:, :self.Nx // 2, self.Nz // 2:] = np.conj(_fy[self.Ny // 2:, :self.Nx // 2, self.Nz // 2:])

        _fz = np.roll(FT[2, ::-1, ::-1, ::-1], 1, axis=(0, 1, 2))
        FT[2, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = np.conj(_fz[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[2, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = np.conj(_fz[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[2, self.Ny // 2:, :self.Nx // 2, :self.Nz // 2] = np.conj(_fz[self.Ny // 2:, :self.Nx // 2, :self.Nz // 2])
        FT[2, :self.Ny // 2, :self.Nx // 2, self.Nz // 2:] = np.conj(_fz[:self.Ny // 2, :self.Nx // 2, self.Nz // 2:])
        FT[2, self.Ny // 2:, :self.Nx // 2, self.Nz // 2:] = np.conj(_fz[self.Ny // 2:, :self.Nx // 2, self.Nz // 2:])

        # I think we have to fix the zero line
        FT[:, self.Ny // 2, :, :] = 0.j
        FT[:, :, self.Nx // 2, :] = 0.j
        FT[:, :, :, self.Nz // 2] = 0.j

        # take the inverse fourier transform
        y: np.ndarray = np.real(
            np.fft.ifftn(
                np.fft.ifftshift(
                    FT
                )
            )
        ) / M * self.Nx * self.Ny * self.Nz
        rms = np.sqrt(np.nanmean(y[0]**2 + y[1]**2 + y[2]**2))
        y *= (amp / rms)
        return np.float32(y)

    def fluctuate2D(self, field, amp, no_div=True):
        """
        Given the initialization create a 2d array the same shape as the simulation which will smoothly fluctuate
        over length scales kinit
        :return y: np.ndarray[np.float32]: random fluctuations set by parameters passed to __init__
        """
        init_mask: np.ndarray = np.where((self.kinit[0] * self.kmin < self.kmag) & (self.kmag < self.kinit[1] * self.kmin))
        M: int = len(init_mask[0])  # number of cells with an amplitude
        phases: np.ndarray = np.exp(2j * pi * np.random.random(field.shape))  # randomized complex phases
        phases[2] *= 0  # don't wiggle the z component
        # Setting the fourier transform
        FT: np.ndarray = np.zeros(field.shape, dtype=complex)  # same shape as field
        FT[0][init_mask] = amp * np.pi / 2 # set x and y amplitudes
        FT[1][init_mask] = amp * np.pi
        FT *= phases  # apply phases
        # subtract off the parallel x/y components
        if no_div:
            FT[:2] -= field_dot(FT[:2], self.k / self.kmag) * self.k / self.kmag
        FT[np.isnan(FT)] = 0
        # apply the condition to make this real
        _fx = np.roll(FT[1, ::-1, ::-1], 1, axis=(0, 1))
        FT[1, :self.Ny // 2] = np.conj(_fx[:self.Ny // 2])

        _fy = np.roll(FT[0, ::-1, ::-1], 1, axis=(0, 1))
        FT[0, :self.Ny // 2] = np.conj(_fy[:self.Ny // 2])

        # I think we have to fix the zero line
        FT[1, self.Ny // 2, 1:self.Nx // 2] = FT[1, self.Ny // 2, self.Nx // 2 + 1:][::-1]
        FT[1, self.Ny // 2, 1:self.Nx // 2] = np.conj(FT[1, self.Ny // 2, self.Nx // 2 + 1:][::-1])
        FT[1, self.Ny // 2, :] = 0.j
        FT[1, :, self.Nx // 2] = 0.j

        FT[0, self.Ny // 2, 1:self.Nx // 2] = FT[0, self.Ny // 2, self.Nx // 2 + 1:][::-1]
        FT[0, self.Ny // 2, 1:self.Nx // 2] = np.conj(FT[0, self.Ny // 2, self.Nx // 2 + 1:][::-1])
        FT[0, self.Ny // 2, :] = 0.j
        FT[0, :, self.Nx // 2] = 0.j

        # take the inverse fourier transform
        y: np.ndarray = np.array([*np.real(
            np.fft.ifft2(
                np.fft.ifftshift(
                    FT[:2]
                )
            )
        ), np.zeros(FT[0].shape)]) / M * self.Nx * self.Ny

        rms = np.sqrt(np.nanmean(y[0]**2 + y[1]**2))
        y *= (amp / rms)
        return np.float32(y)
    
    def fluctuate(self, field, amp, no_div=True):
        if type(amp) in [int, float]: return self.fluctuate2D(field, amp, no_div=no_div)
        elif type(amp) in [list, np.ndarray, tuple]: 
            if len(amp) == 2: amp = [amp[0], amp[0], amp[1]]
            return self.fluctuate3D(field, amp, no_div=no_div)

    def construct_field(self, x, y, z, amp, no_div=True):
        """
        Constructs a 3 x N x N array representing a constant x, y, and z component with additional fluctuations
        :param x:
        :param y:
        :param z:
        :param no_div: whether or not to ensure that the divergence of the field is 0 when applying fluctuations
        :return field:
        """
        base_field = np.array([
            np.zeros(self.shape) + y,
            np.zeros(self.shape) + x,
            np.zeros(self.shape) + z
        ], dtype=np.float32)

        base_rms: float = np.sqrt(np.mean(base_field ** 2))
        fluctuations: np.ndarray = self.fluctuate(base_field, amp, no_div=no_div)
        field: np.ndarray = base_field + fluctuations
        alt_rms: float = np.sqrt(np.mean(base_field ** 2))
        if not any([base_rms == 0, alt_rms == 0]):
            field *= base_rms / alt_rms
        return field
    def build_B_field(self): self.B = self.construct_field(0, 0, 1, self.amplitude[0])
    def build_u_field(self): self.u = self.construct_field(0, 0, 0, self.amplitude[1])

class FlareWaveInit(dHybridRinitializer):
    def __init__(
            self,
            input_file,
            B0 = 1,
            Bg = 2,
            w0 = 2,
            psi0 = 0.5,
            vth = 0.1
    ):
        dHybridRinitializer.__init__(self, input_file)
        self.B0 = B0
        self.Bg = Bg 
        self.w0 = w0 
        self.psi0 = psi0

    def build_B_field(self, unknown_variable=69.12):
        x = np.arange(0, self.Nx) * self.dx
        y = np.arange(0, self.Ny) * self.dy
        Bx = np.array([
            self.B0 * (np.tanh((y - 0.25*self.L[1])/self.w0) - np.tanh((y - 0.75*self.L[1])/self.w0) - 1)
        for i in range(len(x))]).T
        By = np.array([
            (unknown_variable / self.L[0]) * np.cos(2*np.pi*x / self.L[0]) * np.sin(2*np.pi*x / self.L[0])**10
        for i in range(len(y))])
        Bz = np.sqrt(self.B0**2 + self.Bg**2 - Bx**2)
        self.B = np.array([Bx.T, By.T, Bz.T], dtype=np.float32)
    