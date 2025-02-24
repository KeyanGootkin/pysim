from pysim.tristan.tristan import Tristan, TristanParticleSpecies, TristanParticleQuantity
from pysim.fields import VectorField

import numpy as np

class MaskedTristanParticleQuantity(TristanParticleQuantity):
    def __init__(
            self,
            param_key,
            mask_param,
            mask_value,
            species = None,
            parent = None,
            caching = False,
            verbose: bool = False,
            name: str = None, 
            latex: str = None
    ):
        TristanParticleQuantity.__init__(
            self,
            param_key,
            parent = parent,
            caching = caching,
            verbose = verbose,
            name = name, 
            latex = latex 
        )
        self.mask_param = mask_param
        self.mask_value = mask_value
        self.species = species

    def __getitem__(self, item):
        mask = np.where(getattr(self.species, self.mask_param)[item]==self.mask_value)
        return super().__getitem__(item)[mask]

class KappaSubPopulation(TristanParticleSpecies):
    def __init__(
            self,
            parent_species
    ):
        self.parent_species = parent_species
        TristanParticleSpecies.__init__(
            self, 
            self.parent_species.name, 
            parent = self.parent_species.parent, 
            m = self.parent_species.m, 
            q = self.parent_species.q, 
            N = self.parent_species.N, 
            sig = self.parent_species.sig
        )
        kwargs = {'caching':self.parent.caching, 'verbose':self.parent.verbose, 'parent':self.parent}
        self.x = MaskedTristanParticleQuantity(
            'x'+self.sig, "ch", self.parent.kappa_density, 
            species=self.parent_species, name="x", latex="$x$", **kwargs
        )
        self.y = MaskedTristanParticleQuantity(
            'y'+self.sig, "ch", self.parent.kappa_density, 
            species=self.parent_species, name="y", latex="$y$", **kwargs
        )
        self.z = MaskedTristanParticleQuantity(
            'z'+self.sig, "ch", self.parent.kappa_density, 
            species=self.parent_species, name="z", latex="$z$", **kwargs
        )
        self.v = VectorField(
            MaskedTristanParticleQuantity(
                "u"+self.sig, "ch", self.parent.kappa_density, 
                species=self.parent_species, name="vx", latex="$v_x$", **kwargs
            ),
            MaskedTristanParticleQuantity(
                "v"+self.sig, "ch", self.parent.kappa_density, 
                species=self.parent_species, name="vy", latex="$v_y$", **kwargs
            ),
            MaskedTristanParticleQuantity(
                "w"+self.sig, "ch", self.parent.kappa_density, 
                species=self.parent_species, name="vz", latex="$v_z$", **kwargs
            ),
            name="v", latex=r"$\vec{v}$"
        )
        self.gamma = MaskedTristanParticleQuantity(
            "gamma"+self.sig, "ch", self.parent.kappa_density, 
            species=self.parent_species, name="gamma", latex=rf"$\gamma_{self.sig}$", **kwargs
        )

class KappaSim(Tristan):
    def __init__(
        self,
        path: str,
        caching: bool = False,
        verbose: bool = False,
        compressed: bool = False
    ):
        Tristan.__init__(
            self,
            path,
            caching=caching,
            verbose=verbose,
            compressed=compressed
        )
        self.kappa = self.input.kappa.value 
        self.kappa_density = self.input.kappadens.value
        self.ions.kappa = KappaSubPopulation(self.ions)
        self.electrons.kappa = KappaSubPopulation(self.electrons)