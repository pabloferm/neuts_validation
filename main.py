from nu_pyhepmc import NuHepMC
from validation_plots import Compare


if __name__ == '__main__':
    neut6 = NuHepMC(filename="../neut6_test/neut.6.sf.100k.hepmc3", test=False)
    neut5 = NuHepMC(filename="../neut5_test/neutvect.hepmc3", neut5=True, test=False)

    neut6.get_variables("neutrinos", "leptons")
    neut5.get_variables("neutrinos", "leptons")

    validate = Compare(neut5, neut6)
    validate.split_distro_and_ratio("neutrino_energy", show=True, split="interactions")
    validate.stacked_distro_and_ratio("neutrino_energy", show=True, split="interactions")
    validate.split_distro_and_ratio("lepton_momentum", show=True, split="interactions")
    validate.split_distro_and_ratio("momentum_transfer", show=True, split="flavors")
    validate.split_distro_and_ratio("momentum_transfer", show=True, split="interactions")
    validate.split_distro_and_ratio("cos_theta", show=True, split="interactions")
    validate.stacked_distro_and_ratio("cos_theta", show=True, split="interactions")
    # validate.stacked_distro_and_ratio(
    #     "neutrino_energy", show=True, split="interactions")
    # validate.split_distro_and_ratio("neutrino_energy", show=True)
