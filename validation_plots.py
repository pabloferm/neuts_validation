import numpy as np
from particle import literals as lp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cycler import cycler
from itertools import repeat
import os

plt.style.use(os.environ['PYNU'] + '/../utils/plot.mplstyle')


class Compare:
    def __init__(self, neut_hepmc1, neut_hepmc2):
        self.neut_hepmc1 = neut_hepmc1
        self.neut_hepmc2 = neut_hepmc2

        self.interaction_mode_labels = self.neut_hepmc1.interaction_mode_labels
        self.neutrinos_label = self.neut_hepmc1.neutrinos_label
        self.charged_leptons_label = self.neut_hepmc1.charged_leptons_label
        self.leptons_label = self.neut_hepmc1.leptons_label

        self.neutrinos = self.sort(np.unique(self.neut_hepmc1.neutrino_pdg))
        self.interactions = self.sort(np.unique(self.neut_hepmc1.interaction_mode))

        self.cmap = mpl.cm.get_cmap('rainbow')
        self.nbins = int(2**(np.log10(self.neut_hepmc1.N_EVENTS)+1))

        self.variables = [
            "neutrino_energy",
            "neutrino_momentum",
            "neutrino_px",
            "neutrino_py",
            "neutrino_pz",
            "lepton_energy",
            "lepton_momentum",
            "lepton_px",
            "lepton_py",
            "lepton_pz",
            "lepton_mass",
            "cos_theta",
            "momentum_transfer",
            "energy_transfer",
            "interaction_mode"]
        self.variable_titles = {
            "neutrino_energy": r"$E_{\nu}$ (MeV)",
            "neutrino_momentum": r"$E_{\nu}$ (MeV/c)",
            "neutrino_px": r"$p^x_{\nu}$ (MeV/c)",
            "neutrino_py": r"$p^y_{\nu}$ (MeV/c)",
            "neutrino_pz": r"$p^z_{\nu}$ (MeV/c)",
            "lepton_energy": r"$E_{lepton}$ (MeV)",
            "lepton_momentum": r"$p_{lepton}$ (MeV/c)",
            "lepton_px": r"$p^x_{lepton}$ (MeV/c)",
            "lepton_py": r"$p^y_{lepton}$ (MeV/c)",
            "lepton_pz": r"$p^z_{lepton}$ (MeV/c)",
            "lepton_mass": r"$m_{lepton}$ (MeV/c$^2$)",
            "cos_theta": r"$\cos \theta$",
            "momentum_transfer": r"$q_3$ (MeV/c)",
            "energy_transfer": r"$q_0$ (MeV)",
            "interaction_mode": "Interaction mode"}
        self.variable_scale = {
            "neutrino_energy": True,
            "neutrino_momentum": True,
            "neutrino_px": True,
            "neutrino_py": True,
            "neutrino_pz": True,
            "lepton_energy": True,
            "lepton_momentum": True,
            "lepton_px": True,
            "lepton_py": True,
            "lepton_pz": True,
            "lepton_mass": True,
            "cos_theta": True,
            "momentum_transfer": True,
            "energy_transfer": True,
            "interaction_mode": True}

    def sort(self, array):
        array_pos = np.sort(array[array>0])
        array_neg = - np.sort(np.abs(array[array>0]))
        return np.append(array_pos, array_neg)

    def _distro_and_ratio(self):
        _fig = plt.figure(
            tight_layout=True,
            figsize=(
                7,
                6))
        _plot = _fig.add_gridspec(nrows=3, ncols=1, right=0.65)
        _distplot = _fig.add_subplot(_plot[:-1, 0])
        _ratioplot = _fig.add_subplot(_plot[-1, 0], sharex=_distplot)
        _distplot.set_ylabel(r"Events")
        _ratioplot.set_ylabel(r"ratio")
        title = _fig.add_subplot(_plot[:])
        title.axis("off")
        title.set_title(
            f"NEUT-{self.neut_hepmc1.VERSION} (solid) vs NEUT-{self.neut_hepmc2.VERSION} (dashed)",
            fontsize=14)
        # _distplot.get_xaxis().set_visible(False)

        return _fig, _plot, _distplot, _ratioplot

    def separated_distro_and_ratio(
            self,
            variable,
            show=False,
            split=None,
            cut=None):
        data1, data2, = self.data_selector(variable)
        # cut1, cut2 = self.cutter(cut)

        slice_length = 0
        if split:
            slice1, slice2, all_labels = self.slicer(split)
            x1 = []
            x2 = []
            labels = []
            for k, (s1, s2) in enumerate(zip(slice1, slice2)):
                if np.sum(s1)>0 and np.sum(s2)>0:
                    x1.append(data1[s1])
                    x2.append(data2[s2])
                    labels.append(all_labels[k])
                    slice_length += 1
        else:
            x1 = data1
            x2 = data2

        colors = [mpl.colors.to_hex(self.cmap(x)) for x in np.linspace(0, 1, slice_length)]
        custom_cycler = (cycler(color=colors))

        for kk in range(slice_length):
            fig, plot, distplot, ratioplot = self._distro_and_ratio()

            values1, edges, __ = distplot.hist(
                x1[kk], bins=self.nbins, label=labels[kk], histtype='step', linewidth=1)
            values2, edges, __ = distplot.hist(
                x2[kk], bins=edges, histtype='step', linestyle="dashed", linewidth=1)
            centers = 0.5 * (edges[1:] + edges[:-1])
            ratios = np.zeros_like(values1)
            r_errors = np.zeros_like(values1)
            ratios[(values2>0) & (values1>0)] = values1[(values2>0) & (values1>0)]/values2[(values2>0) & (values1>0)]
            r_errors[(values2>0) & (values1>0)] = np.sqrt(1 / values1[(values2>0) & (values1>0)] + 1 / values2[(values2>0) & (values1>0)])
            r_errors *= ratios

            ratioplot.errorbar(
                centers[ratios>0],
                ratios[ratios>0],
                yerr=r_errors[ratios>0],
                fmt='o', markersize=1.5, elinewidth=1
                )
            ratioplot.hlines(1,
                             edges[0],
                             edges[-1],
                             colors="k",
                             linestyles="dotted",
                             linewidths=0.5)
            y0, y1 = ratioplot.get_ylim()
            ratioplot.set_ylim(max(0.1, 1 - max(np.abs(1 - y0), np.abs(1 - y1))), 1 + max(np.abs(1 - y0), np.abs(1 - y1)))
            ratioplot.set_xlabel(self.variable_titles[variable])
            __, y1 = ratioplot.get_ylim()
            if y1>2:
                ratioplot.set_yscale("log")
            distplot.legend(bbox_to_anchor=(1.05, 1), ncol=1, fontsize=10)
            if self.variable_scale[variable]:
                distplot.set_yscale("log")
            fig.savefig("split_" + variable + "_" + str(split) + "_" + str(kk) + ".png")
            if show:
                plt.show()
            plt.clf



    def split_distro_and_ratio(
            self,
            variable,
            show=False,
            split=None,
            cut=None):
        data1, data2, = self.data_selector(variable)
        # cut1, cut2 = self.cutter(cut)

        slice_length = 0
        if split:
            slice1, slice2, all_labels = self.slicer(split)
            x1 = []
            x2 = []
            labels = []
            for k, (s1, s2) in enumerate(zip(slice1, slice2)):
                if np.sum(s1)>0 and np.sum(s2)>0:
                    x1.append(data1[s1])
                    x2.append(data2[s2])
                    labels.append(all_labels[k])
                    slice_length += 1
        else:
            x1 = data1
            x2 = data2
        if slice_length <= 1:
            x1 = data1
            x2 = data2

        fig, plot, distplot, ratioplot = self._distro_and_ratio()
        colors = [mpl.colors.to_hex(self.cmap(x)) for x in np.linspace(0, 1, slice_length)]
        custom_cycler = (cycler(color=colors))
        distplot.set_prop_cycle(custom_cycler)
        ratioplot.set_prop_cycle(custom_cycler)

        if slice_length == 1:
            values1, edges, __ = distplot.hist(
                x1, bins=self.nbins, label=labels, histtype='step', linewidth=1)
            values2, edges, __ = distplot.hist(
                x2, bins=edges, histtype='step', linestyle="dashed", linewidth=1)
            centers = 0.5 * (edges[1:] + edges[:-1])
            ratios = np.zeros_like(values1)
            r_errors = np.zeros_like(values1)
            ratios[(values2>0) & (values1>0)] = values1[(values2>0) & (values1>0)]/values2[(values2>0) & (values1>0)]
            r_errors[(values2>0) & (values1>0)] = np.sqrt(1 / values1[(values2>0) & (values1>0)] + 1 / values2[(values2>0) & (values1>0)])
            r_errors *= ratios

            for ll in range(slice_length):
                ratioplot.errorbar(
                    centers[ratios>0],
                    ratios[ratios>0],
                    yerr=r_errors[ratios>0],
                    fmt='o', markersize=1.5, elinewidth=1
                    )
            ratioplot.hlines(1,
                             edges[0],
                             edges[-1],
                             colors="k",
                             linestyles="dotted",
                             linewidths=0.5)
        else:
            values1, edges, __ = distplot.hist(
                x1, bins=self.nbins, label=labels, histtype='step', linewidth=1)
            values2, edges, __ = distplot.hist(
                x2, bins=edges, histtype='step', linestyle="dashed", linewidth=1)
            centers = 0.5 * (edges[1:] + edges[:-1])
            ratios = np.zeros_like(values1)
            r_errors = np.zeros_like(values1)
            ratios[(values2>0) & (values1>0)] = values1[(values2>0) & (values1>0)]/values2[(values2>0) & (values1>0)]
            r_errors[(values2>0) & (values1>0)] = np.sqrt(1 / values1[(values2>0) & (values1>0)] + 1 / values2[(values2>0) & (values1>0)])
            r_errors *= ratios

            for ll in range(slice_length):
                ratioplot.errorbar(
                    centers[ratios[ll,:]>0],
                    ratios[ll,:][ratios[ll,:]>0],
                    yerr=r_errors[ll,:][ratios[ll,:]>0],
                    fmt='o', markersize=1.5, elinewidth=1
                    )
            ratioplot.hlines(1,
                             edges[0],
                             edges[-1],
                             colors="k",
                             linestyles="dotted",
                             linewidths=0.5)
        y0, y1 = ratioplot.get_ylim()
        ratioplot.set_ylim(max(0.1, 1 - max(np.abs(1 - y0), np.abs(1 - y1))), min(5, 1 + max(np.abs(1 - y0), np.abs(1 - y1))))
        ratioplot.set_xlabel(self.variable_titles[variable])
        __, y1 = ratioplot.get_ylim()
        if y1>2:
            ratioplot.set_yscale("log")
        distplot.legend(bbox_to_anchor=(1.05, 1), ncol=1, fontsize=10)
        if self.variable_scale[variable]:
            distplot.set_yscale("log")
        fig.savefig("split_" + variable + "_" + str(split) + ".png")
        if show:
            plt.show()
        plt.clf

    def stacked_distro_and_ratio(
            self,
            variable,
            show=False,
            split=None,
            cut=None):
        data1, data2, = self.data_selector(variable)
        # cut1, cut2 = self.cutter(cut)

        if split:
            slice1, slice2, all_labels = self.slicer(split)
            x1 = []
            x2 = []
            labels = []
            slice_length = 0
            for k, (s1, s2) in enumerate(zip(slice1, slice2)):
                if np.sum(s1)>0 and np.sum(s2)>0:
                    x1.append(data1[s1])
                    x2.append(data2[s2])
                    labels.append(all_labels[k])
                    slice_length += 1
        else:
            x1 = data1
            x2 = data2
        if slice_length == 1:
            x1 = data1
            x2 = data2

        fig, plot, distplot, ratioplot = self._distro_and_ratio()
        colors = [mpl.colors.to_hex(self.cmap(x)) for x in np.linspace(0, 1, slice_length)]
        custom_cycler = (cycler(color=colors))
        distplot.set_prop_cycle(custom_cycler)
        ratioplot.set_prop_cycle(custom_cycler)

        if slice_length == 1:
            self.split_distro_and_ratio(variable, show=show, split=split, cut=cut)

        else:
            ratios = []
            r_errors = []
            for x, y in zip(x1, x2):
                h1, edges = np.histogram(x, bins=self.nbins)
                h2, __ = np.histogram(y, bins=edges)
                rr = np.zeros_like(h1)
                re = np.zeros_like(h1)
                rr[(h2>0) & (h1>0)] = h1[(h2>0) & (h1>0)]/h2[(h2>0) & (h1>0)]
                ratios.append(rr)
                re[(h2>0) & (h1>0)] = np.sqrt(1 / h1[(h2>0) & (h1>0)] + 1 / h2[(h2>0) & (h1>0)])
                r_errors.append(rr*re)
            ratios = np.array(ratios)
            r_errors = np.array(r_errors)
            

            values1, edges, __ = distplot.hist(
                x1, bins=self.nbins, label=labels, histtype='step', linewidth=1, stacked=True)
            values2, edges, __ = distplot.hist(
                x2, bins=edges, histtype='step', linestyle="dashed", linewidth=1, stacked=True)
            centers = 0.5 * (edges[1:] + edges[:-1])

            for ll in range(slice_length):
                ratioplot.errorbar(
                    centers[ratios[ll,:]>0],
                    ratios[ll,:][ratios[ll,:]>0],
                    yerr=r_errors[ll,:][ratios[ll,:]>0],
                    fmt='o', markersize=1.5, elinewidth=1
                    )
            ratioplot.hlines(1,
                             edges[0],
                             edges[-1],
                             colors="k",
                             linestyles="dotted",
                             linewidths=0.5)
        y0, y1 = ratioplot.get_ylim()
        ratioplot.set_ylim(max(0.1, 1 - max(np.abs(1 - y0), np.abs(1 - y1))), min(5,1 + max(np.abs(1 - y0), np.abs(1 - y1))))
        ratioplot.set_xlabel(self.variable_titles[variable])
        __, y1 = ratioplot.get_ylim()
        if y1>2:
            ratioplot.set_yscale("log")
        distplot.legend(bbox_to_anchor=(1.05, 1), ncol=1, fontsize=10)
        if self.variable_scale[variable]:
            distplot.set_yscale("log")
        fig.savefig("split_" + variable + "_" + str(split) + ".png")
        if show:
            plt.show()
        plt.clf

    def norm_distro_and_ratio(self):
        pass

    def data_selector(self, variable):
        if variable in self.variables:
            return self.neut_hepmc1.__getattribute__(
                variable), self.neut_hepmc2.__getattribute__(variable)
        else:
            print(
                f"Please, select a valid variable from {self.variables} or implement your own.")

    def cutter(self, cut):
        pass

    def slicer(self, split):
        if split is None:
            # return None, None
            return repeat(True), repeat(True)
        elif split == "flavors":
            _slice1 = []
            _slice2 = []
            _labels = []
            for nu in self.neutrinos:
                _slice1 += [self.neut_hepmc1.neutrino_pdg == nu]
                _slice2 += [self.neut_hepmc2.neutrino_pdg == nu]
                _labels += [self.neutrinos_label[nu]]
        elif split == 'interactions':
            _slice1 = []
            _slice2 = []
            _labels = []
            for mode in self.interactions:
                if mode > 0:
                    nu = r'$\nu$ '
                else:
                    nu = r'$\overline{\nu}$ '
                _slice1 += [self.neut_hepmc1.interaction_mode == mode]
                _slice2 += [self.neut_hepmc2.interaction_mode == mode]
                _labels += [nu + self.interaction_mode_labels[mode]]
        elif split == 'CC-interactions':
            _slice1 = []
            _slice2 = []
            _labels = []
            for mode in self.interactions:
                if np.abs(mode) < 27:
                    if mode > 0:
                        nu = r'$\nu$ '
                    else:
                        nu = r'$\overline{\nu}$ '
                    _slice1 += [self.neut_hepmc1.interaction_mode == mode]
                    _slice2 += [self.neut_hepmc2.interaction_mode == mode]
                    _labels += [nu + self.interaction_mode_labels[mode]]
        elif split == 'NC-interactions':
            _slice1 = []
            _slice2 = []
            _labels = []
            for mode in self.interactions:
                if np.abs(mode) > 27:
                    if mode > 0:
                        nu = r'$\nu$ '
                    else:
                        nu = r'$\overline{\nu}$ '
                    _slice1 += [self.neut_hepmc1.interaction_mode == mode]
                    _slice2 += [self.neut_hepmc2.interaction_mode == mode]
                    _labels += [nu + self.interaction_mode_labels[mode]]
        # elif split == 'both':
        #     _energy = []
        #     _px = []
        #     _py = []
        #     _pz = []
        #     _label = []
        #     for nu in self.neutrinos:
        #         for mode in self.interactions:
        #             cut = self.interaction_mode == mode
        #             cut &= self.neutrino_pdg == nu
        #             _energy.append(self.neutrino_energy[cut])
        #             _px.append(self.neutrino_px[cut])
        #             _py.append(self.neutrino_py[cut])
        #             _pz.append(self.neutrino_pz[cut])
        #             _label.append(self.neutrinos_label[nu] + ' ' + self.interaction_mode_labels[mode])
        

        return _slice1, _slice2, _labels

    # def plot_nu_spectra(self, show=False, split=None):
    #     print('Plotting neutrino 4-momenta')
    #     n_bins = int(2**np.log10(self.neutrino_energy.size))
    #     nrows = 5
    #     self.neutrinos = np.unique(self.neutrino_pdg)
    #     self.interactions = np.unique(self.interaction_mode)
    #     fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    #     ax[0,0].set_prop_cycle('color', [self.cm(1.*i/self.NUM_COLORS) for i in range(self.NUM_COLORS)])
    #     ax[0,1].set_prop_cycle('color', [self.cm(1.*i/self.NUM_COLORS) for i in range(self.NUM_COLORS)])
    #     ax[1,0].set_prop_cycle('color', [self.cm(1.*i/self.NUM_COLORS) for i in range(self.NUM_COLORS)])
    #     ax[1,1].set_prop_cycle('color', [self.cm(1.*i/self.NUM_COLORS) for i in range(self.NUM_COLORS)])

    #     if split is None:
    #         ax[0, 0].hist(self.neutrino_energy, bins=n_bins,)
    #         ax[0, 1].hist(self.neutrino_px, bins=n_bins)
    #         ax[1, 0].hist(self.neutrino_py, bins=n_bins)
    #         ax[1, 1].hist(self.neutrino_pz, bins=n_bins)
    #     elif split == 'flavors':
    #         for nu in self.neutrinos:
    #             ax[0, 0].hist(self.neutrino_energy[self.neutrino_pdg == nu],
    #                           bins=n_bins, label=self.neutrinos_label[nu])
    #             ax[0, 1].hist(self.neutrino_px[self.neutrino_pdg == nu],
    #                           bins=n_bins, label=self.neutrinos_label[nu])
    #             ax[1, 0].hist(self.neutrino_py[self.neutrino_pdg == nu],
    #                           bins=n_bins, label=self.neutrinos_label[nu])
    #             ax[1, 1].hist(self.neutrino_pz[self.neutrino_pdg == nu],
    #                           bins=n_bins, label=self.neutrinos_label[nu])
    #         numitems = self.neutrinos.size
    #     elif split == 'interactions':
    #         for mode in self.interactions:
    #             if mode > 0:
    #                 nu = r'$\nu$ '
    #             else:
    #                 nu = r'$\overline[\nu]$ '
    #             ax[0, 0].hist(self.neutrino_energy
    #                           [self.interaction_mode == mode],
    #                           bins=n_bins, label=nu + self.interaction_mode_labels
    #                           [mode])
    #             ax[0, 1].hist(self.neutrino_px
    #                           [self.interaction_mode == mode],
    #                           bins=n_bins, label=nu + self.interaction_mode_labels
    #                           [mode])
    #             ax[1, 0].hist(self.neutrino_py
    #                           [self.interaction_mode == mode],
    #                           bins=n_bins, label=nu + self.interaction_mode_labels
    #                           [mode])
    #             ax[1, 1].hist(self.neutrino_pz
    #                           [self.interaction_mode == mode],
    #                           bins=n_bins, label=nu + self.interaction_mode_labels
    #                           [mode])
    #         numitems = self.interactions.size
    #     elif split == 'both':
    #         _energy = []
    #         _px = []
    #         _py = []
    #         _pz = []
    #         _label = []
    #         for nu in self.neutrinos:
    #             for mode in self.interactions:
    #                 cut = self.interaction_mode == mode
    #                 cut &= self.neutrino_pdg == nu
    #                 _energy.append(self.neutrino_energy[cut])
    #                 _px.append(self.neutrino_px[cut])
    #                 _py.append(self.neutrino_py[cut])
    #                 _pz.append(self.neutrino_pz[cut])
    #                 _label.append(self.neutrinos_label[nu] + ' ' + self.interaction_mode_labels[mode])
    #         ax[0,0].hist(_energy, bins=n_bins, stacked=True)
    #         ax[0,1].hist(_px, bins=n_bins, label=_label, stacked=True)
    #         ax[1,0].hist(_py, bins=n_bins, stacked=True)
    #         ax[1,1].hist(_pz, bins=n_bins, stacked=True)
    #         numitems = self.neutrinos.size * self.interactions.size

    #     ax[0, 0].set_xlabel(r'$E$ [MeV]')
    #     ax[0, 1].set_xlabel(r'$p_x$ [MeV/c]')
    #     ax[1, 0].set_xlabel(r'$p_y$ [MeV/c]')
    #     ax[1, 1].set_xlabel(r'$p_z$ [MeV/c]')
    #     ncols = int(np.ceil(numitems / float(nrows)))
    #     #ax[0, 0].legend(loc='best', ncol=ncols, fontsize=10)
    #     ax[0, 1].legend(bbox_to_anchor=(1.04,1), ncol=1, fontsize=10)
    #     plt.tight_layout()
    #     fig.savefig('neutrino_spectra.png')
    #     if show:
    #         plt.show()
    #     plt.clf

    # def plot_lepton_spectra(self, show=False, flavor=False, interaction=True):
    #     print('Plotting lepton 4-momenta')
    #     n_bins = int(2**np.log10(self.lepton_energy.size))
    #     self.leptons = np.unique(self.lepton_pdg)
    #     fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    #     if flavor:
    #         for nu in self.leptons:
    #             ax[0, 0].hist(self.lepton_energy[self.lepton_pdg == nu],
    #                         bins=n_bins, label=self.leptons_label[nu])
    #             ax[0, 1].hist(self.lepton_px[self.lepton_pdg == nu],
    #                         bins=n_bins, label=self.leptons_label[nu])
    #             ax[1, 0].hist(self.lepton_py[self.lepton_pdg == nu],
    #                         bins=n_bins, label=self.leptons_label[nu])
    #             ax[1, 1].hist(self.lepton_pz[self.lepton_pdg == nu],
    #                         bins=n_bins, label=self.leptons_label[nu])
    #     elif interaction:
    #         for nu in self.leptons:
    #             ax[0, 0].hist(self.lepton_energy[self.lepton_pdg == nu],
    #                         bins=n_bins, label=self.leptons_label[nu])
    #             ax[0, 1].hist(self.lepton_px[self.lepton_pdg == nu],
    #                         bins=n_bins, label=self.leptons_label[nu])
    #             ax[1, 0].hist(self.lepton_py[self.lepton_pdg == nu],
    #                         bins=n_bins, label=self.leptons_label[nu])
    #             ax[1, 1].hist(self.lepton_pz[self.lepton_pdg == nu],
    #                         bins=n_bins, label=self.leptons_label[nu])
    #     else:
    #         ax[0, 0].hist(self.lepton_energy,
    #                     bins=n_bins)
    #         ax[0, 1].hist(self.lepton_px,
    #                     bins=n_bins)
    #         ax[1, 0].hist(self.lepton_py,
    #                     bins=n_bins)
    #         ax[1, 1].hist(self.lepton_pz,
    #                     bins=n_bins)

    #     ax[0, 0].set_xlabel(r'$E$ [MeV]')
    #     ax[0, 1].set_xlabel(r'$p_x$ [MeV/c]')
    #     ax[1, 0].set_xlabel(r'$p_y$ [MeV/c]')
    #     ax[1, 1].set_xlabel(r'$p_z$ [MeV/c]')
    #     ax[0, 0].legend()
    #     fig.tight_layout()
    #     plt.tight_layout()
    #     fig.savefig('lepton_spectra.png')
    #     if show:
    #         plt.show()
    #     plt.clf

    # def plot_kinematics(self, show=False):
    #     print('Plotting the kinematics between incoming neutrino and outgoing lepton')
    #     n_bins = int(2**np.log10(self.lepton_energy.size))
    #     fig, ax3 = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    #     for lep in self.leptons:
    #         ma = self.lepton_pdg == lep
    #         for nu in self.neutrinos:
    #             ma &= self.neutrino_pdg == nu
    #             ax3[0].hist(
    #                 self.cos_theta[ma],
    #                 bins=n_bins,
    #                 label=self.neutrinos_label[nu] +
    #                 '--' +
    #                 self.leptons_label[lep])
    #             ax3[1].hist(
    #                 self.energy_transfer[ma],
    #                 bins=n_bins,
    #                 label=self.neutrinos_label[nu] +
    #                 '--' +
    #                 self.leptons_label[lep])
    #             ax3[2].hist(
    #                 self.momentum_transfer[ma],
    #                 bins=n_bins,
    #                 label=self.neutrinos_label[nu] +
    #                 '--' +
    #                 self.leptons_label[lep])
    #     ax3[0].set_xlabel(r'cos$\theta$')
    #     ax3[1].set_xlabel(r'$q_{0}$  [MeV]')
    #     ax3[2].set_xlabel(r'$q_{3}$  [MeV/c]')
    #     ax3[0].legend()
    #     fig.tight_layout()
    #     plt.tight_layout()
    #     fig.savefig('kineamtics.png')
    #     if show:
    #         plt.show()

    # def plot_event(self, evt_num):
    #     for i, event in enumerate(self.NUHEPMC):
    #         if i == evt_num:
    #             savefig(event, f'event_{evt_num}.svg')
