import pyhepmc
import numpy as np
from particle import literals as lp
from pyhepmc.view import to_dot
from pyhepmc.view import savefig
from itertools import repeat
import inspect


class NuHepMC:
    def __init__(self, filename='neut.6.sf.hepmc3', neut5=False, test=False):

        self.NEUT5 = neut5

        self.NUHEPMC = pyhepmc.open(filename)
        self.VERSION = self.NUHEPMC.read().run_info.tools[0].version
        self.N_EVENTS = self.NUHEPMC.read(
        ).run_info.attributes['NuHepMC.Exposure.NEvents'].astype(int) - 1
        if test:
            self.N_EVENTS = 10000

        self.neutrinos_pid = [-16, -14, -12, 12, 14, 16]
        self.charged_leptons_pid = [-15, -13, -11, 11, 12, 15]
        self.leptons_pid = self.neutrinos_pid + self.charged_leptons_pid

        self.interaction_mode_labels = {
            -11: r'CC RES $n\rightarrow\pi^{-}$', -12:
            r'CC RES $p\rightarrow\pi^{0}$', -13:
            r'CC RES $p\rightarrow\pi^{-}$', -15: 'CC Dif', -16: 'CC Coh.', -
            17: r'CC 1$\gamma$', -1: 'CCQE bound p', -21: r'CC N$\pi$', -
            22: r'CC $\eta$', -23: r'CC $K$', -26: 'CC DIS',
            -2: 'CC 2p2h', -31: r'NC RES $n\rightarrow\pi^{0}$', -32:
            r'NC RES $p\rightarrow \pi^{0}$', -33:
            r'NC RES $p\rightarrow \pi^{-}$', -34:
            r'NC RES $n\rightarrow\pi^{+}$', -35: 'NC Dif', -36: 'NC Coh', -38:
            r'NC 1$\gamma$ n', -39: r'NC 1$\gamma$ p', -41: r'NC N$\pi$', -42:
            r'NC $\eta$ n', -43: r'NC $\eta$ p', -44: r'NC $K$ n', -45:
            r'NC $K$ p', -46: 'ND DIS', -51: 'NC elastic bound p',
            -52: 'NC elastic n', 11: r'CC RES $p\rightarrow\pi^{+}$', 12: r'CC RES $p\rightarrow\pi^{0}$',
            13: r'CC RES $n\rightarrow\pi^{+}$', 15: 'CC Dif', 16: 'CC Coh.',
            17: r'CC 1$\gamma$', 1: 'CCQE', 21: r'CC N$\pi$', 22: r'CC $\eta$',
            23: r'CC $K$', 26: 'CC DIS', 2: 'CC 2p2h',
            31: r'NC RES $n \rightarrow \pi^{0}$', 32:
            r'NC RES $p \rightarrow \pi^{0}$', 33:
            r'NC RES $p \rightarrow \pi^{-}$', 34:
            r'NC RES $n \rightarrow \pi^{+}$', 35: 'NC Dif', 36: 'NC Coh', 38:
            r'NC 1$\gamma$ n', 39: r'NC 1$\gamma$ p', 41: r'NC N$\pi$', 42:
            r'NC $\eta$ n', 43: r'NC $\eta$ p', 44: r'NC $K$ n', 45:
            r'NC $K$ p', 46: 'ND DIS', 51: 'NC elastic bound p',
            52: 'NC elastic n', }

        self.neutrinos_label = {
            -16: r'$\overline{\nu}_{\tau}$', -14: r'$\overline{\nu}_{\mu}$', -
            12: r'$\overline{\nu}_{e}$', 12: r'$\nu_{e}$', 14: r'$\nu_{\mu}$',
            16: r'$\nu_{\tau}$'}
        self.charged_leptons_label = {
            -15: r'$\tau^{+}$', -13: r'$\mu^{+}$', -11: r'$e^{+}$', 11: r'$e$',
            13: r'$\mu$', 15: r'$\tau$'}
        self.leptons_label = self.neutrinos_label | self.charged_leptons_label

        self.neutrino_pdg = np.zeros(self.N_EVENTS)
        self.neutrino_energy = np.zeros(self.N_EVENTS)
        self.neutrino_momentum = np.zeros(self.N_EVENTS)
        self.neutrino_px = np.zeros(self.N_EVENTS)
        self.neutrino_py = np.zeros(self.N_EVENTS)
        self.neutrino_pz = np.zeros(self.N_EVENTS)
        self.neutrino_mass = np.zeros(self.N_EVENTS)

        self.interaction_mode = np.zeros(self.N_EVENTS)

        self.lepton_pdg = np.zeros(self.N_EVENTS)
        self.lepton_energy = np.zeros(self.N_EVENTS)
        self.lepton_momentum = np.zeros(self.N_EVENTS)
        self.lepton_px = np.zeros(self.N_EVENTS)
        self.lepton_py = np.zeros(self.N_EVENTS)
        self.lepton_pz = np.zeros(self.N_EVENTS)
        self.lepton_mass = np.zeros(self.N_EVENTS)

    def get_variables(self, *particles):
        for i, event in enumerate(self.NUHEPMC):
            if i == self.N_EVENTS:
                break
            print(f'Processing event {i}')
            print('------------------------------')
            for particle in particles:
                self.add_event(i, event, particle)
            self.add_event(i, event, 'interaction_mode')
            print('==============================\n')

        if 'neutrinos' in particles and 'leptons' in particles:
            print(r'Computing kinematics')
            self.get_cos_theta()
            self.get_energy_transfer()
            self.get_momentum_transfer()

    def get_energy_transfer(self):
        self.energy_transfer = self.neutrino_energy - self.lepton_energy

    def get_momentum_transfer(self):
        # self.momentum_transfer = np.zeros_like(self.neutrino_px)
        # for i, (a, b,) in enumerate(zip(self.neutrino_pz, self.lepton_pz)):
        #     self.momentum_transfer[i] = a - b
        self.momentum_transfer = self.neutrino_momentum - \
            self.lepton_momentum * self.cos_theta

    def get_cos_theta(self):
        self.cos_theta = (
            self.neutrino_px *
            self.lepton_px +
            self.neutrino_py *
            self.lepton_py +
            self.neutrino_pz *
            self.lepton_pz)
        self.cos_theta /= np.sqrt(self.neutrino_px *
                                  self.neutrino_px +
                                  self.neutrino_py *
                                  self.neutrino_py +
                                  self.neutrino_pz *
                                  self.neutrino_pz)
        self.cos_theta /= np.sqrt(self.lepton_px *
                                  self.lepton_px +
                                  self.lepton_py *
                                  self.lepton_py +
                                  self.lepton_pz *
                                  self.lepton_pz)

    def add_event(self, index, event, item):
        print(f'Adding variables for {item}')
        if item == 'neutrinos':
            p = event.numpy.particles
            ma = p.status == 4
            ma &= self._matching(p.pid, self.neutrinos_pid)
            _e = p.e
            _px = p.px
            _py = p.py
            _pz = p.pz
            _pid = p.pid
            _mass = p.generated_mass
            self.neutrino_energy[index] = _e[ma]
            self.neutrino_px[index] = _px[ma]
            self.neutrino_py[index] = _py[ma]
            self.neutrino_pz[index] = _pz[ma]
            self.neutrino_pdg[index] = _pid[ma]
            self.neutrino_mass[index] = _mass[ma]
            self.neutrino_momentum[index] = np.sqrt(_e[ma]**2 - _mass[ma]**2)

        elif item == 'leptons':
            p = event.numpy.particles
            ma = p.id == 4
            _e = p.e
            _px = p.px
            _py = p.py
            _pz = p.pz
            _pid = p.pid
            _mass = p.generated_mass
            self.lepton_energy[index] = _e[ma]
            self.lepton_px[index] = _px[ma]
            self.lepton_py[index] = _py[ma]
            self.lepton_pz[index] = _pz[ma]
            self.lepton_pdg[index] = _pid[ma]
            self.lepton_mass[index] = _mass[ma]
            self.lepton_momentum[index] = np.sqrt(_e[ma]**2 - _mass[ma]**2)

        elif item == 'interaction_mode':
            if self.NEUT5:
                _mode = int(str(event.attributes['NEUT.Mode']))
            else:
                _mode = int(str(event.attributes['ProcID']))
            self.interaction_mode[index] = _mode

    def _matching(self, iter1, iter2):
        result = np.zeros_like(iter1, dtype=bool)
        for i, it1 in enumerate(iter1):
            for it2 in iter2:
                if it1 == it2:
                    result[i] = True
        return result
