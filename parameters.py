from numpy import *
from scipy.interpolate import interp1d
from scipy.integrate import quad
from constants import *


class Detector:
    """One can choose from Ge or Si"""
    def __init__(self, ty):
        self.ty = ty
        self.erMin = 0.4*(10**-6)
        self.erMax = 1.1*(10**-6)    # GeV
        self.background = 100   # dru
        self.bgUn = 0.1
        if ty.lower() == 'ge':
            # self.nIso = 5
            self.z = 32     # array([32, 32, 32, 32, 32])
            self.n = 40    # array([38, 40, 41, 42, 44])
            # self.fraction = array([0.2123, 0.2766, 0.0773, 0.3594, 0.0744])
            self.m = 66.99    # array([65.13, 66.99, 67.92, 68.85, 70.72])
        elif ty.lower() == 'si':
            # self.nIso = 3
            self.z = 14     # array([14, 14, 14])
            self.n = 14     # array([14, 15, 16])
            # self.fraction = array([0.9223, 0.0467, 0.031])
            self.m = 26.06   # array([26.06, 26.99, 27.92])
        else:
            raise Exception("No such detector defined in code yet.")

    def set_ermin(self, er):
        self.erMin = er

    def set_ermax(self, er):
        self.erMax = er

    def set_background(self, bg):
        self.background = bg


class Flux:
    """reactor flux"""
    def __init__(self, ty):
        if ty.lower() == 'reactor':
            self.evMin = 0.0
            self.evMax = 0.01   # GeV
            self.flUn = 0.02
            self.__tb = genfromtxt('reactor_flux.csv', delimiter=",")
            self.__t0 = self.__tb[:, 0]
            self.__t1 = self.__tb[:, 1]
            self.__fl = interp1d(self.__t0, self.__t1)
            def f(ev):
                return self.__fl(ev)[()]
            self.__norm = quad(f, self.evMin, self.evMax, limit=2*self.__t0.shape[0], points=self.__t0)[0]
            fpers = 3.0921*(10**16)     # antineutrinos per fission
            nuperf = 6.14102
            self.__nuflux1m = nuperf*fpers/(4*pi)*((MeterByJoule*GeVPerJoule)**2)
        else:
            raise Exception("No such flux in code yet.")

    def flux(self, ev):
        return self.__fl(ev)[()]*self.__nuflux1m/self.__norm

    def fint(self, er, m):
        emin = 0.5*(sqrt(er**2+2*er*m)+er)
        p = self.__t0[where(self.__t0 >= emin)]
        if p.shape[0] == 0:
            return 0.0
        return quad(self.flux, emin, self.evMax, limit=2*p.shape[0], points=p)[0]

    def fintinv(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        p = self.__t0[where(self.__t0 >= emin)]
        if p.shape[0] == 0:
            return 0.0
        def finv(ev):
            return self.flux(ev)/ev
        return quad(finv, emin, self.evMax, limit=2 * p.shape[0], points=p)[0]

    def fintinvs(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        p = self.__t0[where(self.__t0 >= emin)]
        if p.shape[0] == 0:
            return 0.0
        def finvs(ev):
            return self.flux(ev)/(ev**2)
        return quad(finvs, emin, self.evMax, limit=2 * p.shape[0], points=p)[0]


def couplings():
    return {'uee': 0, 'umm': 0, 'utt': 0, 'uem': 0, 'uet': 0, 'umt': 0,
            'dee': 0, 'dmm': 0, 'dtt': 0, 'dem': 0, 'det': 0, 'dmt': 0}
