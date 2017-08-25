from scipy.integrate import quad
from scipy.interpolate import interp1d

from constants import *


class Detector:
    """One can choose from Ge, Si or Ar"""

    def __init__(self, ty):
        self.ty = ty
        self.erMin = 0.4 * (10 ** -6)
        self.erMax = 1.1 * (10 ** -6)  # GeV
        self.background = 100  # dru
        self.bgUn = 0.1
        if ty.lower() == 'ge':
            self.nIso = 5
            self.z = array([32, 32, 32, 32, 32])
            self.n = array([38, 40, 41, 42, 44])
            self.fraction = array([0.2123, 0.2766, 0.0773, 0.3594, 0.0744])
            self.m = array([65.13, 66.99, 67.92, 68.85, 70.72])
        elif ty.lower() == 'si':
            self.nIso = 3
            self.z = array([14, 14, 14])
            self.n = array([14, 15, 16])
            self.fraction = array([0.9223, 0.0467, 0.031])
            self.m = array([26.06, 26.99, 27.92])
        elif ty.lower() == 'ar':
            self.nIso = 1
            self.z = 18
            self.n = 22
            self.m = 37.211
            self.fraction = 1
        elif ty.lower() == 'csi':
            self.nIso = 2
            self.z = array([55, 53])
            self.n = array([78, 74])
            self.fraction = array([0.5, 0.5])
            self.m = array([123.8, 118.21])
        else:
            raise Exception("No such detector defined in code yet.")

    def set_ermin(self, er):
        self.erMin = er

    def set_ermax(self, er):
        self.erMax = er

    def set_background(self, bg):
        self.background = bg


class Flux:
    """fluxes"""

    def __init__(self, ty):
        self.ty = ty
        if ty.lower() == 'reactor':
            self.evMin = 0.0
            self.evMax = 0.01  # GeV
            self.flUn = 0.02
            # self.__tb = genfromtxt('reactor_flux.csv', delimiter=",")
            # self.__t0 = self.__tb[:, 0]
            # self.__t1 = self.__tb[:, 1]
            # self.__fl = interp1d(self.__t0, self.__t1)
            #
            # def f(ev):
            #     return self.__fl(ev)[()]
            #
            # self.__norm = quad(f, self.evMin, self.evMax, limit=2 * self.__t0.shape[0], points=self.__t0)[0]
            fpers = 3.0921 * (10 ** 16)  # antineutrinos per fission
            nuperf = 6.14102
            self.__nuflux1m = nuperf * fpers / (4 * pi) * ((MeterByJoule * GeVPerJoule) ** 2)
        elif ty.lower() == 'sns':
            self.evMin = 0
            self.evMax = 52 * (10 ** -3)
            self.flUn = 0.02
            self.__norm = 4.3 * (10 ** 11) * ((MeterByJoule * GeVPerJoule) ** 2)
        else:
            raise Exception("No such flux in code yet.")

    def flux(self, ev):
        if self.ty == 'sns':
            return self.nuef(ev)
        # return self.__fl(ev)[()] * self.__nuflux1m / self.__norm
        return exp(0.87 - 0.16 * (ev * 1000) - 0.091 * ((ev * 1000) ** 2)) / 0.005323608902707208 * self.__nuflux1m

    def fint(self, er, m):
        if self.ty == 'sns':
            return self.nuefint(er, m)
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        # p = self.__t0[where(self.__t0 >= emin)]
        # if p.shape[0] == 0:
        #     return 0.0
        # return quad(self.flux, emin, self.evMax, limit=2 * p.shape[0], points=p)[0]
        return quad(self.flux, emin, self.evMax)[0]

    def fintinv(self, er, m):
        if self.ty == 'sns':
            return self.nuefinv(er, m)
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        # p = self.__t0[where(self.__t0 >= emin)]
        # if p.shape[0] == 0:
        #     return 0.0

        def finv(ev):
            return self.flux(ev) / ev

        # return quad(finv, emin, self.evMax, limit=2 * p.shape[0], points=p)[0]
        return quad(finv, emin, self.evMax)[0]

    def fintinvs(self, er, m):
        if self.ty == 'sns':
            return self.nuefinvs(er, m)
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        # p = self.__t0[where(self.__t0 >= emin)]
        # if p.shape[0] == 0:
        #     return 0.0

        def finvs(ev):
            return self.flux(ev) / (ev ** 2)

        # return quad(finvs, emin, self.evMax, limit=2 * p.shape[0], points=p)[0]
        return quad(finvs, emin, self.evMax)[0]

    def nuef(self, ev):
        return (3 * ((ev * 1000 / (2 / 3 * 52)) ** 2) - 2 * (
            (ev * 1000 / (2 / 3 * 52)) ** 2)) / 29.25 * 1000 * self.__norm  # in GeV

    def nuefint(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        return quad(self.nuef, emin, self.evMax)[0]

    def nuefinv(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)

        def finv(ev):
            return self.nuef(ev) / ev

        return quad(finv, emin, self.evMax)[0]

    def nuefinvs(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)

        def finvs(ev):
            return self.nuef(ev) / (ev ** 2)

        return quad(finvs, emin, self.evMax)[0]

    def numf(self, ev):
        return (3 * ((ev * 1000 / 52) ** 2) - 2 * ((ev * 1000 / 52) ** 2)) / 26 * 1000 * self.__norm

    def numfint(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        return quad(self.numf, emin, self.evMax)[0]

    def numfinv(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)

        def finv(ev):
            return self.numf(ev) / ev

        return quad(finv, emin, self.evMax)[0]

    def numfinvs(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)

        def finvs(ev):
            return self.numf(ev) / (ev ** 2)

        return quad(finvs, emin, self.evMax)[0]

    @staticmethod
    def nupf(ev):
        return math.inf if ev == 0.029 else 0

    def nupfint(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        return self.__norm if emin <= 0.029 else 0

    def nupfinv(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        return self.__norm / 0.029 if emin <= 0.029 else 0

    def nupfinvs(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        return self.__norm / (0.029 ** 2) if emin <= 0.029 else 0


def couplings():
    return {'uee': 0, 'umm': 0, 'utt': 0, 'uem': 0, 'uet': 0, 'umt': 0,
            'dee': 0, 'dmm': 0, 'dtt': 0, 'dem': 0, 'det': 0, 'dmt': 0}
