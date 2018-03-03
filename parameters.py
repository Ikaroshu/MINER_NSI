from scipy.integrate import quad
from scipy.interpolate import interp1d

from constants import *


class Epsilon:
    def __init__(self):
        self.epel = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.eper = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.epu = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.epd = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}

    def ee(self):
        epe = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        for i in epe:
            epe[i] = self.epel[i] + self.eper[i]
        return matrix([[epe['ee'], epe['em'], epe['et']],
                       [conj(epe['em']), epe['mm'], epe['mt']],
                       [conj(epe['et']), conj(epe['mt']), epe['tt']]]) + diag(array([1, 0, 0]))

    def eu(self):
        return matrix([[self.epu['ee'], self.epu['em'], self.epu['et']],
                       [conj(self.epu['em']), self.epu['mm'], self.epu['mt']],
                       [conj(self.epu['et']), conj(self.epu['mt']), self.epu['tt']]])

    def ed(self):
        return matrix([[self.epd['ee'], self.epd['em'], self.epd['et']],
                       [conj(self.epd['em']), self.epd['mm'], self.epd['mt']],
                       [conj(self.epd['et']), conj(self.epd['mt']), self.epd['tt']]])


def oparameters():
    return {'t12': 0.5763617589722192, 't13': 0.14819001778459273, 't23': 0.7222302630963306, 'd': 1.35 * pi,
            'd21': 7.37e-5 * (GevPerEv ** 2), 'd31': 2.5e-3 * (GevPerEv ** 2) + 7.37e-5 * (GevPerEv ** 2) / 2}


class Densityp:
    def __init__(self):
        density = genfromtxt('./solarnu/densitydata.txt', delimiter='  ')
        rs = density[:, 1]
        rho = density[:, 3]
        npd = rho * (density[:, 6] / massofh + density[:, 7] / massof4he * 2 + density[:, 8] / massof3he * 2 +
                     density[:, 9] / massof12c * 6 + density[:, 10] / massof14n * 7 + density[:, 11] / massof16o * 8) *\
            1e6 * ((MeterByJoule * GeVPerJoule) ** 3)
        nnd = rho * (density[:, 7] / massof4he * 2 + density[:, 8] / massof3he * 1 +
                     density[:, 9] / massof12c * 6 + density[:, 10] / massof14n * 7 + density[:, 11] / massof16o * 8) *\
            1e6 * ((MeterByJoule * GeVPerJoule) ** 3)
        nud = 2 * npd + nnd
        ndd = npd + 2 * nnd
        self.__nud_interp = interp1d(rs, nud)
        self.__ndd_interp = interp1d(rs, ndd)

    def nu(self, r):
        return self.__nud_interp(r)[()]

    def nd(self, r):
        return self.__ndd_interp(r)[()]

    def ne(self, r):
        return (2 * self.nu(r) - self.nd(r)) / 3


dp = Densityp()


def ne(r):
    return dp.ne(r)


def nu(r):
    return dp.nu(r)


def nd(r):
    return dp.nd(r)


def survp(ev, r, epsi, nu1, nu2, op):
    o23 = matrix([[1, 0, 0],
                  [0, cos(op['t23']), sin(op['t23'])],
                  [0, -sin(op['t23']), cos(op['t23'])]])
    u13 = matrix([[cos(op['t13']), 0, sin(op['t13']) * (e ** (- op['d'] * 1j))],
                  [0, 1, 0],
                  [-sin(op['t13'] * (e ** (op['d'] * 1j))), 0, cos(op['t13'])]])
    o12 = matrix([[cos(op['t12']), sin(op['t12']), 0],
                  [-sin(op['t12']), cos(op['t12']), 0],
                  [0, 0, 1]])
    umix = o23 * u13 * o12
    m = diag(array([0, op['d21'] / (2 * ev), op['d31'] / (2 * ev)]))
    v = sqrt(2) * gf * (ne(r) * epsi.ee() + nu(r) * epsi.eu() + nd(r) * epsi.ed())
    hvac = umix * m * umix.H

    def sorteig(w, vec):
        minindex = 0
        maxindex = 0
        for j in range(3):
            if w[minindex] > w[j]:
                minindex = j
        for j in range(3):
            if w[maxindex] < w[j]:
                maxindex = j
        midindex = 3 - minindex - maxindex
        avec = array(vec)
        return matrix([avec[:, minindex], avec[:, midindex], avec[:, maxindex]]).T

    wr, vecr = linalg.eigh(hvac + v)
    utr = sorteig(wr, vecr)
    ws, vecs = linalg.eigh(hvac)
    uts = sorteig(ws, vecs)
    res = 0
    for i in range(3):
        res += conj(utr[nu1, i]) * utr[nu1, i] * conj(uts[nu2, i]) * uts[nu2, i]
    return real(res)


class Detector:
    """One can choose from Ge, Si or Ar"""

    def __init__(self, ty):
        self.ty = ty
        if ty.lower() == 'ge':
            self.nIso = 5
            self.z = array([32, 32, 32, 32, 32])
            self.n = array([38, 40, 41, 42, 44])
            self.fraction = array([0.2123, 0.2766, 0.0773, 0.3594, 0.0744])
            self.m = array([65.13, 66.99, 67.92, 68.85, 70.72])
            self.erMin = 100 * (10 ** -9)
            self.erMax = 10 * (10 ** -6)
            self.background = 1
            self.bgUn = 0.1
        elif ty.lower() == 'si':
            self.nIso = 3
            self.z = array([14, 14, 14])
            self.n = array([14, 15, 16])
            self.fraction = array([0.9223, 0.0467, 0.031])
            self.m = array([26.06, 26.99, 27.92])
            self.erMin = 100 * (10 ** -9)
            self.erMax = 20 * (10 ** -6)
            self.background = 1
            self.bgUn = 0.1
        elif ty.lower() == 'ar':
            self.nIso = 1
            self.z = array([18])
            self.n = array([22])
            self.m = array([37.211])
            self.fraction = array([1])
            self.erMin = 30 * (10 ** -6)
            self.erMax = 1e-4
            self.background = 5e-3
            self.bgUn = 0.05
        elif ty.lower() == 'csi':
            self.nIso = 2
            self.z = array([55, 53])
            self.n = array([78, 74])
            self.fraction = array([0.5, 0.5])
            self.m = array([123.8, 118.21])
            self.erMin = 4.25 * (10 ** -6)
            self.erMax = 26 * (10 ** -6)
            self.background = 5e-3
            self.bgUn = 0.05
        elif ty.lower() == 'xe':
            self.nIso = 7
            self.z = array([54])
            self.n = array([78])
            self.m = array([122.3])
            self.fraction = array([1.0])
        elif ty.lower() == "nai":
            self.nIso = 2
            self.z = array([11, 53])
            self.n = array([12, 74])
            self.fraction = array([0.5, 0.5])
            self.m = array([21.42, 118.21])
            self.erMin = 2 * (10 ** -6)
            self.erMax = 4e-5
            self.background = 5e-3
            self.bgUn = 0.05
        else:
            raise Exception("No such detector defined in code yet.")


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
            self.flUn = 0.1
            self.__norm = 1.05 * (10 ** 11) * ((MeterByJoule * GeVPerJoule) ** 2)
        elif ty.lower() == 'solar':
            b8 = genfromtxt('./b8.csv', delimiter=',')
            self.__b8x = b8[:, 0] * 1e-3
            self.__b8y = b8[:, 1] * ((100 * GeVPerJoule * MeterByJoule) ** 2) * 5.58e6 * 1e3  # per MeV -> per GeV!!
            self.__b8interp = interp1d(self.__b8x, self.__b8y)
            f17 = genfromtxt('./f17.csv', delimiter=',')
            self.__f17x = f17[:, 0] * 1e-3
            self.__f17y = f17[:, 1] * ((100 * GeVPerJoule * MeterByJoule) ** 2) * 5.52e6 * 1e3
            self.__f17interp = interp1d(self.__f17x, self.__f17y)
            hep = genfromtxt('./hep.csv', delimiter=',')
            self.__hepx = hep[:, 0] * 1e-3
            self.__hepy = hep[:, 1] * ((100 * GeVPerJoule * MeterByJoule) ** 2) * 8.04e3 * 1e3
            self.__hepinterp = interp1d(self.__hepx, self.__hepy)
            n13 = genfromtxt('./n13.csv', delimiter=',')
            self.__n13x = n13[:, 0] * 1e-3
            self.__n13y = n13[:, 1] * ((100 * GeVPerJoule * MeterByJoule) ** 2) * 2.96e8 * 1e3
            self.__n13interp = interp1d(self.__n13x, self.__n13y)
            o15 = genfromtxt('./o15.csv', delimiter=',')
            self.__o15x = o15[:, 0] * 1e-3
            self.__o15y = o15[:, 1] * ((100 * GeVPerJoule * MeterByJoule) ** 2) * 2.23e8 * 1e3
            self.__o15interp = interp1d(self.__o15x, self.__o15y)
            pp = genfromtxt('./pp.csv', delimiter=',')
            self.__ppx = pp[:, 0] * 1e-3
            self.__ppy = pp[:, 1] * ((100 * GeVPerJoule * MeterByJoule) ** 2) * 5.98e10 * 1e3
            self.__ppinterp = interp1d(self.__ppx, self.__ppy)
            self.evMax = 20e-3
        else:
            raise Exception("No such flux in code yet.")

    def flux(self, ev, epsi=None, flav=None, op=None):
        if self.ty == 'sns':
            return self.nuef(ev)
        # return self.__fl(ev)[()] * self.__nuflux1m / self.__norm
        if self.ty == 'solar':
            res = 0
            res += self.__b8interp(ev)[()] if self.__b8x[0] <= ev <= self.__b8x[self.__b8x.shape[0] - 1] else 0
            res += self.__f17interp(ev)[()] if self.__f17x[0] <= ev <= self.__f17x[self.__f17x.shape[0] - 1] else 0
            res += self.__hepinterp(ev)[()] if self.__hepx[0] <= ev <= self.__hepx[self.__hepx.shape[0] - 1] else 0
            res += self.__n13interp(ev)[()] if self.__n13x[0] <= ev <= self.__n13x[self.__n13x.shape[0] - 1] else 0
            res += self.__o15interp(ev)[()] if self.__o15x[0] <= ev <= self.__o15x[self.__o15x.shape[0] - 1] else 0
            res += self.__ppinterp(ev)[()] if self.__ppx[0] <= ev <= self.__ppx[self.__ppx.shape[0] - 1] else 0
            res *= survp(ev, 0.1, epsi, 0, flav, op)
            return res
        return exp(0.87 - 0.16 * (ev * 1000) - 0.091 * ((ev * 1000) ** 2)) / 0.005323608902707208 * self.__nuflux1m

    def fint(self, er, m, epsi=None, flav=None, op=None):
        if self.ty == 'sns':
            return self.nuefint(er, m)
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        # p = self.__t0[where(self.__t0 >= emin)]
        # if p.shape[0] == 0:
        #     return 0.0
        # return quad(self.flux, emin, self.evMax, limit=2 * p.shape[0], points=p)[0]
        if self.ty == 'solar':
            res = 0
            res += quad(self.flux, emin, self.evMax, args=(epsi, flav, op))[0]  # , points=p, limit=2 * p.shape[0]
            res += 1.44e8 * ((100 * GeVPerJoule * MeterByJoule) ** 2) * survp(1.439e-3, 0.1, epsi, 0, flav, op) \
                if emin < 1.439e-3 else 0  # pep
            res += 5e9 * ((100 * GeVPerJoule * MeterByJoule) ** 2) * survp(0.8613e-3, 0.1, epsi, 0, flav, op) \
                if emin < 0.8613e-3 else 0  # be7
            return res
        return quad(self.flux, emin, self.evMax)[0]

    def fintinv(self, er, m, epsi=None, flav=None, op=None):
        if self.ty == 'sns':
            return self.nuefinv(er, m)
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        # p = self.__t0[where(self.__t0 >= emin)]
        # if p.shape[0] == 0:
        #     return 0.0

        def finv(ev):
            return self.flux(ev, epsi, flav, op) / ev

        # return quad(finv, emin, self.evMax, limit=2 * p.shape[0], points=p)[0]
        if self.ty == 'solar':
            res = 0
            res += quad(finv, emin, self.evMax)[0]
            res += 1.44e8 * ((100 * GeVPerJoule * MeterByJoule) ** 2) / 1.439e-3 * \
                survp(1.439e-3, 0.1, epsi, 0, flav, op) if emin < 1.439e-3 else 0
            res += 5e9 * ((100 * GeVPerJoule * MeterByJoule) ** 2) / 0.8613e-3 * \
                survp(0.8613e-3, 0.1, epsi, 0, flav, op) if emin < 0.8613e-3 else 0
            return res
        return quad(finv, emin, self.evMax)[0]

    def fintinvs(self, er, m, epsi=None, flav=None, op=None):
        if self.ty == 'sns':
            return self.nuefinvs(er, m)
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        # p = self.__t0[where(self.__t0 >= emin)]
        # if p.shape[0] == 0:
        #     return 0.0

        def finvs(ev):
            return self.flux(ev, epsi, flav, op) / (ev ** 2)

        # return quad(finvs, emin, self.evMax, limit=2 * p.shape[0], points=p)[0]
        if self.ty == 'solar':
            res = 0
            res += quad(finvs, emin, self.evMax)[0]
            res += 1.44e8 * ((100 * GeVPerJoule * MeterByJoule) ** 2) / (1.439e-3 ** 2) * \
                   survp(1.439e-3, 0.1, epsi, 0, flav, op) if emin < 1.439e-3 else 0
            res += 5e9 * ((100 * GeVPerJoule * MeterByJoule) ** 2) / (0.8613e-3 ** 2) * \
                   survp(0.8613e-3, 0.1, epsi, 0, flav, op) if emin < 0.8613e-3 else 0
            return res
        return quad(finvs, emin, self.evMax)[0]

    def nuef(self, ev):
        return (3 * ((ev * 1000 / (2 / 3 * 52)) ** 2) - 2 * (
            (ev * 1000 / (2 / 3 * 52)) ** 3)) / 29.25 * 1000 * self.__norm  # in GeV

    def nuefint(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        if type(emin) != ndarray:
            return quad(self.nuef, emin, self.evMax)[0]
        re = zeros_like(emin)
        for i in range(emin.shape[0]):
            re[i] = quad(self.nuef, emin[i], self.evMax)[0]
        return re
        # return quad(self.nuef, emin, self.evMax)[0]

    def nuefinv(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)

        def finv(ev):
            return self.nuef(ev) / ev

        if type(emin) != ndarray:
            return quad(finv, emin, self.evMax)[0]

        re = zeros_like(emin)
        for i in range(emin.shape[0]):
            re[i] = quad(finv, emin[i], self.evMax)[0]
        return re
        # return quad(finv, emin, self.evMax)[0]

    def nuefinvs(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)

        def finvs(ev):
            return self.nuef(ev) / (ev ** 2)

        if type(emin) != ndarray:
            return quad(finvs, emin, self.evMax)[0]

        re = zeros_like(emin)
        for i in range(emin.shape[0]):
            re[i] = quad(finvs, emin[i], self.evMax)[0]
        return re
        # return quad(finvs, emin, self.evMax)[0]

    def numf(self, ev):
        return (3 * ((ev * 1000 / 52) ** 2) - 2 * ((ev * 1000 / 52) ** 3)) / 26 * 1000 * self.__norm

    def numfint(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        if type(emin) != ndarray:
            return quad(self.numf, emin, self.evMax)[0]
        re = zeros_like(emin)
        for i in range(emin.shape[0]):
            re[i] = quad(self.numf, emin[i], self.evMax)[0]
        return re
        # return quad(self.numf, emin, self.evMax)[0]

    def numfinv(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)

        def finv(ev):
            return self.numf(ev) / ev

        if type(emin) != ndarray:
            return quad(finv, emin, self.evMax)[0]

        re = zeros_like(emin)
        for i in range(emin.shape[0]):
            re[i] = quad(finv, emin[i], self.evMax)[0]
        return re
        # return quad(finv, emin, self.evMax)[0]

    def numfinvs(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)

        def finvs(ev):
            return self.numf(ev) / (ev ** 2)

        if type(emin) != ndarray:
            return quad(finvs, emin, self.evMax)[0]
        re = zeros_like(emin)
        for i in range(emin.shape[0]):
            re[i] = quad(finvs, emin[i], self.evMax)[0]
        return re
        # return quad(finvs, emin, self.evMax)[0]

    @staticmethod
    def nupf(ev):
        return math.inf if ev == 0.029 else 0

    def nupfint(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        if type(emin) != ndarray:
            return self.__norm if emin <= 0.029 else 0
        re = zeros_like(emin)
        for i in range(emin.shape[0]):
            re[i] = self.__norm if emin[i] <= 0.029 else 0
        return re
        # return self.__norm if emin <= 0.029 else 0

    def nupfinv(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        if type(emin) != ndarray:
            return self.__norm / 0.029 if emin <= 0.029 else 0
        re = zeros_like(emin)
        for i in range(emin.shape[0]):
            re[i] = self.__norm/0.029 if emin[i] <= 0.029 else 0
        return re
        # return self.__norm / 0.029 if emin <= 0.029 else 0

    def nupfinvs(self, er, m):
        emin = 0.5 * (sqrt(er ** 2 + 2 * er * m) + er)
        if type(emin) != ndarray:
            return self.__norm / (0.029 ** 2) if emin <= 0.029 else 0
        re = zeros_like(emin)
        for i in range(emin.shape[0]):
            re[i] = self.__norm/ (0.029 ** 2) if emin[i] <= 0.029 else 0
        return re
        # return self.__norm / (0.029 ** 2) if emin <= 0.029 else 0


def couplings():
    return {'uee': 0, 'umm': 0, 'utt': 0, 'uem': 0, 'uet': 0, 'umt': 0,
            'dee': 0, 'dmm': 0, 'dtt': 0, 'dem': 0, 'det': 0, 'dmt': 0,
            'elee': 0, 'elmm': 0, 'eltt': 0, 'elem': 0, 'elet': 0, 'elmt': 0,
            'eree': 0, 'ermm': 0, 'ertt': 0, 'erem': 0, 'eret': 0, 'ermt': 0}
