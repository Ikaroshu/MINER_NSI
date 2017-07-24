from scipy.optimize import minimize

from events import *


class Chisquare:
    """solution for chisquares"""

    def __init__(self, det, fx, g, mv, expo, nbins, div='linear'):
        self.fx = fx
        self.det = det
        self.g = g
        self.expo = expo
        self.mv = mv
        self.th = det.erMin
        # self.smt = totoal(expo, mv, det, fx, couplings())
        # self.bgt = det.background*expo*(det.erMax-det.erMin)*1000    # 因为dru是per kev
        if div == 'linear':
            self.ebin = linspace(self.th, self.det.erMax, nbins + 1)
        elif div == 'log':
            self.ebin = logspace(log10(self.th), log10(self.det.erMax), nbins + 1)
        else:
            raise Exception("div = linear or log")
        self.binned_sm = \
            array([binned_events(self.ebin[i], self.ebin[i + 1], self.expo, self.mv, self.det, self.fx, couplings())
                   for i in range(self.ebin.shape[0] - 1)])
        self.binned_bg = array([binned_background(self.ebin[i], self.ebin[i + 1], self.det, self.expo)
                                for i in range(self.ebin.shape[0] - 1)])
        self.binned_nsi = \
            array([binned_events(self.ebin[i], self.ebin[i + 1], self.expo, self.mv, self.det, self.fx, self.g)
                   for i in range(self.ebin.shape[0] - 1)])

    # def tmus(self):
    #     if self.th != self.det.erMin:
    #         self.smt = totoal(self.expo, self.mv, self.det, self.fx, couplings())
    #         self.bgt = self.det.backgound*self.expo*(self.det.erMax-self.det.erMin)*1000
    #         self.th = self.det.erMin
    #     nsit = totoal(self.expo, self.mv, self.det, self.fx, self.g)
    #     lg0 = -(self.smt+self.bgt) + (nsit+self.bgt)*log(self.smt+self.bgt)
    #     lgnu = -(nsit+self.bgt) + (nsit+self.bgt)*log(nsit+self.bgt)
    #     return -2*(lg0-lgnu)

    def lgl(self, nf, nb, mu, bsm, sm, bg):
        nui = nf * (mu * bsm + sm + nb * bg)
        ni = bsm + sm + bg
        return sum(ni * log(nui) - nui - ((nf - 1) ** 2) / (2 * (self.fx.flUn ** 2)) - ((nb - 1) ** 2) / (
            2 * (self.det.bgUn ** 2)))  # n! is constant

    def findl0(self, bsm, sm, bg):
        def f(x):
            return -self.lgl(x[0], x[1], 0, bsm, sm, bg)  # minimize

        def f_der(x):
            ni = bsm + sm + bg
            nui = x[0] * (sm + x[1] * bg)
            der = zeros_like(x)
            der[0] = sum((ni / nui - 1) * (sm + x[1] * bg) - (x[0] - 1) / (self.fx.flUn ** 2))
            der[1] = sum((ni / nui - 1) * x[0] * bg - (x[1] - 1) / (self.det.bgUn ** 2))
            return -der

        def f_hess(x):
            ni = bsm + sm + bg
            nui = x[0] * (sm + x[1] * bg)
            hess = array([[0.0, 0.0], [0.0, 0.0]])
            hess[0][0] = -sum(ni / (nui ** 2) * ((sm + x[1] * bg) ** 2) - 1 / (self.fx.flUn ** 2))
            hess[1][1] = -sum(ni / (nui ** 2) * ((x[0] * bg) ** 2) - 1 / (self.det.bgUn ** 2))
            hess[0][1] = -sum(ni / (nui ** 2) * x[0] * bg * (sm + x[1] * bg) + (ni / nui - 1) * bg)
            hess[1][0] = -sum(ni / (nui ** 2) * (sm + x[1] * bg) * x[0] * bg + (ni / nui - 1) * bg)
            return -hess

        res = minimize(f, array([1.0, 1.0]), method='Newton-CG', jac=f_der, hess=f_hess)
        if not res.success:
            print(self.g, self.ebin.shape[0], self.mv, self.det.ty)
            print(res.message)
            raise Exception("optimization failed!")
        # print(res.x)
        return -res.fun

    def tmu_binned(self):
        if self.th != self.det.erMin:
            self.binned_sm = \
                array([binned_events(self.ebin[i], self.ebin[i + 1], self.expo, self.mv, self.det, self.fx, couplings())
                       for i in range(self.ebin.shape[0] - 1)])
            self.binned_bg = array([binned_background(self.ebin[i], self.ebin[i + 1], self.det, self.expo)
                                    for i in range(self.ebin.shape[0] - 1)])
            self.th = self.det.erMin
        binned_nsi = \
            array([binned_events(self.ebin[i], self.ebin[i + 1], self.expo, self.mv, self.det, self.fx, self.g)
                   for i in range(self.ebin.shape[0] - 1)])
        bsm = binned_nsi - self.binned_sm
        lgl0 = self.findl0(bsm, self.binned_sm, self.binned_bg)
        lglmu = self.lgl(1, 1, 1, bsm, self.binned_sm, self.binned_bg)
        return -2 * (lgl0 - lglmu)

    def tmuc(self, coup):
        if self.th != self.det.erMin:
            self.binned_sm = \
                array([binned_events(self.ebin[i], self.ebin[i + 1], self.expo, self.mv, self.det, self.fx, couplings())
                       for i in range(self.ebin.shape[0] - 1)])
            self.binned_bg = array([binned_background(self.ebin[i], self.ebin[i + 1], self.det, self.expo)
                                    for i in range(self.ebin.shape[0] - 1)])
            self.th = self.det.erMin
        sm = array([binned_events(self.ebin[i], self.ebin[i + 1], self.expo, self.mv, self.det, self.fx, coup)
                    for i in range(self.ebin.shape[0] - 1)])
        bsm = self.binned_nsi - sm
        # print('bnsi', self.binned_nsi)
        lgl0 = self.findl0(bsm, sm, self.binned_bg)
        lglmu = self.lgl(1, 1, 1, bsm, sm, self.binned_bg)
        return -2 * (lgl0 - lglmu)

        # def tmu_1bin_analytic(self):
        #     if self.th != self.det.erMin:
        #         self.smt = totoal(self.expo, self.mv, self.det, self.fx, couplings())
        #         self.bgt = self.det.backgound*self.expo*(self.det.erMax-self.det.erMin)*1000
        #         self.th = self.det.erMin
        #     bg = self.bgt
        #     sm = self.smt
        #     sgf = self.fx.flUn**2
        #     sgb = self.det.bgUn**2
        #     n = totoal(self.expo, self.mv, self.det, self.fx, self.g)+bg
        #     def func(x):
        #         nu = x[0]*(sm+x[1]*bg)
        #         f1 = (n/nu-1)*(sm+x[1]*bg)-(x[0]-1)/sgf
        #         f2 = (n/nu-1)*x[0]*bg-(x[1]-1)/sgb
        #         return f1, f2
        #     nf, nb = fsolve(func, (1, 1))
        #     # print(nf, nb)
        #     # print(func((nf, nb)))
        #     nu0 = nf*(sm+nb*bg)
        #     lgl0 = n*log(nu0)-nu0-((nf-1)**2)/(2*sgf)-((nb-1)**2)/(2*sgb)
        #     numu = n
        #     lglmu =n*log(numu)-numu
        #     return -2*(lgl0-lglmu)


def find_discov(chi, nsi, d, sigma=4.28):
    chi.g[nsi] = d * 10 ** -8
    if chi.mv > 10 ** -3:
        chi.g[nsi] *= 10 ** (floor(log10(chi.mv)) + 3)
    if chi.det.ty == 'Si' and chi.det.erMin == 1e-07:
        chi.g[nsi] /= 10
    if nsi == 'uee' or nsi == 'dee':
        deno = 2 * sqrt(2) * gf * (2 * chi.det.m * (0.8 * (10 ** -7)) + chi.mv ** 2)
        te = 0.5 * chi.det.z * (0.5 - 2 * ssw + 2 * chi.g['uee'] / deno + chi.g['dee'] / deno) + \
            0.5 * chi.det.n * (-0.5 + chi.g['uee'] / deno + 2 * chi.g['dee'] / deno)
        while te >= 0:
            chi.g[nsi] /= 10
            deno = 2 * sqrt(2) * gf * (2 * chi.det.m * (0.8 * (10 ** -7)) + chi.mv ** 2)
            te = 0.5 * chi.det.z * (0.5 - 2 * ssw + 2 * chi.g['uee'] / deno + chi.g['dee'] / deno) + \
                0.5 * chi.det.n * (-0.5 + chi.g['uee'] / deno + 2 * chi.g['dee'] / deno)
    elif nsi == 'umm' or nsi == 'dmm':
        deno = 2 * sqrt(2) * gf * (2 * chi.det.m * (0.8 * (10 ** -7)) + chi.mv ** 2)
        te = 0.5 * chi.det.z * (0.5 - 2 * ssw + 2 * chi.g['umm'] / deno + chi.g['dmm'] / deno) + \
            0.5 * chi.det.n * (-0.5 + chi.g['umm'] / deno + 2 * chi.g['dmm'] / deno)
        while te >= 0:
            chi.g[nsi] /= 10
            deno = 2 * sqrt(2) * gf * (2 * chi.det.m * (0.8 * (10 ** -7)) + chi.mv ** 2)
            te = 0.5 * chi.det.z * (0.5 - 2 * ssw + 2 * chi.g['umm'] / deno + chi.g['dmm'] / deno) + \
                0.5 * chi.det.n * (-0.5 + chi.g['umm'] / deno + 2 * chi.g['dmm'] / deno)
    ts = tsp = chi.tmu_binned()
    while tsp >= ts > sigma ** 2 or tsp <= ts < sigma ** 2:
        if tsp >= ts > sigma ** 2:
            chi.g[nsi] /= 10
            tsp = ts
            ts = chi.tmu_binned()
        if tsp <= ts < sigma ** 2:
            chi.g[nsi] *= 10
            tsp = ts
            ts = chi.tmu_binned()
    chi.g[nsi] = chi.g[nsi] if tsp >= ts else chi.g[nsi] / 10
    order = floor(log10(d * chi.g[nsi]))
    while abs(chi.tmu_binned() - sigma ** 2) > 0.1:
        while chi.tmu_binned() < sigma ** 2:
            chi.g[nsi] += d * (10 ** order)
        chi.g[nsi] -= d * (10 ** order)
        order -= 1


def find_excl(chi, nsi, d, sigma=3):
    g = couplings()
    effg = ((2 * chi.det.z + chi.det.n) * chi.g['u' + nsi] + (chi.det.z + 2 * chi.det.n) * chi.g['d' + nsi]) / \
           (2 * chi.det.z + chi.det.n)
    if effg != 0.0:
        order = floor(log10(effg))
        g['u' + nsi] = effg + 0.5 * d * (10 ** order)
    else:
        g['u' + nsi] = d * (10 ** -9)
    # print(effg)
    # print(g['u'+nsi])
    ts = tsp = chi.tmuc(g)
    while tsp >= ts > sigma ** 2 or tsp <= ts < sigma ** 2:
        if tsp >= ts > sigma ** 2:
            g['u' + nsi] = effg + (g['u' + nsi] - effg) / 10
            tsp = ts
            ts = chi.tmuc(g)
        if tsp <= ts < sigma ** 2:
            g['u' + nsi] = effg + (g['u' + nsi] - effg) * 10
            tsp = ts
            ts = chi.tmuc(g)
    g['u' + nsi] = g['u' + nsi] if tsp >= ts else effg + (g['u' + nsi] - effg) / 10
    # if effg != 0.0:
    #     si = sign(g['u'+nsi])
    # else:
    #     si = 1
    order = floor(log10(abs(g['u' + nsi]))) - 1
    tsp = chi.tmuc(g)
    while abs(tsp - sigma ** 2) > 0.1:
        while tsp < sigma ** 2:
            g['u' + nsi] += d * (10 ** order)
            tsp = chi.tmuc(g)
        g['u' + nsi] -= d * (10 ** order)
        order -= 1
        tsp = chi.tmuc(g)
        # print(tsp)
    return g['u' + nsi]
