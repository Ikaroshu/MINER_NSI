from scipy.optimize import minimize, root
from scipy.interpolate import Rbf
from events import *
from multiprocessing.pool import Pool


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
        self.binned_nsi_e = \
            array([binned_events_e(self.ebin[i], self.ebin[i + 1], self.expo, self.mv, self.det, self.fx, self.g)
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
        scale = 1
        if self.expo >= 1000:
            scale = self.expo / 1000
        res = sum(ni * log(nui) - nui) - ((nf - 1) ** 2) / (2 * (self.fx.flUn ** 2) * scale) - \
            ((nb - 1) ** 2) / (2 * (self.det.bgUn ** 2) * scale)  # n! is constant
        # print(res)
        return res

    def findl0(self, bsm, sm, bg):
        scale = 1
        if self.expo >= 1000:
            scale = self.expo / 1000

        def f(n):
            return -self.lgl(n[0], n[1], 0, bsm, sm, bg)  # minimize

        def f_der(n):
            ni = bsm + sm + bg
            nui = n[0] * (sm + n[1] * bg)
            der = zeros_like(n)
            der[0] = sum((ni / nui - 1) * (sm + n[1] * bg)) - (n[0] - 1) / (self.fx.flUn ** 2 * scale)
            der[1] = sum((ni / nui - 1) * n[0] * bg) - (n[1] - 1) / (self.det.bgUn ** 2 * scale)
            return -der

        def f_hess(n):
            ni = bsm + sm + bg
            nui = n[0] * (sm + n[1] * bg)
            hess = array([[0.0, 0.0], [0.0, 0.0]])
            hess[0][0] = -sum(ni / (nui ** 2) * ((sm + n[1] * bg) ** 2)) - 1 / (self.fx.flUn ** 2 * scale)
            hess[1][1] = -sum(ni / (nui ** 2) * ((n[0] * bg) ** 2)) - 1 / (self.det.bgUn ** 2 * scale)
            hess[0][1] = -sum(ni / (nui ** 2) * n[0] * bg * (sm + n[1] * bg) + (ni / nui - 1) * bg)
            hess[1][0] = -sum(ni / (nui ** 2) * (sm + n[1] * bg) * n[0] * bg + (ni / nui - 1) * bg)
            return -hess

        res = minimize(f, array([1.0, 1.0]), method='SLSQP', jac=f_der, bounds=((1e-10, None), (0, None)))
        # res = root(f_der, [1, 1], jac=f_hess, method='hybr')
        if not res.success:
            # print(self.g, self.ebin.shape[0], self.mv, self.det.ty)
            print(res.message)
            raise Exception("optimization failed!")
        # print('nf,nb', res.x)
        # print(res.x)
        return -res.fun

    def tmu_binned(self):
        if self.th != self.det.erMin:       # 潜在的bug，可能造成死循环
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
        scale = 1
        if self.expo >= 1000:
            scale = self.expo / 1000
        lgl0 = self.findl0(bsm / scale, sm / scale, self.binned_bg / scale)
        lglmu = self.lgl(1, 1, 1, bsm / scale, sm / scale, self.binned_bg / scale)
        return -2 * scale * (lgl0 - lglmu)

    def l0(self, coup):
        if self.th != self.det.erMin:
            self.binned_sm = \
                array([binned_events(self.ebin[i], self.ebin[i + 1], self.expo, self.mv, self.det, self.fx, couplings())
                       for i in range(self.ebin.shape[0] - 1)])
            self.binned_bg = array([binned_background(self.ebin[i], self.ebin[i + 1], self.det, self.expo)
                                    for i in range(self.ebin.shape[0] - 1)])
            self.th = self.det.erMin
        if self.det.ty == 'csi' and self.ebin.shape[0]-1 == 1:
            sm = array([134])
        else:
            sm = array([binned_events(self.ebin[i], self.ebin[i + 1], self.expo, self.mv, self.det, self.fx, coup)
                        for i in range(self.ebin.shape[0] - 1)])
        bsm = self.binned_nsi - sm
        # print('bnsi', self.binned_nsi)
        scale = 1
        if self.expo >= 1000:
            scale = self.expo / 1000
        lgl0 = self.findl0(bsm / scale, sm / scale, self.binned_bg / scale)
        # lglmu = self.lgl(1, 1, 1, bsm / scale, sm / scale, self.binned_bg / scale)
        return scale * lgl0

    def le0(self, coup):
        if self.th != self.det.erMin:
            self.binned_sm = \
                array([binned_events(self.ebin[i], self.ebin[i + 1], self.expo, self.mv, self.det, self.fx, couplings())
                       for i in range(self.ebin.shape[0] - 1)])
            self.binned_bg = array([binned_background(self.ebin[i], self.ebin[i + 1], self.det, self.expo)
                                    for i in range(self.ebin.shape[0] - 1)])
            self.th = self.det.erMin
        sm = array([binned_events_e(self.ebin[i], self.ebin[i + 1], self.expo, self.mv, self.det, self.fx, coup)
                    for i in range(self.ebin.shape[0] - 1)])
        bsm = self.binned_nsi_e - sm
        # print('bnsi', self.binned_nsi)
        scale = 1
        if self.expo >= 1000:
            scale = self.expo / 1000
        lgl0 = self.findl0(bsm / scale, sm / scale, self.binned_bg / scale)
        # lglmu = self.lgl(1, 1, 1, bsm / scale, sm / scale, self.binned_bg / scale)
        return scale * lgl0

    def l0_non(self, coup):
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
        # scale = 1
        # if self.expo >= 1000:
        #     scale = self.expo / 1000
        # lgl0 = self.findl0(bsm / scale, sm / scale, self.binned_bg / scale)
        # lglmu = self.lgl(1, 1, 1, bsm / scale, sm / scale, self.binned_bg / scale)
        return self.lgl(1, 1, 0, bsm, sm, self.binned_bg)

    def le0_non(self, coup):
        if self.th != self.det.erMin:
            self.binned_sm = \
                array([binned_events(self.ebin[i], self.ebin[i + 1], self.expo, self.mv, self.det, self.fx, couplings())
                       for i in range(self.ebin.shape[0] - 1)])
            self.binned_bg = array([binned_background(self.ebin[i], self.ebin[i + 1], self.det, self.expo)
                                    for i in range(self.ebin.shape[0] - 1)])
            self.th = self.det.erMin
        sm = array([binned_events_e(self.ebin[i], self.ebin[i + 1], self.expo, self.mv, self.det, self.fx, coup)
                    for i in range(self.ebin.shape[0] - 1)])
        bsm = self.binned_nsi_e - sm
        # print('bnsi', self.binned_nsi)
        # scale = 1
        # if self.expo >= 1000:
        #     scale = self.expo / 1000
        # lgl0 = self.findl0(bsm / scale, sm / scale, self.binned_bg / scale)
        # lglmu = self.lgl(1, 1, 1, bsm / scale, sm / scale, self.binned_bg / scale)
        return self.lgl(1, 1, 0, bsm, sm, self.binned_bg)

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
    m = average(chi.det.m)
    chi.g[nsi] = d * 10 ** -8
    if chi.mv > 10 ** -3:
        chi.g[nsi] *= 10 ** (floor(log10(chi.mv)) + 3)
    if chi.det.ty == 'Si' and chi.det.erMin == 1e-07:
        chi.g[nsi] /= 10
    if nsi == 'uee' or nsi == 'dee':
        deno = 2 * sqrt(2) * gf * (2 * m * (0.8 * (10 ** -7)) + chi.mv ** 2)
        te = 0.5 * chi.det.z * (0.5 - 2 * ssw + 2 * chi.g['uee'] / deno + chi.g['dee'] / deno) + \
            0.5 * chi.det.n * (-0.5 + chi.g['uee'] / deno + 2 * chi.g['dee'] / deno)
        while te >= 0:
            chi.g[nsi] /= 10
            deno = 2 * sqrt(2) * gf * (2 * m * (0.8 * (10 ** -7)) + chi.mv ** 2)
            te = 0.5 * chi.det.z * (0.5 - 2 * ssw + 2 * chi.g['uee'] / deno + chi.g['dee'] / deno) + \
                0.5 * chi.det.n * (-0.5 + chi.g['uee'] / deno + 2 * chi.g['dee'] / deno)
    elif nsi == 'umm' or nsi == 'dmm':
        deno = 2 * sqrt(2) * gf * (2 * m * (0.8 * (10 ** -7)) + chi.mv ** 2)
        te = 0.5 * chi.det.z * (0.5 - 2 * ssw + 2 * chi.g['umm'] / deno + chi.g['dmm'] / deno) + \
            0.5 * chi.det.n * (-0.5 + chi.g['umm'] / deno + 2 * chi.g['dmm'] / deno)
        while te >= 0:
            chi.g[nsi] /= 10
            deno = 2 * sqrt(2) * gf * (2 * m * (0.8 * (10 ** -7)) + chi.mv ** 2)
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
        g['u' + nsi] = d * (10 ** -10)
    # print(effg)
    # print(g['u'+nsi])
    ts = tsp = chi.tmuc(g)
    while tsp >= ts > sigma ** 2 or tsp <= ts < sigma ** 2:
        if tsp >= ts > sigma ** 2:
            g['u' + nsi] = effg + (g['u' + nsi] - effg) / (10 ** 0.5)
            tsp = ts
            ts = chi.tmuc(g)
        if tsp <= ts < sigma ** 2:
            g['u' + nsi] = effg + (g['u' + nsi] - effg) * (10 ** 0.5)
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


def tmuc(chi, g):
    return chi.tmuc(g)


def find_excl_v2(chi, nsi, sigma=3):
    eu = linspace(-1, 1, 50)
    ed = linspace(-1, 1, 50)
    eeu, eed = meshgrid(eu, ed)
    cr = zeros_like(eeu)
    g = couplings()
    p = Pool()
    for i in range(50):
        for j in range(50):
            g['u' + nsi] = eeu[i][j] * (chi.mv ** 2) * 2 * sqrt(2) * gf
            g['d' + nsi] = eed[i][j] * (chi.mv ** 2) * 2 * sqrt(2) * gf
            r = p.apply_async(tmuc, args=(chi, g))
            cr[i][j] = r.get()
            print(cr[i][j])
    p.close()
    p.join()
    savez('./outputdata/' + chi.det.ty + chi.fx.ty + '.npz', eu, ed, cr)


def find_excl_v3(chi, nsi, sigma=3):
    poi = 200
    if chi.det.background < 1 or chi.expo > 10000:
        poi = 400
    eu = linspace(-1, 1, 50)
    ed = linspace(-2, 2, poi)
    edl = full_like(eu, 1e10)
    edh = full_like(eu, 1e10)
    edl2 = full_like(eu, 1e10)
    edh2 = full_like(eu, 1e10)
    g = couplings()
    p = Pool()
    for i in range(50):
        vv = 1e10
        fl = 0
        for j in range(poi):
            g['u' + nsi] = eu[i] * (chi.mv ** 2) * 2 * sqrt(2) * gf
            g['d' + nsi] = ed[j] * (chi.mv ** 2) * 2 * sqrt(2) * gf
            r = p.apply_async(tmuc, args=(chi, g))
            v = r.get()
            if vv > sigma ** 2 >= v:
                if j == 0:
                    vv = v
                    continue
                else:
                    l = ed[j-1]
                    h = ed[j]
                    g['d' + nsi] = (h + l) / 2 * (chi.mv ** 2) * 2 * sqrt(2) * gf
                    r = p.apply_async(tmuc, args=(chi, g))
                    mid = r.get()
                    while abs(mid - sigma ** 2) > 0.1:
                        if mid > sigma ** 2:
                            l = (h + l) / 2
                            g['d' + nsi] = (h + l) / 2 * (chi.mv ** 2) * 2 * sqrt(2) * gf
                            r = p.apply_async(tmuc, args=(chi, g))
                            mid = r.get()
                        else:
                            h = (h + l) / 2
                            g['d' + nsi] = (h + l) / 2 * (chi.mv ** 2) * 2 * sqrt(2) * gf
                            r = p.apply_async(tmuc, args=(chi, g))
                            mid = r.get()
                    vv = v
                    if fl == 0:
                        edl[i] = (h + l) / 2
                        print(eu[i], edl[i])
                    else:
                        edl2[i] = (h + l) / 2
                        print(eu[i], edl2[i])
            elif vv < sigma ** 2 <= v:
                if j == poi:
                    continue
                else:
                    l = ed[j - 1]
                    h = ed[j]
                    g['d' + nsi] = (h + l) / 2 * (chi.mv ** 2) * 2 * sqrt(2) * gf
                    r = p.apply_async(tmuc, args=(chi, g))
                    mid = r.get()
                    while abs(mid - sigma ** 2) > 0.1:
                        if mid > sigma ** 2:
                            h = (h + l) / 2
                            g['d' + nsi] = (h + l) / 2 * (chi.mv ** 2) * 2 * sqrt(2) * gf
                            r = p.apply_async(tmuc, args=(chi, g))
                            mid = r.get()
                        else:
                            l = (h + l) / 2
                            g['d' + nsi] = (h + l) / 2 * (chi.mv ** 2) * 2 * sqrt(2) * gf
                            r = p.apply_async(tmuc, args=(chi, g))
                            mid = r.get()
                    vv = v
                    if fl == 0:
                        edh[i] = (h + l) / 2
                        print(eu[i], edh[i])
                        fl = 1
                    else:
                        edh2[i] = (h + l) / 2
                        print(eu[i], edh2[i])
            else:
                vv = v
    p.close()
    p.join()
    savez('./outputdata/' + chi.det.ty + chi.fx.ty + nsi + str(int(chi.expo)) + 'expo' + str(chi.det.background) +
          '.npz', eu, edl, edh, edl2, edh2)


# todo: x^2_solar+x^2_detector
# todo: ee vs mumu, 6 dimension
