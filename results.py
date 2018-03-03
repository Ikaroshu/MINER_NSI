from chisquares import *
from numpy import *
from multiprocessing.pool import Pool


def res(nsi, d, f, mv, nbins, div="linear"):
    g = zeros_like(mv)
    expo = 1000
    for i in range(mv.shape[0]):
        chi = Chisquare(d, f, couplings(), mv[i], expo, nbins, div)
        find_discov(chi, nsi, 1)
        g[i] = chi.g[nsi]
        print(i, g[i])
    save('./outputdata/' + f.ty + nsi + str(int(d.erMin * (10 ** 7))) + d.ty + str(int(nbins)) + div + '.npy', g)
    if nsi == 'uee' or nsi == 'dee' or nsi == 'umm' or 'dmm':
        for i in range(mv.shape[0]):
            chi = Chisquare(d, f, couplings(), mv[i], expo, nbins, div)
            find_discov(chi, nsi, -1)
            g[i] = chi.g[nsi]
            print(i, g[i])
        save('./outputdata/' + f.ty + nsi + 'd' + str(int(d.erMin * (10 ** 7))) + d.ty + str(int(nbins)) + div + '.npy',
             g)


def generate_1d_results(ty, f, nsi, nbins, div="linear"):  # dee可以被倒推出来！
    mv = logspace(-6, 1, 20)
    save('./mv', mv)
    fx = Flux(f)
    p = Pool()
    det = Detector(ty)
    p.apply_async(res, args=('u' + nsi, det, fx, mv, nbins, div,))
    # p.apply_async(res, args=('d'+nsi, det, fx, mv, nbins, div, ))
    det2 = Detector(ty)
    det2.erMin /= 2
    p.apply_async(res, args=('u' + nsi, det2, fx, mv, nbins, div,))
    # p.apply_async(res, args=('d'+nsi, det2, fx, mv, nbins, div, ))
    det1 = Detector(ty)
    det1.erMin /= 4
    p.apply_async(res, args=('u' + nsi, det1, fx, mv, nbins, div,))
    # p.apply_async(res, args=('d'+nsi, det1, fx, mv, nbins, div, ))
    p.close()
    p.join()


def ts(chi):  # in order to use multiprocessing
    return chi.tmu_binned()


def generate_2d_results(ty, f, mv, th, nbins, div="linear"):
    fx = Flux(f)
    det = Detector(ty)
    det.erMin *= th / 4
    expo = 1000
    chi = Chisquare(det, fx, couplings(), mv, expo, nbins, div)
    epsi = arange(-1, 1.05, 0.05)
    x, y = meshgrid(epsi, epsi)
    re = full(x.shape, 0.0)
    p = Pool()
    if f == 'sns':
        nsi = 'mm'
    else:
        nsi = 'ee'
    for i in range(epsi.shape[0]):
        for j in range(epsi.shape[0]):
            chi.g['u' + nsi] = x[i][j] * (mv ** 2) * 2 * sqrt(2) * gf
            chi.g['d' + nsi] = y[i][j] * (mv ** 2) * 2 * sqrt(2) * gf
            r = p.apply_async(ts, args=(chi,))
            re[i][j] = r.get()
    p.close()
    p.join()
    save('./' + f + ty + str(int(th)) + '-2d' + str(int(log10(mv))) + '-' + str(int(nbins)) + div, re)


def tmu_bins():
    nbins = linspace(1, 20, 20)
    tmulinear = zeros_like(nbins)
    tmulog = zeros_like(nbins)
    p = Pool()
    d = Detector("Ge")
    f = Flux("reactor")
    g = couplings()
    g['uee'] = 10 ** -9
    for i in range(nbins.shape[0]):
        chilinear = Chisquare(d, f, g, 10 ** -3, 1000, nbins[i])
        chilog = Chisquare(d, f, g, 10 ** -3, 1000, nbins[i], div='log')
        r1 = p.apply_async(ts, args=(chilinear,))
        r2 = p.apply_async(ts, args=(chilog,))
        tmulinear[i] = r1.get()
        tmulog[i] = r2.get()
    p.close()
    p.join()
    save('./outputdata/tmu_bins_-9_-3_1000_linear.npy', tmulinear)
    save('./outputdata/tmu_bins_-9_-3_1000_log.npy', tmulog)
    save('./outputdata/nbins.npy', nbins)


def tmu_sigma():
    sigma = 2 * logspace(-6, -1, 6)
    tmu1b = zeros_like(sigma)
    tmu20b = zeros_like(sigma)
    p = Pool()
    d = Detector("Ge")
    f = Flux("reactor")
    g = couplings()
    g['uee'] = 10 ** -9
    for i in range(sigma.shape[0]):
        chi1b = Chisquare(d, f, g, 10 ** -3, 1000, 1)
        chi20b = Chisquare(d, f, g, 10 ** -3, 1000, 20)
        chi1b.det.bgUn = chi1b.fx.flUn = sigma[i]
        chi20b.det.bgUn = chi20b.fx.flUn = sigma[i]
        r1 = p.apply_async(ts, args=(chi1b,))
        r2 = p.apply_async(ts, args=(chi20b,))
        tmu1b[i] = r1.get()
        tmu20b[i] = r2.get()
    p.close()
    p.join()
    save('./outputdata/tmu_sigma_-9_-3_1000_linear_1b.npy', tmu1b)
    save('./outputdata/tmu_sigma_-9_-3_1000_linear_20b.npy', tmu20b)
    save('outputdata/sigma.npy', sigma)


def generate_excl(ty, f, th, tag):
    mv = logspace(-3, -1, 5)
    p = Pool()
    uee = zeros_like(mv)
    ueed = zeros_like(mv)
    d = Detector(ty)
    d.erMin /= (4 / th)
    fx = Flux(f)
    if tag == '0':
        epsi = [0, 0]
    elif tag == '3':
        epsi = [0.3, 0.3]
    if f == 'sns':
        nsi = 'ee'
        expo = 10000
    else:
        nsi = 'ee'
        expo = 1000
    for i in range(mv.shape[0]):
        g = couplings()
        g['u' + nsi] = epsi[0] * ((mv[i] ** 2) * 2 * sqrt(2) * gf)
        g['d' + nsi] = epsi[1] * ((mv[i] ** 2) * 2 * sqrt(2) * gf)
        chi = Chisquare(d, fx, g, mv[i], expo, 20)
        r1 = p.apply_async(find_excl, args=(chi, nsi, 1, 3,))
        r2 = p.apply_async(find_excl, args=(chi, nsi, -1, 3,))
        uee[i] = r1.get()
        ueed[i] = r2.get()
        print(uee[i], ueed[i])
    p.close()
    p.join()
    save('./outputdata/' + f + 'u' + nsi + '_excl_-' + str(int(th)) + '_' + str(int(expo)) + '_linear' +
         tag + ty + '.npy', uee)
    save('./outputdata/' + f + 'u' + nsi + '_excl_-' + str(int(th)) + 'd_' + str(int(expo)) + '_linear' +
         tag + ty + '.npy', ueed)


def generate_dark():
    ty = ['Ge', 'Si']
    th = ['1', '2', '4']
    for i in ty:
        for j in th:
            f = load('./outputdata/uem' + j + i + '20linear.npy')
            f = -f
            save('./outputdata/uemd' + j + i + '20linear.npy', f)


def findepsi(ty, expo, epsity, nb, sig, sb, div='linear'):
    if ty == 'reactor':
        dar = Detector('ge')
        dnai = Detector('si')
        f = Flux('reactor')
    elif ty == 'sns':
        dar = Detector('ar')
        dnai = Detector('nai')
        dar.erMin = dnai.erMin
        f = Flux('sns')

    flg = 0
    att = 0

    def x2(g, mv):
        gg = couplings()
        if epsity == 'zz':
            gg['uee'] = 0.5 * ((0.303 / (0.471 * 0.882)) ** 2) * (0.5 - 4 / 3 * ssw) * g ** 2
            gg['dee'] = 0.5 * ((0.303 / (0.471 * 0.882)) ** 2) * (-0.5 + 2 / 3 * ssw) * g ** 2
            gg['umm'] = 0.5 * ((0.303 / (0.471 * 0.882)) ** 2) * (0.5 - 4 / 3 * ssw) * g ** 2
            gg['dmm'] = 0.5 * ((0.303 / (0.471 * 0.882)) ** 2) * (-0.5 + 2 / 3 * ssw) * g ** 2
        elif epsity == 'bz':
            gg['uee'] = -0.5 * (0.35 ** 2) * (1 / 6 + 2 / 3) * g ** 2
            gg['dee'] = -0.5 * (0.35 ** 2) * (1 / 6 - 1 / 3) * g ** 2
            gg['umm'] = -0.5 * (0.35 ** 2) * (1 / 6 + 2 / 3) * g ** 2
            gg['dmm'] = -0.5 * (0.35 ** 2) * (1 / 6 - 1 / 3) * g ** 2
        # print(gg)
        if div == 'linear':
            ebin_ar = linspace(dar.erMin, dar.erMax, nb + 1)
            ebin_nai = linspace(dnai.erMin, dnai.erMax, nb + 1)
        elif div == 'log':
            ebin_ar = logspace(log10(dar.erMin), log10(dar.erMax), nb + 1)
            ebin_nai = logspace(log10(dnai.erMin), log10(dnai.erMax), nb + 1)
        nar = array([binned_events(ebin_ar[i], ebin_ar[i + 1], expo, mv, dar, f, gg)
                     for i in range(ebin_ar.shape[0] - 1)])
        nobsar = array([binned_events(ebin_ar[i], ebin_ar[i + 1], expo, mv, dar, f, couplings())
                        for i in range(ebin_ar.shape[0] - 1)])
        nnai = array([binned_events(ebin_nai[i], ebin_nai[i + 1], expo, mv, dnai, f, gg)
                     for i in range(ebin_nai.shape[0] - 1)])
        nobsnai = array([binned_events(ebin_nai[i], ebin_nai[i + 1], expo, mv, dnai, f, couplings())
                        for i in range(ebin_nai.shape[0] - 1)])
        bgar = array([binned_background(ebin_ar[i], ebin_ar[i + 1], dar, expo)
                      for i in range(ebin_ar.shape[0] - 1)])
        bgnai = array([binned_background(ebin_nai[i], ebin_nai[i + 1], dnai, expo)
                      for i in range(ebin_nai.shape[0] - 1)])
        if sb == 'y':
            bgnai = 1/3*nobsnai
            bgar = 1 / 3 * nobsar
        a = sum(2 * nar * (nobsar - nar) / (nobsar + bgar) + 2 * nnai * (nobsnai - nnai) / (nobsnai + bgnai)) / \
            (2 / sig ** 2 + sum(2 * nar ** 2 / (nobsar + bgar) + 2 * nnai ** 2 / (nobsnai + bgnai)))
        if flg == 1:
            print('a', a)
            nonlocal att
            att = a
            print('nobar-nar', sum(nobsar-nar))
        return sum((nobsar - nar * (1 + a)) ** 2 / (nobsar + bgar)) + \
               sum((nobsnai - nnai * (1 + a)) ** 2 / (nobsnai + bgnai)) + (a / sig) ** 2 - 4

    ml = logspace(-3, 1)
    gl = zeros_like(ml)
    al = zeros_like(ml)
    for i in range(50):
        print(ml[i])
        gi = 1e-3
        gn = gi
        m = ml[i]
        while x2(gi, m) > 0 and x2(gn, m) > 0:
            gi = gn
            gn /= 10
        while x2(gi, m) < 0 and x2(gn, m) < 0:
            gn = gi
            gi *= 10
        order = floor(log10(gn))
        while abs(x2(gn, m)) > 0.01:
            while x2(gn, m) < 0:
                gn += 10 ** order
            gn -= 10 ** order
            order -= 1
        gl[i] = gn
        print(gn)
        flg = 1
        x2(gn, m)
        flg = 0
        al[i] = att
        print(al[i])
    # save('./out/al' + ty + epsity + str(expo) + str(sig) + sb + str(nb) + div + '.npy', al)
    save('./out/' + ty + epsity + str(expo) + str(sig) + sb + str(nb) + div + '.npy', gl)


def findepsisingle(ty, expo, epsity, nb, div='log'):
    dar = Detector(ty)
    # if ty == 'ar':
    #     dar.erMin = 2 * (10 ** -6)
    # elif ty == 'nai':
    #     dar.erMin = 30 * (10 ** -6)
    dar.erMin = 2e-6
    f = Flux('sns')

    def x2(g, mv):
        gg = couplings()
        if epsity == 'zz':
            gg['uee'] = 0.5 * ((0.303 / (0.471 * 0.882)) ** 2) * (0.5 - 4 / 3 * ssw) * g ** 2
            gg['dee'] = 0.5 * ((0.303 / (0.471 * 0.882)) ** 2) * (-0.5 + 2 / 3 * ssw) * g ** 2
            gg['umm'] = 0.5 * ((0.303 / (0.471 * 0.882)) ** 2) * (0.5 - 4 / 3 * ssw) * g ** 2
            gg['dmm'] = 0.5 * ((0.303 / (0.471 * 0.882)) ** 2) * (-0.5 + 2 / 3 * ssw) * g ** 2
        elif epsity == 'bz':
            gg['uee'] = -0.5 * (0.35 ** 2) * (1 / 6 + 2 / 3) * g ** 2
            gg['dee'] = -0.5 * (0.35 ** 2) * (1 / 6 - 1 / 3) * g ** 2
            gg['umm'] = -0.5 * (0.35 ** 2) * (1 / 6 + 2 / 3) * g ** 2
            gg['dmm'] = -0.5 * (0.35 ** 2) * (1 / 6 - 1 / 3) * g ** 2
        elif epsity == 'lz':
            gg['umm'] = (2 / 3) * 8 * 0.303 * (g ** 2) * log(mtau / mmu) / (3 * 16 * pi ** 2)
            gg['dmm'] = (-1 / 3) * 8 * 0.303 * (g ** 2) * log(mtau / mmu) / (3 * 16 * pi ** 2)
        # print(gg)
        if div == 'linear':
            ebin_ar = linspace(dar.erMin, dar.erMax, nb + 1)
            # ebin_nai = linspace(dnai.erMin, dnai.erMax, nb + 1)
        elif div == 'log':
            ebin_ar = logspace(log10(dar.erMin), log10(dar.erMax), nb + 1)
            # ebin_nai = logspace(log10(dnai.erMin), log10(dnai.erMax), nb + 1)
        nar = array([binned_events(ebin_ar[i], ebin_ar[i + 1], expo, mv, dar, f, gg)
                     for i in range(ebin_ar.shape[0] - 1)])
        nobsar = array([binned_events(ebin_ar[i], ebin_ar[i + 1], expo, mv, dar, f, couplings())
                        for i in range(ebin_ar.shape[0] - 1)])
        if ty == 'csi' and expo == 4466:
            nobsar = 142
        # nnai = totoal(expo, mv, dnai, f, gg)
        # nobsnai = totoal(expo, mv, dnai, f, couplings())
        bgar = array([binned_background(ebin_ar[i], ebin_ar[i + 1], dar, expo)
                      for i in range(ebin_ar.shape[0] - 1)])
        # bgnai = binned_background(dnai.erMin, dnai.erMax, dnai, expo)
        a = sum(2 * nar * (nobsar - nar) / (nobsar + bgar)) / \
            (2 / 0.1 ** 2 + sum(2 * nar ** 2 / (nobsar + bgar)))
        return sum((nobsar - nar * (1 + a)) ** 2 / (nobsar + bgar)) + (a / 0.1) ** 2 - 4

    ml = logspace(-3, 1)
    gl = zeros_like(ml)
    for i in range(50):
        gi = 1e-8
        gn = gi
        m = ml[i]
        while x2(gi, m) > 0 and x2(gn, m) > 0:
            gi = gn
            gn /= sqrt(2)
        while x2(gi, m) < 0 and x2(gn, m) < 0:
            gn = gi
            gi *= sqrt(2)
        order = floor(log10(gn))-1
        while abs(x2(gn, m)) > 0.01:
            while x2(gn, m) < 0:
                gn += 10 ** order
            gn -= 10 ** order
            order -= 1
        gl[i] = gn
        print(gn)
    save('./out/' + ty + epsity + str(expo) + str(nb) + '.npy', gl)


def find1epsi(ty, expo):
    if ty == 'reactor':
        dar = Detector('ge')
        dnai = Detector('si')
        f = Flux('reactor')
    elif ty == 'sns':
        dar = Detector('ar')
        dnai = Detector('nai')
        dar.erMin = dnai.erMin
        f = Flux('sns')

    def x2(g, mv):
        gg = couplings()
        # gg['uee'] = 0.1 * 0.35 * (1 / 6 + 2 / 3) * g
        # gg['dee'] = 0.1 * 0.35 * (1 / 6 - 1 / 3) * g
        gg['umm'] = (2/3) * 8 * 0.303 * (g ** 2) * log(mtau/mmu) / (3 * 16 * pi ** 2)
        gg['dmm'] = (-1/3) * 8 * 0.303 * (g ** 2) * log(mtau/mmu) / (3 * 16 * pi ** 2)
        # print(gg)
        # def mut(expo, mv, det, fx, g):
        #     def rm(er, mv, det, fx, g):
        #         return ratesm(er, mv, det, fx, g) + ratesp(er, mv, det, fx, g)
        #     return quad(rm, det.erMin, det.erMax, args=(mv, det, fx, g))[0] * \
        #         expo * JoulePerKg * GeVPerJoule * 24 * 60 * 60 / dot(det.m, det.fraction)
        # print(gg)
        nar = totoal(expo, mv, dar, f, gg)
        # print('nar', nar)
        nobsar = totoal(expo, mv, dar, f, couplings())
        # print('nobs', nobsar)
        nnai = totoal(expo, mv, dnai, f, gg)
        nobsnai = totoal(expo, mv, dnai, f, couplings())
        bgar = binned_background(dar.erMin, dar.erMax, dar, expo)
        bgnai = binned_background(dnai.erMin, dnai.erMax, dnai, expo)
        a = (2 * nar * (nobsar - nar) / (nobsar + bgar) + 2 * nnai * (nobsnai - nnai) / (nobsnai + bgnai)) / \
            (2 / 0.1 ** 2 + 2 * nar ** 2 / (nobsar + bgar) + 2 * nnai ** 2 / (nobsnai + bgnai))
        return (nobsar - nar * (1 + a)) ** 2 / (nobsar + bgar) + \
               (nobsnai - nnai * (1 + a)) ** 2 / (nobsnai + bgnai) + (a / 0.1) ** 2 - 4

    ml = logspace(-3, 1)
    gl = zeros_like(ml)
    for i in range(50):
        gi = 1e-5
        gn = gi
        m = ml[i]
        while x2(gi, m) > 0 and x2(gn, m) > 0:
            gi = gn
            gn /= sqrt(10)
        while x2(gi, m) < 0 and x2(gn, m) < 0:
            gn = gi
            gi *= sqrt(10)
            # print(gi)
        order = floor(log10(gn))
        while abs(x2(gn, m)) > 0.01:
            while x2(gn, m) < 0:
                gn += 10 ** order
            gn -= 10 ** order
            order -= 1
        gl[i] = gn
        print(gn)
    save('./out/1epsi' + ty + str(expo) + '.npy', gl)


def av(p, ty):
    if ty == 'zz':
        if p == 'u':
            return 1/4 - 2/3*ssw
        if p == 'd' or p == 's':
            return -1/4 + 1/3*ssw
        if p == 'e' or p == 'mu' or p == 'tau':
            return -1/4 + ssw
        if p == 'nu':
            return 1/4
    if ty == 'bz':
        if p == 'u':
            return 5/12
        if p == 'd' or p == 's':
            return -1/12
        if p == 'e' or p == 'mu' or p == 'tau':
            return -3/4
        if p == 'nu':
            return -1/4


def aa(p, ty):
    if ty == 'zz':
        if p == 'u':
            return -1/4
        if p == 'd' or p == 's':
            return 1/4
        if p == 'e' or p == 'mu' or p == 'tau':
            return 1/4
        if p == 'nu':
            return -1/4
    if ty == 'bz':
        if p == 'u':
            return 1/4
        if p == 'd' or p == 's':
            return -1/4
        if p == 'e' or p == 'mu' or p == 'tau':
            return -1/4
        if p == 'nu':
            return 1/4


def ms(p):
    if p == 'u':
        return 2.2
    if p == 'd':
        return 5
    if p == 's':
        return 95
    if p == 'e':
        return 0.511
    if p == 'mu':
        return 105.66
    if p == 'tau':
        return 1777
    if p == 'nu':
        return 0


seterr(all='raise')


def wdth(p, ty, mv):
    try:
        rslt = (av(p, ty)**2 * (2*ms(p)**2 + mv**2) + aa(p, ty)**2 * (-4*ms(p)**2 + mv**2))*sqrt(mv**2 - 4*ms(p)**2)
        print(p, rslt)
        return rslt
    except FloatingPointError as fpe:
        print(mv, p)


def brch(p, ty, mv):
    tot = wdth('e', ty, mv) + 3 * wdth('nu', ty, mv)
    if mv > 139:
        tot += wdth('u', ty, mv) + wdth('d', ty, mv)
    if mv > 2*ms('mu'):
        tot += wdth('mu', ty, mv)
    if mv > 2*ms('tau'):
        tot += wdth('tau', ty, mv)
    if mv > 497:
        tot += wdth('s', ty, mv)
    return wdth(p, ty, mv) / tot


mz = 91187.6

import pymultinest
from scipy.stats import norm
import os
import shutil
def findepsi_mltnst(ty, expo, epsity, nb, sig, div='log'):
    if ty == 'reactor':
        dar = Detector('ge')
        dnai = Detector('si')
        f = Flux('reactor')
    elif ty == 'sns':
        dar = Detector('ar')
        dnai = Detector('nai')
        f = Flux('sns')
    dar.erMin = dnai.erMin
    def prior(cube, ndim, nparams):
        cube[0] = (cube[0] - 0.5) * 2

    ml = logspace(-3, 1)
    for i in range(50):
        chiar = Chisquare(dar, f, couplings(), ml[i], expo, nb, div)
        chinai = Chisquare(dnai, f, couplings(), ml[i], expo, nb, div)
        def loglike(cube, ndim, nparams):
            gg = couplings()
            if epsity == 'zz':
                gg['uee'] = 0.5 * ((0.303 / (0.471 * 0.882)) ** 2) * (0.5 - 4 / 3 * ssw) * cube[0] ** 2
                gg['dee'] = 0.5 * ((0.303 / (0.471 * 0.882)) ** 2) * (-0.5 + 2 / 3 * ssw) * cube[0] ** 2
                gg['umm'] = 0.5 * ((0.303 / (0.471 * 0.882)) ** 2) * (0.5 - 4 / 3 * ssw) * cube[0] ** 2
                gg['dmm'] = 0.5 * ((0.303 / (0.471 * 0.882)) ** 2) * (-0.5 + 2 / 3 * ssw) * cube[0] ** 2
            elif epsity == 'bz':
                gg['uee'] = -0.5 * (0.35 ** 2) * (1 / 6 + 2 / 3) * cube[0] ** 2
                gg['dee'] = -0.5 * (0.35 ** 2) * (1 / 6 - 1 / 3) * cube[0] ** 2
                gg['umm'] = -0.5 * (0.35 ** 2) * (1 / 6 + 2 / 3) * cube[0] ** 2
                gg['dmm'] = -0.5 * (0.35 ** 2) * (1 / 6 - 1 / 3) * cube[0] ** 2
            cube[1] = norm.ppf(cube[1], loc=1, scale=sig)
            return chiar.l0_nob_obg(gg, cube[1]) + chinai.l0_nob_obg(gg, cube[1])
        n_params = 2
        cwd = os.getcwd()
        fname = os.path.join(cwd, ty+str(expo)+str(nb)+div)
        if os.path.isdir(fname):
            shutil.rmtree(fname)
        os.makedirs(fname)
        pymultinest.run(loglike, prior, n_params, outputfiles_basename=fname + '/' + str(ml[i]),
                        resume=False, verbose=False, n_live_points=1500, evidence_tolerance=0.1, sampling_efficiency=0.3)


def cornerPoint(fl, cl=0.9545, nbins=80):
    param = fl[:, 1]
    prob = fl[:, 0]
    minx = amin(param)
    maxx = amax(param)
    xbin = (maxx - minx) / nbins
    nl = linspace(minx + xbin / 2, maxx - xbin / 2, nbins)
    pl = zeros_like(nl)
    for i in range(param.shape[0] - 1):
        psit = int((param[i] - minx) / xbin)
        if param[i] == maxx:
            psit = nbins - 1
        pl[psit] += prob[i]
    ind = pl.argsort()[::-1]
    inx = 0
    tot = 0
    while tot <= cl and inx < nbins:
        tot += pl[ind[inx]]
        inx += 1
    inx -= 1
    if nl[ind[inx]] >= 0:
        return nl[ind[inx]]
    else:
        return -nl[ind[inx]]

