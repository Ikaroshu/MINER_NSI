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
    save('./outputdata/'+nsi+str(int(d.erMin*(10**7)))+d.ty+str(int(nbins))+div+'.npy', g)
    if nsi == 'uee' or nsi == 'dee':
        for i in range(mv.shape[0]):
            chi = Chisquare(d, f, couplings(), mv[i], expo, nbins, div)
            find_discov(chi, nsi, -1)
            g[i] = chi.g[nsi]
            print(i, g[i])
        save('./outputdata/'+nsi+'d'+str(int(d.erMin*(10**7)))+d.ty+str(int(nbins))+div+'.npy', g)


def generate_1d_results(ty, nsi, nbins, div="linear"):      #dee可以被倒推出来！
    mv = logspace(-6, 1, 20)
    save('./mv', mv)
    fx = Flux("reactor")
    p = Pool()
    det = Detector(ty)
    p.apply_async(res, args=('u'+nsi, det, fx, mv, nbins, div, ))
    # p.apply_async(res, args=('d'+nsi, det, fx, mv, nbins, div, ))
    det2 = Detector(ty)
    det2.erMin /= 2
    p.apply_async(res, args=('u'+nsi, det2, fx, mv, nbins, div, ))
    # p.apply_async(res, args=('d'+nsi, det2, fx, mv, nbins, div, ))
    det1 = Detector(ty)
    det1.erMin /= 4
    p.apply_async(res, args=('u'+nsi, det1, fx, mv, nbins, div, ))
    # p.apply_async(res, args=('d'+nsi, det1, fx, mv, nbins, div, ))
    p.close()
    p.join()


def ts(chi):    # in order to use multiprocessing
    return chi.tmu_binned()


def generate_2d_results(ty, mv, th, nbins, div="linear"):
    fx = Flux("reactor")
    det = Detector(ty)
    det.erMin *= th/4
    expo = 1000
    chi = Chisquare(det, fx, couplings(), mv, expo, nbins, div)
    epsi = arange(-1, 1.05, 0.05)
    x, y = meshgrid(epsi, epsi)
    re = full(x.shape, 0.0)
    p = Pool()
    for i in range(epsi.shape[0]):
        for j in range(epsi.shape[0]):
            chi.g["uee"] = x[i][j]*(mv**2)*2*sqrt(2)*gf
            chi.g["dee"] = y[i][j]*(mv**2)*2*sqrt(2)*gf
            r = p.apply_async(ts, args=(chi,))
            re[i][j] = r.get()
    p.close()
    p.join()
    save('./'+ty+str(int(th))+'-2d'+str(int(log10(mv)))+'-'+str(int(nbins))+div, re)


def tmu_bins():
    nbins = linspace(1, 20, 20)
    tmulinear = zeros_like(nbins)
    tmulog = zeros_like(nbins)
    p = Pool()
    d = Detector("Ge")
    f = Flux("reactor")
    g = couplings()
    g['uee'] = 10**-9
    for i in range(nbins.shape[0]):
        chilinear = Chisquare(d, f, g, 10**-3, 1000, nbins[i])
        chilog = Chisquare(d, f, g, 10**-3, 1000, nbins[i], div = 'log')
        r1 = p.apply_async(ts, args = (chilinear, ))
        r2 = p.apply_async(ts, args = (chilog, ))
        tmulinear[i] = r1.get()
        tmulog[i] = r2.get()
    p.close()
    p.join()
    save('./outputdata/tmu_bins_-9_-3_1000_linear.npy', tmulinear)
    save('./outputdata/tmu_bins_-9_-3_1000_log.npy', tmulog)
    save('./outputdata/nbins.npy', nbins)


def tmu_sigma():
    sigma = 2*logspace(-6, -1, 6)
    tmu1b = zeros_like(sigma)
    tmu20b = zeros_like(sigma)
    p = Pool()
    d = Detector("Ge")
    f = Flux("reactor")
    g = couplings()
    g['uee'] = 10**-9
    for i in range(sigma.shape[0]):
        chi1b = Chisquare(d, f, g, 10**-3, 1000, 1)
        chi20b = Chisquare(d, f, g, 10**-3, 1000, 20)
        chi1b.det.bgUn = chi1b.fx.flUn = sigma[i]
        chi20b.det.bgUn = chi20b.fx.flUn = sigma[i]
        r1 = p.apply_async(ts, args = (chi1b, ))
        r2 = p.apply_async(ts, args = (chi20b, ))
        tmu1b[i] = r1.get()
        tmu20b[i] = r2.get()
    p.close()
    p.join()
    save('./outputdata/tmu_sigma_-9_-3_1000_linear_1b.npy', tmu1b)
    save('./outputdata/tmu_sigma_-9_-3_1000_linear_20b.npy', tmu20b)
    save('outputdata/sigma.npy', sigma)


def generate_excl(ty, tag):
    mv = logspace(-3, -1, 5)
    p = Pool()
    uee = zeros_like(mv)
    ueed = zeros_like(mv)
    d = Detector(ty)
    fx = Flux("reactor")
    if tag == '0':
        epsi = [0, 0]
    elif tag == '3':
        epsi = [0.3, 0.3]
    for i in range(mv.shape[0]):
        g = couplings()
        g['uee'] = epsi[0]*((mv[i]**2)*2*sqrt(2)*gf)
        g['dee'] = epsi[1]*((mv[i]**2)*2*sqrt(2)*gf)
        chi = Chisquare(d, fx, g, mv[i], 1000, 20)
        r1 = p.apply_async(find_excl, args = (chi, 1, 3, ))
        r2 = p.apply_async(find_excl, args = (chi, -1, 3, ))
        uee[i] = r1.get()
        ueed[i] = r2.get()
        print(uee[i], ueed[i])
    p.close()
    p.join()
    save('./outputdata/uee_excl_-4_1000_linear'+tag+ty+'.npy', uee)
    save('./outputdata/uee_excl_-4d_1000_linear'+tag+ty+'.npy', ueed)


def generate_dark():
    ty = ['Ge', 'Si']
    th = ['1', '2', '4']
    for i in ty:
        for j in th:
            f = load('./outputdata/uem'+j+i+'20linear.npy')
            f = -f
            save('./outputdata/uemd'+j+i+'20linear.npy', f)
