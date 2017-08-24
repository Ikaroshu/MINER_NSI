from matplotlib.pyplot import subplots, get_cmap
from chisquares import *


def plot1d(ty):
    mv = logspace(-6, 1, 20)
    mvv = logspace(log10(mv[0]), log10(mv[mv.shape[0] - 1]), num=100)
    fig, axes = subplots(nrows=2, ncols=3, sharey='row', sharex='col')
    if ty.lower() == 'sns':
        ec = 'sns'
    elif ty.lower() == 'reactor':
        ec = ''
    else:
        raise Exception('no such data yet!')

    def p1u(ax, nsi):
        uge = load('./outputdata/' + ec + 'u' + nsi + '4Ge20linear.npy')
        usi = load('./outputdata/' + ec + 'u' + nsi + '4Si20linear.npy')
        uged = load('./outputdata/' + ec + 'u' + nsi + 'd4Ge20linear.npy')
        usid = load('./outputdata/' + ec + 'u' + nsi + 'd4Si20linear.npy')
        gge_intp = interp1d(mv, uge, kind='cubic')
        gsi_intp = interp1d(mv, usi, kind='cubic')
        gged_intp = interp1d(mv, uged, kind='cubic')
        gsid_intp = interp1d(mv, usid, kind='cubic')
        ggge = gge_intp(mvv) / ((mvv ** 2) * 2 * sqrt(2) * gf)
        ggsi = gsi_intp(mvv) / ((mvv ** 2) * 2 * sqrt(2) * gf)
        ggged = gged_intp(mvv) / ((mvv ** 2) * 2 * sqrt(2) * gf)
        ggsid = gsid_intp(mvv) / ((mvv ** 2) * 2 * sqrt(2) * gf)
        ax.plot(mvv, ggge, color="r", alpha=0.8)
        ax.plot(mvv, ggsi, color="b", alpha=0.8)
        ax.plot(mvv, ggged, color="r", alpha=0.8)
        ax.plot(mvv, ggsid, color="b", alpha=0.8)

    def p1d(ax, nsi):       # down quark
        uge = load('./outputdata/' + ec + 'u' + nsi + '4Ge20linear.npy')
        usi = load('./outputdata/' + ec + 'u' + nsi + '4Si20linear.npy')
        uged = load('./outputdata/' + ec + 'u' + nsi + 'd4Ge20linear.npy')
        usid = load('./outputdata/' + ec + 'u' + nsi + 'd4Si20linear.npy')
        gge_intp = interp1d(mv, uge, kind='cubic')
        gsi_intp = interp1d(mv, usi, kind='cubic')
        gged_intp = interp1d(mv, uged, kind='cubic')
        gsid_intp = interp1d(mv, usid, kind='cubic')
        ggge = gge_intp(mvv) / ((mvv ** 2) * 2 * sqrt(2) * gf)
        ggsi = gsi_intp(mvv) / ((mvv ** 2) * 2 * sqrt(2) * gf)
        ggged = gged_intp(mvv) / ((mvv ** 2) * 2 * sqrt(2) * gf)
        ggsid = gsid_intp(mvv) / ((mvv ** 2) * 2 * sqrt(2) * gf)
        detge = Detector("Ge")
        ggge *= (detge.z * 2 + detge.n) / (detge.z + 2 * detge.n)
        ggged *= (detge.z * 2 + detge.n) / (detge.z + 2 * detge.n)
        detsi = Detector("Si")
        ggsi *= (detsi.z * 2 + detsi.n) / (detsi.z + 2 * detsi.n)
        ggsid *= (detsi.z * 2 + detsi.n) / (detsi.z + 2 * detsi.n)
        ax.plot(mvv, ggge, color="r", alpha=0.8)
        ax.plot(mvv, ggsi, color="b", alpha=0.8)
        ax.plot(mvv, ggged, color="r", alpha=0.8)
        ax.plot(mvv, ggsid, color="b", alpha=0.8)

    axes[0][0].set_xscale('log')
    axes[0][1].set_xscale('log')
    axes[0][2].set_xscale('log')
    axes[0][0].set_ylim([-10, 10])
    axes[1][0].set_ylim([-10, 10])
    # axes[0][0].set_yscale('symlog', lintreshy=2)
    # axes[1][0].set_yscale('symlog', lintreshy=2)
    if ty.lower() == 'sns':
        p1u(axes[0][0], 'mm')
        p1d(axes[1][0], 'mm')
        p1u(axes[0][1], 'em')
        p1u(axes[0][2], 'em')
        p1d(axes[1][1], 'em')
        p1d(axes[1][2], 'em')
    elif ty.lower() == 'reactor':
        p1u(axes[0][0], 'ee')
        p1d(axes[1][0], 'ee')
        p1u(axes[0][1], 'em')
        p1u(axes[0][2], 'em')
        p1d(axes[1][1], 'em')
        p1d(axes[1][2], 'em')

    def f1(ax, l, h):
        c1 = full(mvv.shape, l)
        c2 = full(mvv.shape, h)
        ax.fill_between(mvv, c1, c2, alpha=0.5, color="gray")

    if ty.lower() == 'reactor':
        f1(axes[0][0], -1.19, -0.81)
        f1(axes[0][0], 0.00, 0.51)
        f1(axes[0][1], -0.09, 0.10)
        f1(axes[0][2], -0.15, 0.14)
        f1(axes[1][0], -1.17, -1.03)
        f1(axes[1][0], 0.02, 0.51)
        f1(axes[1][1], -0.09, 0.08)
        f1(axes[1][2], -0.13, 0.14)
        axes[0][0].text(1e-5, 0.7, r"$\epsilon^{u}_{ee}$")
        axes[0][1].text(1e-5, 0.7, r"$\epsilon^{u}_{e\mu}$")
        axes[0][2].text(1e-5, 0.7, r"$\epsilon^{u}_{e\tau}$")
        axes[1][0].text(1e-5, 0.7, r"$\epsilon^{d}_{ee}$")
        axes[1][1].text(1e-5, 0.7, r"$\epsilon^{d}_{e\mu}$")
        axes[1][2].text(1e-5, 0.7, r"$\epsilon^{d}_{e\tau}$")
    elif ty.lower() == 'sns':
        axes[0][0].text(1e-5, 0.7, r"$\epsilon^{u}_{\mu\mu}$")
        axes[0][1].text(1e-5, 0.7, r"$\epsilon^{u}_{e\mu}$")
        axes[0][2].text(1e-5, 0.7, r"$\epsilon^{u}_{\mu\tau}$")
        axes[1][0].text(1e-5, 0.7, r"$\epsilon^{d}_{\mu\mu}$")
        axes[1][1].text(1e-5, 0.7, r"$\epsilon^{d}_{e\mu}$")
        axes[1][2].text(1e-5, 0.7, r"$\epsilon^{d}_{\mu\tau}$")
    axes[0][1].plot([], [], color="r", label="Ge")
    axes[0][1].plot([], [], color="b", label="Si")
    axes[0][1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    axes[1][1].set_xlabel(r"$m_v/GeV$")
    # fig.tight_layout()
    fig.savefig("./plots/" + ty + "1d.pdf")


def plot2d(mv, th):
    chisge = load('./Ge' + str(int(th)) + '-2d' + str(int(mv)) + '.npy')
    chissi = load('./Si' + str(int(th)) + '-2d' + str(int(mv)) + '.npy')
    p1, ax1 = subplots()
    im1 = ax1.imshow(chisge, interpolation='bilinear', extent=(-1, 1, -1, 1), origin="lower")
    p1.colorbar(im1, ax=ax1)
    ax1.set_xlabel(r"$\epsilon^u_{ee}$")
    ax1.set_ylabel(r"$\epsilon^d_{ee}$")
    ax1.set_title(r"$\chi^2$ for Ge detector threshold = " + str(th * 100) + " kev")
    p1.tight_layout()
    p1.savefig('2d' + str(int(th)) + 'Ge' + str(int(mv)) + '.pdf')
    p2, ax2 = subplots()
    im2 = ax2.imshow(chissi, interpolation='bilinear', extent=(-1, 1, -1, 1), origin="lower")
    p2.colorbar(im2, ax=ax2)
    ax2.set_xlabel(r"$\epsilon^u_{ee}$")
    ax2.set_ylabel(r"$\epsilon^d_{ee}$")
    ax2.set_title(r"$\chi^2$ for Si detector threshold = " + str(th * 100) + " kev")
    p2.tight_layout()
    p2.savefig('2d' + str(int(th)) + 'Si' + str(int(mv)) + '.pdf')


def plot2dl(med, th):
    gueege = load('./uee' + str(th) + 'Ge.npy')
    gueedge = load('./ueed' + str(th) + 'Ge.npy')
    mv = load('./mv.npy')
    u0ge = gueege[where(mv == 10 ** med)][0]
    ud0ge = gueedge[where(mv == 10 ** med)][0]
    uee = arange(-1, 1.05, 0.05)
    dge = Detector("Ge")
    deege = (2 * dge.z + dge.n) / (dge.z + 2 * dge.n) * (u0ge / (((10 ** med) ** 2) * 2 * sqrt(2) * gf) - uee)
    deedge = (2 * dge.z + dge.n) / (dge.z + 2 * dge.n) * (ud0ge / (((10 ** med) ** 2) * 2 * sqrt(2) * gf) - uee)
    gueesi = load('./uee' + str(th) + 'Si.npy')
    gueedsi = load('./ueed' + str(th) + 'Si.npy')
    u0si = gueesi[where(mv == 10 ** med)][0]
    ud0si = gueedsi[where(mv == 10 ** med)][0]
    uee = arange(-1, 1.05, 0.05)
    dsi = Detector("Si")
    deesi = (2 * dsi.z + dsi.n) / (dsi.z + 2 * dsi.n) * (u0si / (((10 ** med) ** 2) * 2 * sqrt(2) * gf) - uee)
    deedsi = (2 * dsi.z + dsi.n) / (dsi.z + 2 * dsi.n) * (ud0si / (((10 ** med) ** 2) * 2 * sqrt(2) * gf) - uee)
    p1, ax1 = subplots()
    ax1.set_ylim([-1, 1])
    ax1.set_xlim([-1, 1])
    ax1.fill_between(uee, deege, 1, alpha=0.5, color='green', label="Ge")
    ax1.fill_between(uee, deedge, -1, alpha=0.5, color='green')
    ax1.fill_between(uee, deesi, 1, alpha=0.5, color='orange', label="Si")
    ax1.fill_between(uee, deedsi, -1, alpha=0.5, color='orange')
    ax1.set_xlabel(r"$\epsilon^u_{ee}$")
    ax1.set_ylabel(r"$\epsilon^d_{ee}$")
    ax1.set_title(r"$m_v$ = " + str(1000 * 10 ** med) + " MeV, threshold = " + str(th) + "00 KeV")
    ax1.legend()
    p1.savefig('t2d' + str(th) + str(med) + '.pdf')


def p_tmu_nbins():
    nb = load('./outputdata/nbins.npy')
    tmulinear = load('./outputdata/tmu_bins_-7_-3_1000_linear.npy')
    tmulog = load('./outputdata/tmu_bins_-7_-3_1000_log.npy')
    fig, ax = subplots()
    ax.plot(nb, tmulinear, label="linear")
    ax.plot(nb, tmulog, label="log")
    ax.set_xlim([1, 20])
    ax.set_xlabel(r"number of bins")
    ax.set_ylabel(r"$t_0$")
    ax.set_title(r"$m_v = 10^{-3}$ GeV, $g^{u}_{ee} = 10^{-9}$")
    ax.legend(loc=4)
    fig.savefig('./plots/tmu_bins_-9_-3_1000.pdf')


# def p_tmu_sigma():
#     sigma = load('outputdata/sigma.npy')
#     tmu1b = load('./outputdata/tmu_sigma_-9_-3_1000_linear_1b.npy')
#     tmu20b = load('./outputdata/tmu_sigma_-9_-3_1000_linear_20b.npy')
#     d = Detector("Ge")
#     f = Flux("reactor")
#     g = couplings()
#     g['uee'] = 10 ** -9
#     chi = Chisquare(d, f, g, 10 ** -3, 1000, 10)
#     tmu_ori = full(sigma.shape, chi.tmus())
#     fig, ax = subplots()
#     ax.plot(sigma, tmu1b, label='1 bin')
#     ax.plot(sigma, tmu20b, label='20 bins')
#     ax.plot(sigma, tmu_ori, label='no sigma')
#     ax.set_xscale('log')
#     ax.set_xlabel(r"$\sigma_F = \sigma_B = \sigma$")
#     ax.set_ylabel(r"$t_0$")
#     ax.set_title(r"$m_v = 10^{-3}$ GeV, $g^{u}_{ee} = 10^{-9}$")
#     ax.legend()
#     fig.savefig('./plots/tmu_sigma_-9_-3.pdf')


def plot_excl(ty, f, tag):
    mv = logspace(-3, -1, 5)
    if f == 'sns':
        uee = load('./outputdata/snsumm_excl_-4_1000_linear' + tag + ty + '.npy')
        ueed = load('./outputdata/snsumm_excl_-4d_1000_linear' + tag + ty + '.npy')
    elif f == 'reactor':
        uee = load('./outputdata/uee_excl_-4_1000_linear' + tag + ty + '.npy')
        ueed = load('./outputdata/uee_excl_-4d_1000_linear' + tag + ty + '.npy')
    else:
        raise Exception("no such flux yet!")
    det = Detector(ty)
    cmap = get_cmap('jet_r')
    fig, ax = subplots()
    for i in range(mv.shape[0]):
        u = linspace(-1, 1, 100)
        d = (2 * det.z + det.n) / (2 * det.n + det.z) * (uee[i] / ((mv[i] ** 2) * 2 * sqrt(2) * gf) - u)
        dd = (2 * det.z + det.n) / (2 * det.n + det.z) * (ueed[i] / ((mv[i] ** 2) * 2 * sqrt(2) * gf) - u)
        ax.plot(u, d, label='{:.1e} MeV'.format(mv[i] * 1000), color=cmap(float(i) / 5))
        ax.plot(u, dd, color=cmap(float(i) / 5))
    if f == 'sns':
        ax.set_xlabel(r"$\epsilon^u_{\mu\mu}$")
        ax.set_ylabel(r"$\epsilon^d_{\mu\mu}$")
    elif f == 'reactor':
        ax.set_xlabel(r"$\epsilon^u_{ee}$")
        ax.set_ylabel(r"$\epsilon^d_{ee}$")
    ax.set_title(r"$3\sigma$ region for different mediator mass")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.legend()
    fig.savefig('./plots/excl_' + f + ty + tag + '.pdf')


def plot_combined(f, tag, th, m):
    mv = logspace(-3, -1, 5)
    if f == 'sns':
        ueege = load('./outputdata/snsuee_excl_-' + str(int(th)) + '_10000_linear' + tag + 'Ge' + '.npy')[m]
        ueesi = load('./outputdata/snsuee_excl_-' + str(int(th)) + '_10000_linear' + tag + 'ar' + '.npy')[m]
        ueeged = load('./outputdata/snsuee_excl_-' + str(int(th)) + 'd_10000_linear' + tag + 'Ge' + '.npy')[m]
        ueesid = load('./outputdata/snsuee_excl_-' + str(int(th)) + 'd_10000_linear' + tag + 'ar' + '.npy')[m]
    elif f == 'reactor':
        ueege = load('./outputdata/reactoruee_excl_-' + str(int(th)) + '_1000_linear' + tag + 'Ge' + '.npy')[m]
        ueesi = load('./outputdata/reactoruee_excl_-' + str(int(th)) + '_1000_linear' + tag + 'Si' + '.npy')[m]
        ueeged = load('./outputdata/reactoruee_excl_-' + str(int(th)) + 'd_1000_linear' + tag + 'Ge' + '.npy')[m]
        ueesid = load('./outputdata/reactoruee_excl_-' + str(int(th)) + 'd_1000_linear' + tag + 'Si' + '.npy')[m]
    else:
        raise Exception("no such flux yet!")
    detge = Detector('Ge')
    detsi = Detector('Si')
    # if f == 'reactor':
    #     u = linspace(-0.1, 0.1, 100)
    # elif f == 'sns':
    #     u = linspace(-1, 1, 100)
    # else:
    #     raise Exception("no such flux yet!")
    kge = (2 * detge.z + detge.n) / (2 * detge.n + detge.z)
    ksi = (2 * detsi.z + detsi.n) / (2 * detsi.n + detsi.z)
    low = (ksi * ueesid / ((mv[m] ** 2) * 2 * sqrt(2) * gf) - kge * ueege / ((mv[m] ** 2) * 2 * sqrt(2) * gf)) / \
          (ksi - kge)
    high = (ksi * ueesi / ((mv[m] ** 2) * 2 * sqrt(2) * gf) - kge * ueeged / ((mv[m] ** 2) * 2 * sqrt(2) * gf)) / \
           (ksi - kge)
    u = linspace(low, high, 100)
    dge = (2 * detge.z + detge.n) / (2 * detge.n + detge.z) * (ueege / ((mv[m] ** 2) * 2 * sqrt(2) * gf) - u)
    ddge = (2 * detge.z + detge.n) / (2 * detge.n + detge.z) * (ueeged / ((mv[m] ** 2) * 2 * sqrt(2) * gf) - u)
    dsi = (2 * detsi.z + detsi.n) / (2 * detsi.n + detsi.z) * (ueesi / ((mv[m] ** 2) * 2 * sqrt(2) * gf) - u)
    ddsi = (2 * detsi.z + detsi.n) / (2 * detsi.n + detsi.z) * (ueesid / ((mv[m] ** 2) * 2 * sqrt(2) * gf) - u)
    s = zeros_like(u)
    sd = zeros_like(u)
    for i in range(u.shape[0]):
        s[i] = dge[i] if dge[i] < dsi[i] else dsi[i]
        sd[i] = ddge[i] if ddge[i] > ddsi[i] else ddsi[i]
    fig, ax = subplots()
    ax.plot(u, dge, color='red', label='ge')
    ax.plot(u, ddge, color='red')
    ax.plot(u, dsi, color='blue', label='si' if f == 'reactor' else 'ar')
    ax.plot(u, ddsi, color='blue')
    ax.fill_between(u, s, sd, color='gray')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.legend()
    if f == 'sns':
        ax.set_xlabel(r"$\epsilon^u_{ee}$")
        ax.set_ylabel(r"$\epsilon^d_{ee}$")
    elif f == 'reactor':
        ax.set_xlabel(r"$\epsilon^u_{ee}$")
        ax.set_ylabel(r"$\epsilon^d_{ee}$")
    ax.set_title(r"$m_v$ = {0:.1e} MeV, $m_{{th}}$ = {1} MeV".format(mv[m] * 1000, th * 100))
    fig.savefig('./plots/combine_' + str(int(th)) + f + 'uee' + tag + str(m) + '.pdf')


def plot_combine_diff(tag, th, m):
    mv = logspace(-3, -1, 5)

    def gen(f):
        if f == 'sns':
            ueege = load('./outputdata/snsumm_excl_-' + str(int(th)) + '_10000_linear' + tag + 'Ge' + '.npy')[m]
            ueesi = load('./outputdata/snsumm_excl_-' + str(int(th)) + '_10000_linear' + tag + 'Si' + '.npy')[m]
            ueeged = load('./outputdata/snsumm_excl_-' + str(int(th)) + 'd_10000_linear' + tag + 'Ge' + '.npy')[m]
            ueesid = load('./outputdata/snsumm_excl_-' + str(int(th)) + 'd_10000_linear' + tag + 'Si' + '.npy')[m]
        elif f == 'reactor':
            ueege = load('./outputdata/reactoruee_excl_-' + str(int(th)) + '_1000_linear' + tag + 'Ge' + '.npy')[m]
            ueesi = load('./outputdata/reactoruee_excl_-' + str(int(th)) + '_1000_linear' + tag + 'Si' + '.npy')[m]
            ueeged = load('./outputdata/reactoruee_excl_-' + str(int(th)) + 'd_1000_linear' + tag + 'Ge' + '.npy')[m]
            ueesid = load('./outputdata/reactoruee_excl_-' + str(int(th)) + 'd_1000_linear' + tag + 'Si' + '.npy')[m]
        else:
            raise Exception("no such flux yet!")
        detge = Detector('Ge')
        detsi = Detector('Si')
        kge = (2 * detge.z + detge.n) / (2 * detge.n + detge.z)
        ksi = (2 * detsi.z + detsi.n) / (2 * detsi.n + detsi.z)
        low = (ksi * ueesid / ((mv[m] ** 2) * 2 * sqrt(2) * gf) - kge * ueege / ((mv[m] ** 2) * 2 * sqrt(2) * gf)) / \
            (ksi - kge)
        high = (ksi * ueesi / ((mv[m] ** 2) * 2 * sqrt(2) * gf) - kge * ueeged / ((mv[m] ** 2) * 2 * sqrt(2) * gf)) / \
            (ksi - kge)
        u = linspace(low, high, 100)
        dge = (2 * detge.z + detge.n) / (2 * detge.n + detge.z) * (ueege / ((mv[m] ** 2) * 2 * sqrt(2) * gf) - u)
        ddge = (2 * detge.z + detge.n) / (2 * detge.n + detge.z) * (ueeged / ((mv[m] ** 2) * 2 * sqrt(2) * gf) - u)
        dsi = (2 * detsi.z + detsi.n) / (2 * detsi.n + detsi.z) * (ueesi / ((mv[m] ** 2) * 2 * sqrt(2) * gf) - u)
        ddsi = (2 * detsi.z + detsi.n) / (2 * detsi.n + detsi.z) * (ueesid / ((mv[m] ** 2) * 2 * sqrt(2) * gf) - u)
        s = zeros_like(u)
        sd = zeros_like(u)
        for i in range(u.shape[0]):
            s[i] = dge[i] if dge[i] < dsi[i] else dsi[i]
            sd[i] = ddge[i] if ddge[i] > ddsi[i] else ddsi[i]
        return s, sd, u

    uu = linspace(-1, 1, 100)
    sns, snsd, sx = gen('sns')
    reac, reacd, rx = gen('reactor')
    sns_interp = interp1d(sx, sns)
    snsd_interp = interp1d(sx, snsd)
    reac_interp = interp1d(rx, reac)
    reacd_interp = interp1d(rx, reacd)
    ux = linspace(-1.5, 1, 100)
    uy = zeros_like(ux)
    uyd = zeros_like(ux)
    for eemm in ux:
        pos = where(ux == eemm)
        uy[pos] = -1000
        uyd[pos] = 1000
        for ee in rx:
            mm = ee - eemm
            t = reac_interp(ee)[()] - snsd_interp(mm)[()]
            td = reacd_interp(ee)[()] - sns_interp(mm)[()]
            uy[pos] = t if t > uy[pos] else uy[pos]
            uyd[pos] = td if td < uyd[pos] else uyd[pos]
    fig, ax = subplots()
    ax.fill_between(ux, uy, uyd)
    ax.set_xlabel(r"$\epsilon^{u}_{ee}-\epsilon^{u}_{\mu\mu}$")
    ax.set_ylabel(r"$\epsilon^{d}_{ee}-\epsilon^{d}_{\mu\mu}$")
    fig.savefig("./test.pdf")
