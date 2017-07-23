from plots import *
from results import *
from multiprocessing.pool import Pool
from time import time
from numpy import load, logspace, log10
from matplotlib.pyplot import subplots


# f = Flux("reactor")
# d = Detector("Si")
# mv = load("mv.npy")
# gc = load("uee4Si.npy")
# t1 = Chisquare(d, f, mv[10], 1000)
# t1.g["uee"] = gc[10]
# print(t1.g)
# t2 = Chisquare(d, f, mv[11], 1000)
# t2.g["uee"] = gc[11]
# print(t2.g)
# t3 = Chisquare(d, f, mv[12], 1000)
# t3.g["uee"] = gc[12]
# print(t3.g)
# find3sigma(t1, "uee", 1)
# print(t1.g['uee'])
# print(t1.tmus())
# g = couplings()
# print(rates(10**-6, 10, d, f, g), rates(10**-6, 10**-1, d, f, g), rates(10**-6, 10**-2, d, f, g))
# print(f.fint(d.erMax, d.m), f.fintinvs(d.erMax, d.m))
# print(totoal(1000, 10**-6, Detector("Ge"), Flux("reactor"), Couplings()))
# start = time()
# for i in arange(-6, 1.5, 0.5):
#     chi = Chisquare(d, f, 10 ** i, 1000)
#     find3sigma(chi, 'uee', 1)
#     print(chi.g['uee'])
# end = time()
# print("time elapse:", end-start)
# mv = arange(-6, 1.5, 0.5)
# nsi = ['uee', 'dee']
# t = [0, 1, 2, 3, 4, 5]
# def cube(x):
#     t[x] = x**3
# pool = Pool(processes=4)
# for i in range(6):
#     pool.apply_async(cube, args=(i, ))
# pool.close()
# pool.join()
# for x in t:
#     print(x)
# mv = arange(-6, 1.5, 0.5)
# p = Pool()
# lt = ["uee", "dee"]
# for i in lt:
#     p.apply_async(res, args=(i, d, f, mv, ))
# p.close()
# p.join()




# mv = load('mv.npy')
# uee4 = load('uee4Ge.npy')
# uee4interp = interp1d(mv, uee4, kind="cubic")
# mvv = logspace(log10(mv[0]), log10(mv[mv.shape[0]-1]), num=100)
# uee4s = uee4interp(mvv)
# fig, ax = subplots()
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlim([mv[0], mv[14]])
# ax.set_ylim([10**-11, 5*(10**-5)])
# ax.plot(mv, uee4)
# fig.savefig("test.pdf")
# fig2, ax2 = subplots()
# ax2.set_xscale("log")
# ax2.set_ylim([0, 1.5])
# ax2.set_xlim([10**-5, 10])
# ax2.plot(mvv, uee4s/((mvv**2)*2*sqrt(2)*gf))
# fig2.savefig("t2.pdf")
# start = time()
# generate_results("Si", "ee")
# end = time()
# print(end - start)

# t = array([1,2,3])
# tt = array([0.0,0.0,0.0])
# print(tt)
# def f():
#     return 2.5**2
# p = Pool()
# for i in t:
#     re = p.apply_async(f)
#     print(re.get())
#     tt[i-1] = re.get()
# p.close()
# p.join()
# print(tt)

# generate_1d_results('Ge', 'sns', 'mm', 20, "linear")
# generate_1d_results('Si', 'sns', 'mm', 20, "linear")
# generate_1d_results('Ge', 'sns', 'em', 20, "linear")
# generate_1d_results('Si', 'sns', 'em', 20, "linear")
# generate_1d_results('Ge', 'sns', 'mt', 20, "linear")
# generate_1d_results('Si', 'sns', 'mt', 20, "linear")
# generate_2d_results('Ge', 10**-3, 4)
# generate_2d_results('Si', 10**-3, 4)

# g1 = couplings()
# g2 = couplings()
# g2['uee'] = 0.3
# g2['dee'] = 0.3
# generate_excl('Ge', 'sns', '0')
# # generate_excl('Ge', g2, '3')
# generate_excl('Si', 'sns', '0')
# generate_excl('Si', g2, '3')

# plot2d(-4, 2)

# chisge = load('./Ge2-2d'+str(int(-3))+'.npy')
# chissi = load('./Si2-2d'+str(int(-3))+'.npy')
# p1, ax1 = subplots()
# ax1.contour(chisge, levels=array([4.28**2]), extent=(-1, 1, -1, 1))
# ax1.contour(chissi, levels=array([4.28**2]), extent=(-1, 1, -1, 1))
# ax1.set_xlabel(r"$\epsilon^u_{ee}$")
# ax1.set_ylabel(r"$\epsilon^d_{ee}$")
# ax1.set_title(r"$\chi^2$ for Ge detector")
# p1.tight_layout()
# p1.savefig('2d.pdf')

# gueege = load('./uee4Ge.npy')
# gueedge = load('./ueed4Ge.npy')
# mv = load('./mv.npy')
# u0ge = gueege[where(mv == 10**-3)][0]
# ud0ge = gueedge[where(mv == 10**-3)][0]
# uee = arange(-1, 1.05, 0.05)
# dge = Detector("Ge")
# deege = (2*dge.z+dge.n)/(dge.z+2*dge.n)*(u0ge / (((10**-3) ** 2) * 2 * sqrt(2) * gf) - uee)
# deedge = (2*dge.z+dge.n)/(dge.z+2*dge.n)*(ud0ge / (((10**-3) ** 2) * 2 * sqrt(2) * gf) - uee)
# gueesi = load('./uee4Si.npy')
# gueedsi = load('./ueed4Si.npy')
# u0si = gueesi[where(mv == 10**-3)][0]
# ud0si = gueedsi[where(mv == 10**-3)][0]
# uee = arange(-1, 1.05, 0.05)
# dsi = Detector("Si")
# deesi = (2*dsi.z+dsi.n)/(dsi.z+2*dsi.n)*(u0si / (((10**-3) ** 2) * 2 * sqrt(2) * gf) - uee)
# deedsi = (2*dsi.z+dsi.n)/(dsi.z+2*dsi.n)*(ud0si / (((10**-3) ** 2) * 2 * sqrt(2) * gf) - uee)
# p1, ax1 = subplots()
# ax1.set_ylim([-1, 1])
# ax1.set_xlim([-1, 1])
# ax1.fill_between(uee, deege, 1, alpha=0.5, color='green', label="Ge")
# ax1.fill_between(uee, deedge, -1, alpha=0.5, color='green')
# ax1.fill_between(uee, deesi, 1, alpha=0.5, color='orange', label="Si")
# ax1.fill_between(uee, deedsi, -1, alpha=0.5, color='orange')
# ax1.set_xlabel(r"$\epsilon^u_{ee}$")
# ax1.set_ylabel(r"$\epsilon^d_{ee}$")
# ax1.set_title(r"$m_v = 1 MeV$, threshold = $400 KeV$")
# ax1.legend()
# p1.savefig('t2d4-3.pdf')

# plot2dl(-3, 1)

# dash line for different
# contour for different mv
# when will close the bands
# representative one exclude region
# start = time()
# d = Detector("Ge")
# f = Flux("reactor")
# mv = logspace(-6, 1, 10)
# expo = 1000
# g = couplings()
# # g['uee'] = 10**-9
# res('uee', d, f, mv, 20, 'linear')
# end = time()
# print(end-start)


# def test():
#     d = Detector("Ge")
#     f = Flux("reactor")
#     mv = 10**-3
#     expo = 1000
#     g = couplings()
#     # g['uee'] = 10**-9
#     chi = Chisquare(d, f, g, mv, expo, 20, "linear")
#     chi.g['uee'] = 10**-9
#     r1 = chi.tmu_binned()
#     print(r1)
#     # chi.g['uee'] = 10**-8
#     # r2 = chi.tmu_binned()
#     # print(r2)
#     # chi.g['uee'] = 10**-7
#     # r3 = chi.tmu_binned()
#     # print(r3)
# test()
# start = time()
# d = Detector("Ge")
# f = Flux("sns")
# mv = 0.0316227766017
# expo = 1000
# g = couplings()
# g['umm'] = 10**-9
# chi = Chisquare(d, f, couplings(), mv, expo, 20, "linear")
# # chi.g['umm'] = 3e-5
# # print(chi.det.erMin)
# r = find_excl(chi, 'mm', 1)
# print(r)
# end = time()
# print(end-start)
#
# start = time()
# d = Detector("Ge")
# f = Flux("reactor")
# mv = 10**1
# expo = 1000
# g = couplings()
# # g['uee'] = 10**-9
# chi = Chisquare(d, f, g, mv, expo, 20, "linear")
# find_discov(chi, 'uee', -1)
# print(chi.g['uee'])
# end = time()
# print(end-start)
# chi = Chisquare(d, f, g, mv, expo, 4, "linear")
# print(chi.tmu_binned())
# chi = Chisquare(d, f, g, mv, expo, 6, "linear")
# print(chi.tmu_binned())
# chi = Chisquare(d, f, g, mv, expo, 8, "linear")
# print(chi.tmu_binned())
# chi = Chisquare(d, f, g, mv, expo, 20, "linear")
# print(chi.tmu_binned())
# chi.det.bgUn = chi.fx.flUn = 0.2
# print(chi.tmu_binned())
# chi.det.bgUn = chi.fx.flUn = 0.2
# print(chi.tmu_binned())
# print(chi.tmu_binned(), chi.tmus())
# chi = Chisquare(d, f, g, mv, expo, 12, "linear")
# print(chi.tmu_binned())
# chi = Chisquare(d, f, g, mv, expo, 14, "linear")
# print(chi.tmu_binned())
# chi = Chisquare(d, f, g, mv, expo, 16, "linear")
# print(chi.tmu_binned())
# chi.det.bgUn = 0.0000001
# chi.fx.flUn = 0.0000001
# print(chi.tmu_binned())
# chi2 = Chisquare(d, f, mv, expo, 100, "linear")
# chi2.g["uee"] = 10**-8
# print(chi2.tmu_binned(), chi2.tmus())
# chi3 = Chisquare(d, f, mv, expo, 100, "log")
# chi3.g["uee"] = 10**-8
# print(chi3.tmu_binned(), chi3.tmus())

# tmu_sigma()
# p_tmu_sigma()


# mv = logspace(-6, 1, 20)
# fx = Flux("reactor")
# p = Pool()
# detge = Detector('Ge')
# detsi = Detector('Si')
# detsi.erMin /= 4
# p.apply_async(res, args=('uem', detge, fx, mv, 20, 'linear', ))
# p.apply_async(res, args=('uem', detsi, fx, mv, 20, 'linear', ))
# p.close()
# p.join()

generate_excl('Ge', 'sns', '0')
generate_excl('Si', 'sns', '0')
# generate_excl('Ge', '3')
# generate_excl('Si', '3')
# plot_excl('Si', '0')
# plot_excl('Si', '3')

# plot1d()
# epsi = [0.3, 0.3]
# g = couplings()
# g['uee'] = epsi[0]*(((10**-3)**2)*2*sqrt(2)*gf)
# g['dee'] = epsi[1]*(((10**-3)**2)*2*sqrt(2)*gf)
# chi = Chisquare(Detector('Ge'), Flux('reactor'), g, 10**-3, 1000, 20)
# r = find_excl(chi, -1)
# print(r)

# mv = 10**-3
# det = Detector("Ge")
# fx = Flux('sns')
# g = couplings()
# r = snsrates(10**-6, mv, det, fx, g)
# print(r)

#
# d = Detector("Ge")
# f = Flux("sns")
# mv = 0.0316227766017
# expo = 1000
# g = couplings()
# g['umm'] = 10**-9
# chi = Chisquare(d, f, couplings(), mv, expo, 20, "linear")
# r = find_excl(chi, 'mm', -1)
# print(r)