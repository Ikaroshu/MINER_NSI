from numpy import *

GevPerEv = 10 ** -9
GeVPerMeV = 10 ** -3
GeVPerJoule = GevPerEv / (1.6 * (10 ** -19))
MeVPerJoule = GeVPerJoule / GeVPerMeV
JoulePerKg = (3 * (10 ** 8)) ** 2
MeterByJoule = 6.626 * (10 ** -34) * 3 * (10 ** 8) / (2 * pi)
ssw = 0.2312
gf = 1.1664 * (10 ** -5)
# in grams
massofh = 1.67372e-24
massof4he = 6.646479e-24
massof3he = 5.008234e-24
massof12c = 1.993e-23
massof14n = 2.3252651e-23
massof16o = 2.656018e-23
# in GeV
massofe = 5.11e-4
rho = 1.0086
knu = 0.9978
lul = -0.0031
ldl = -0.0025
ldr = 7.5e-5
lur = ldr / 2

mtau = 1.77699
mmu = 105.658369e-3
mb = 4.18

echarge = 0.303
