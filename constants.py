from numpy import *

GevPerEv = 10 ** -9
GeVPerMeV = 10 ** -3
GeVPerJoule = GevPerEv / (1.6 * (10 ** -19))
MeVPerJoule = GeVPerJoule / GeVPerMeV
JoulePerKg = (3 * (10 ** 8)) ** 2
MeterByJoule = 6.626 * (10 ** -34) * 3 * (10 ** 8) / (2 * pi)
ssw = 0.2312
gf = 1.1664 * (10 ** -5)
