from numpy import *

fplus = 1
w = 1.73e-6
mk = 493
mz = 911876
mw = 80403
mt = 174200
g = 0.651

echarge = 0.303

A = 1 / ((4 * pi) ** 2) * echarge * w / (mk ** 2 * fplus / 2)

uckm = matrix([[0.974351, 0.224998, 0.00156292 - 0.00368202j],
               [-0.224865 - 0.00015068j, 0.973484 - 0.0000347951j, 0.0419997]
               [0.00792849 - 0.00358444j, -0.0412744 - 0.000827723j, 0.99911]])
utd = uckm[2][0]
uts = uckm[2][1]


def x1(mh):
    mhs = mh ** 2
    mts = mt ** 2
    mws = mw ** 2
    mht = mhs - mts
    mtw = mts - mws
    mhw = mhs - mws
    return 2 + mhs / mht - 3 * mws / mtw + \
           3 * mws * mws * (mht - mtw) / (mhw * mtw * mtw) * log(mts / mws) + \
           mhs / mht * (mhs / mht - 6 * mws / mhw) * log(mts / mhs)


def B(mh):
    return 1 / (16 * (pi ** 2)) * (g ** 3) * (mt ** 2) * mz / (8 * (mw ** 3)) * conj(utd) * uts * x1(mh)