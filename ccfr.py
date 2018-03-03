from constants import *
from scipy.special import spherical_jn
from scipy.integrate import nquad


def v4(p1, p2):
    ret = 0
    ret += p1[0] * p2[0]
    for i in range(3):
        ret += -p1[i+1] * p2[i+1]
    return ret


def v3(p1, p2):
    ret = 0
    for i in range(3):
        ret += p1[i+1] * p2[i+1]
    return ret


cv = 0.5 + 2 * ssw
ca = 0.5
enu = 20


def formfsquared(er):
    a = 56
    r = 1.2 * (10 ** -15) * (a ** (1 / 3)) / (MeterByJoule * GeVPerJoule)
    s = 0.5 * (10 ** -15) / (MeterByJoule * GeVPerJoule)
    r0 = sqrt(r ** 2 - 5 * (s ** 2))
    return (3 * spherical_jn(1, er * r0) / (er * r0) * exp((-(er * s) ** 2) / 2)) ** 2


def dsig(th, th1, ph1, qs, sp, s, gp, mzp):
    q = array([0.5 * sqrt(s), 0, 0, 0.5 * sqrt(s)])
    k = array([0.5 * sqrt(s), 0, 0, -0.5 * sqrt(s)])
    p12 = array([(s + sp) / (2 * sqrt(s)), (s - sp) / (2 * sqrt(s)) * sin(th), 0, (s - sp) / (2 * sqrt(s)) * cos(th)])
    kp = array([(s - sp) / (2 * sqrt(s)), -(s - sp) / (2 * sqrt(s)) * sin(th), 0, -(s - sp) / (2 * sqrt(s)) * cos(th)])

    p120 = p12[0]
    p12c = (s - sp) / (2 * sqrt(s)) * sin(th) * sin(th1) * cos(ph1) + (s - sp) / (2 * sqrt(s)) * cos(th) * cos(th1)
    p12p = (s - sp) / (2 * sqrt(s))

    sr = (p120 ** 2) * ((p12p ** 4) - 2 * (p12p ** 2) * (p120 ** 2) + (p120 ** 4) - 4 * (mmu ** 2) * ((p120 ** 2) - (p12c ** 2)))
    pts1 = 1 / (2 * (p120 ** 2 - p12c ** 2)) * (-(p12p ** 2) * p12c + (p120 ** 2) * p12c + sqrt(sr)) if sr >= 0 and -(p12p ** 2) * p12c + (p120 ** 2) * p12c + sqrt(sr) > 0 else 0
    pts2 = 1 / (2 * (p120 ** 2 - p12c ** 2)) * (-(p12p ** 2) * p12c + (p120 ** 2) * p12c - sqrt(sr)) if sr >= 0 and -(p12p ** 2) * p12c + (p120 ** 2) * p12c - sqrt(sr) > 0 else 0
    p1s1 = array([sqrt(pts1 ** 2 + mmu ** 2), pts1 * sin(th1) * cos(ph1), pts1 * sin(th1) * sin(ph1), pts1 * cos(th1)])
    p1s2 = array([sqrt(pts2 ** 2 + mmu ** 2), pts2 * sin(th1) * cos(ph1), pts2 * sin(th1) * sin(ph1), pts2 * cos(th1)])
    p2s1 = p12 - p1s1
    p2s2 = p12 - p1s2

    cvz = (246 ** 2) * ((gp / mzp) ** 2)
    kkp = v4(k - kp, k - kp)
    va = 0.5 * ((cv ** 2) + (ca ** 2) - 2 * cv * cvz * ((mzp ** 2) / (kkp - (mzp ** 2))) + (cvz * ((mzp ** 2) / (kkp - (mzp ** 2)))) ** 2)

    ret = 0

    if pts1 > 0:
        a = v4(p1s1 - q, p1s1 - q) - mmu ** 2
        b = v4(p2s1 - q, p2s1 - q) - mmu ** 2
        p1q = v4(p1s1, q)
        kq = v4(k, q)
        p2kp = v4(p2s1, kp)
        p2q = v4(p2s1, q)
        kpq = v4(kp, q)
        p1k = v4(p1s1, k)
        p1p2 = v4(p1s1, p2s1)
        p2k = v4(p2s1, k)
        p1kp = v4(p1s1, kp)
        ret += va * (echarge ** 2) * (gf ** 2) / ((pi ** 5) * kq) * \
               (p1q * kq * p2kp / (a ** 2) + p2q * kpq * p1k / (b ** 2) +
                1 / (a * b) * ((2 * p1p2 - p1q - p2q) * p1k * p2kp - p1p2 * p1k * kpq - p1p2 * p2kp * kq + p1q * p2k * p2kp + p2q * p1k * p1kp)) * 1 / 4 * \
               ((pts1 ** 2) / (pts1 * p120 - sqrt(pts1 ** 2 + mmu ** 2) * p12c)) * 1 / 4 * kp[0] / sqrt(sp)

    if pts2 > 0:
        a = v4(p1s2 - q, p1s2 - q) - mmu ** 2
        b = v4(p2s2 - q, p2s2 - q) - mmu ** 2
        p1q = v4(p1s2, q)
        kq = v4(k, q)
        p2kp = v4(p2s2, kp)
        p2q = v4(p2s2, q)
        kpq = v4(kp, q)
        p1k = v4(p1s2, k)
        p1p2 = v4(p1s2, p2s2)
        p2k = v4(p2s2, k)
        p1kp = v4(p1s2, kp)
        ret += va * (echarge ** 2) * (gf ** 2) / ((pi ** 5) * kq) * \
               (p1q * kq * p2kp / (a ** 2) + p2q * kpq * p1k / (b ** 2) +
                1 / (a * b) * ((2 * p1p2 - p1q - p2q) * p1k * p2kp - p1p2 * p1k * kpq - p1p2 * p2kp * kq + p1q * p2k * p2kp + p2q * p1k * p1kp)) * 1 / 4 * \
               ((pts2 ** 2) / (pts2 * p120 - sqrt(pts2 ** 2 + mmu ** 2) * p12c)) * 1 / 4 * kp[0] / sqrt(sp)

    return ret * 2 * pi * (26 ** 2) * (echarge ** 2) / (4 * (pi ** 2)) / s * formfsquared(sqrt(qs)) / qs


def sigma(gp, mzp):
    def lim_th(th1, ph1, qs, sp, s, gp, mzp):
        return [0, pi]

    def lim_th1(ph1, qs, sp, s, gp, mzp):
        return [0, pi]

    def lim_ph1(qs, sp, s, gp, mzp):
        return [0, 2 * pi]

    def lim_qs(sp, s, gp, mzp):
        return [(s / (2 * enu)) ** 2, 0.05]

    def lim_sp(s, gp, mzp):
        return [4 * (mmu ** 2), s]

    def lim_s(gp, mzp):
        return [4 * (mmu ** 2), 2 * enu * sqrt(0.05)]

    return nquad(dsig, [lim_th, lim_th1, lim_ph1, lim_qs, lim_sp, lim_s], args=(gp, mzp))[0]
