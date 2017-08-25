from parameters import *
from numpy import sqrt, exp
from scipy.special import spherical_jn


def formfsquared(er, a):
    r = 1.2 * (10 ** -15) * (a ** (1 / 3)) / (MeterByJoule * GeVPerJoule)
    s = 0.5 * (10 ** -15) / (MeterByJoule * GeVPerJoule)
    r0 = sqrt(r ** 2 - 5 * (s ** 2))
    return (3 * spherical_jn(1, er * r0) / (er * r0) * exp((-(er * s) ** 2) / 2)) ** 2


def rates(er, mv, det, fx, g):  # per nucleus
    deno = 2 * sqrt(2) * gf * (2 * det.m * er + mv ** 2)
    qvs = (0.5 * det.z * (0.5 - 2 * ssw + 2 * g['uee'] / deno + g['dee'] / deno) +
           0.5 * det.n * (-0.5 + g['uee'] / deno + 2 * g['dee'] / deno)) ** 2 + \
          (0.5 * det.z * (2 * g['uem'] / deno + g['dem'] / deno) +
           0.5 * det.n * (g['uem'] / deno + 2 * g['dem'] / deno)) ** 2 + \
          (0.5 * det.z * (2 * g['uet'] / deno + g['det'] / deno) +
           0.5 * det.n * (g['uet'] / deno + 2 * g['det'] / deno)) ** 2
    m = average(det.m)
    return dot(2 / pi * (gf ** 2) * (2 * fx.fint(er, m) - det.m * er * fx.fintinvs(er, m)) *
               det.m * qvs * formfsquared(sqrt(2 * det.m * er), det.z + det.n), det.fraction)


def totoal(expo, mv, det, fx, g):
    return quad(rates, det.erMin, det.erMax, args=(mv, det, fx, g))[0] * \
           expo * JoulePerKg * GeVPerJoule * 24 * 60 * 60 / sum(det.m)


def binned_events(era, erb, expo, mv, det, fx, g):
    if fx.ty == 'sns':
        return quad(snsrates, era, erb, args=(mv, det, fx, g,))[0] * \
               expo * JoulePerKg * GeVPerJoule * 24 * 60 * 60 / sum(det.m)
    return quad(rates, era, erb, args=(mv, det, fx, g))[0] * expo * JoulePerKg * GeVPerJoule * 24 * 60 * 60 / sum(det.m)


def binned_background(era, erb, det, expo):
    return det.background * (erb - era) * expo * 1000  # because dru is per kev


def ratesm(er, mv, det, fx, g):
    deno = 2 * sqrt(2) * gf * (2 * det.m * er + mv ** 2)
    qvs = (0.5 * det.z * (0.5 - 2 * ssw + 2 * g['umm'] / deno + g['dmm'] / deno) +
           0.5 * det.n * (-0.5 + g['umm'] / deno + 2 * g['dmm'] / deno)) ** 2 + \
          (0.5 * det.z * (2 * g['uem'] / deno + g['dem'] / deno) +
           0.5 * det.n * (g['uem'] / deno + 2 * g['dem'] / deno)) ** 2 + \
          (0.5 * det.z * (2 * g['umt'] / deno + g['dmt'] / deno) +
           0.5 * det.n * (g['umt'] / deno + 2 * g['dmt'] / deno)) ** 2
    m = average(det.m)
    return dot(2 / pi * (gf ** 2) * (2 * fx.numfint(er, m) - det.m * er * fx.numfinvs(er, m)) *
               det.m * qvs * formfsquared(sqrt(2 * det.m * er), det.z + det.n), det.fraction)


def ratesp(er, mv, det, fx, g):
    deno = 2 * sqrt(2) * gf * (2 * det.m * er + mv ** 2)
    qvs = (0.5 * det.z * (0.5 - 2 * ssw + 2 * g['umm'] / deno + g['dmm'] / deno) +
           0.5 * det.n * (-0.5 + g['umm'] / deno + 2 * g['dmm'] / deno)) ** 2 + \
          (0.5 * det.z * (2 * g['uem'] / deno + g['dem'] / deno) +
           0.5 * det.n * (g['uem'] / deno + 2 * g['dem'] / deno)) ** 2 + \
          (0.5 * det.z * (2 * g['umt'] / deno + g['dmt'] / deno) +
           0.5 * det.n * (g['umt'] / deno + 2 * g['dmt'] / deno)) ** 2
    m = average(det.m)
    return dot(2 / pi * (gf ** 2) * (2 * fx.nupfint(er, m) - det.m * er * fx.nupfinvs(er, m)) *
               det.m * qvs * formfsquared(sqrt(2 * det.m * er), det.z + det.n), det.fraction)


def snsrates(er, mv, det, fx, g):
    return rates(er, mv, det, fx, g) + ratesm(er, mv, det, fx, g) + ratesp(er, mv, det, fx, g)
