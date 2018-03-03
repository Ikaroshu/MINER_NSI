from parameters import *
from numpy import sqrt, exp
from scipy.special import spherical_jn, erf


def e_rates(er, det, fx, epsi, op):
    eu = epsi.epu
    ed = epsi.epd
    qvs = (0.5 * det.z * (0.5 - 2 * ssw + 2 * eu['ee'] + ed['ee']) +
           0.5 * det.n * (-0.5 + eu['ee'] + 2 * ed['ee'])) ** 2 + \
          (0.5 * det.z * (2 * eu['em'] + ed['em']) + 0.5 * det.n * (eu['em'] + 2 * ed['em'])) * \
        conj(0.5 * det.z * (2 * eu['em'] + ed['em']) + 0.5 * det.n * (eu['em'] + 2 * ed['em'])) + \
          (0.5 * det.z * (2 * eu['et'] + ed['et']) + 0.5 * det.n * (eu['et'] + 2 * ed['et'])) * \
        conj(0.5 * det.z * (2 * eu['et'] + ed['et']) + 0.5 * det.n * (eu['et'] + 2 * ed['et']))
    m = dot(det.m, det.fraction)
    return dot(2 / pi * (gf ** 2) * (2 * fx.fint(er, m, epsi, 0, op) - det.m * er * fx.fintinvs(er, m, epsi, 0, op)) *
               det.m * qvs * formfsquared(sqrt(2 * det.m * er), det.z + det.n), det.fraction)


def m_rates(er, det, fx, epsi, op):
    eu = epsi.epu
    ed = epsi.epd
    qvs = (0.5 * det.z * (0.5 - 2 * ssw + 2 * eu['mm'] + ed['mm']) +
           0.5 * det.n * (-0.5 + eu['mm'] + 2 * ed['mm'])) ** 2 + \
          (0.5 * det.z * (2 * eu['em'] + ed['em']) + 0.5 * det.n * (eu['em'] + 2 * ed['em'])) * \
          (0.5 * det.z * (2 * eu['em'] + ed['em']) + 0.5 * det.n * (eu['em'] + 2 * ed['em'])) + \
          (0.5 * det.z * (2 * eu['mt'] + ed['mt']) + 0.5 * det.n * (eu['mt'] + 2 * ed['mt'])) * \
        conj(0.5 * det.z * (2 * eu['mt'] + ed['mt']) + 0.5 * det.n * (eu['mt'] + 2 * ed['mt']))
    m = dot(det.m, det.fraction)
    return dot(2 / pi * (gf ** 2) * (2 * fx.fint(er, m, epsi, 1, op) - det.m * er * fx.fintinvs(er, m, epsi, 1, op)) *
               det.m * qvs * formfsquared(sqrt(2 * det.m * er), det.z + det.n), det.fraction)


def t_rates(er, det, fx, epsi, op):
    eu = epsi.epu
    ed = epsi.epd
    qvs = (0.5 * det.z * (0.5 - 2 * ssw + 2 * eu['tt'] + ed['tt']) +
           0.5 * det.n * (-0.5 + eu['tt'] + 2 * ed['tt'])) ** 2 + \
          (0.5 * det.z * (2 * eu['et'] + ed['et']) + 0.5 * det.n * (eu['et'] + 2 * ed['et'])) * \
        conj(0.5 * det.z * (2 * eu['et'] + ed['et']) + 0.5 * det.n * (eu['et'] + 2 * ed['et'])) + \
          (0.5 * det.z * (2 * eu['mt'] + ed['mt']) + 0.5 * det.n * (eu['mt'] + 2 * ed['mt'])) * \
        conj(0.5 * det.z * (2 * eu['mt'] + ed['mt']) + 0.5 * det.n * (eu['mt'] + 2 * ed['mt']))
    m = dot(det.m, det.fraction)
    return dot(2 / pi * (gf ** 2) * (2 * fx.fint(er, m, epsi, 2, op) - det.m * er * fx.fintinvs(er, m, epsi, 2, op)) *
               det.m * qvs * formfsquared(sqrt(2 * det.m * er), det.z + det.n), det.fraction)


def formfsquared(er, a):
    r = 1.2 * (10 ** -15) * (a ** (1 / 3)) / (MeterByJoule * GeVPerJoule)
    s = 0.5 * (10 ** -15) / (MeterByJoule * GeVPerJoule)
    r0 = sqrt(r ** 2 - 5 * (s ** 2))
    return (3 * spherical_jn(1, er * r0) / (er * r0) * exp((-(er * s) ** 2) / 2)) ** 2
    # er = er * 1e6
    # t=9.05322e6 * exp(-1.9394e-5 * a * er ) * (-0.00692 * sqrt(2.17706 + pow(-0.6 + 1.23 * pow(a,0.3333),2)) * sqrt(a * er) * cos(0.00692 * sqrt(2.17706 + pow(-0.6 + 1.23 * pow(a,0.3333),2)) * sqrt(a * er)) + sin( 0.00692 * sqrt(2.17706 + pow(-0.6 + 1.23 * pow(a,0.3333),2)) * sqrt(a * er) )) / ( pow( 2.17706 + pow(-0.6 + 1.23 * pow(a,0.3333),2),1.5) * pow(a * er,1.5))
    # return t ** 2

def f2(er, a):
    r = 1.2 * (10 ** -15) * (a ** (1 / 3)) / (MeterByJoule * GeVPerJoule)
    s = 0.5 * (10 ** -15) / (MeterByJoule * GeVPerJoule)
    r0 = sqrt(r ** 2 - 5 * (s ** 2))
    return (3 * spherical_jn(1, er * r0) / (er * r0) * exp((-(er * s) ** 2) / 2)) ** 2

def rates(er, mv, det, fx, g):  # per nucleus
    deno = 2 * sqrt(2) * gf * (2 * det.m * er + mv ** 2) * (mb ** 2) / (2 * det.m * er)
    qvs = (0.5 * det.z * (rho*(0.5-2*knu*ssw)+2*lul+2*lur+ldl+ldr + 2 * g['uee'] / deno + g['dee'] / deno) +
           0.5 * det.n * (-0.5*rho+lul+lur+2*ldl+2*ldr + g['uee'] / deno + 2 * g['dee'] / deno)) ** 2 + \
          (0.5 * det.z * (2 * g['uem'] / deno + g['dem'] / deno) +
           0.5 * det.n * (g['uem'] / deno + 2 * g['dem'] / deno)) ** 2 + \
          (0.5 * det.z * (2 * g['uet'] / deno + g['det'] / deno) +
           0.5 * det.n * (g['uet'] / deno + 2 * g['det'] / deno)) ** 2
    if fx.ty == 'sns':
        m = det.m
    else:
        m = dot(det.m, det.fraction)
    if fx.ty == 'solar':
        epsi = Epsilon()
        epsi.epu['ee'] = g['uee'] * (mv ** 2) * 2 * sqrt(2) * gf
        epsi.epu['mm'] = g['umm'] * (mv ** 2) * 2 * sqrt(2) * gf
        epsi.epd['ee'] = g['dee'] * (mv ** 2) * 2 * sqrt(2) * gf
        epsi.epd['mm'] = g['dmm'] * (mv ** 2) * 2 * sqrt(2) * gf
    if fx.ty == 'snst':
        q = er
    else:
        q = sqrt(2 * det.m * er)
    return dot(2 / pi * (gf ** 2) * (2 * fx.fint(er, m) - 2 * er * fx.fintinv(er, m) + er * er *fx.fintinvs(er, m) - det.m * er * fx.fintinvs(er, m)) *
               det.m * qvs * formfsquared(q, det.z + det.n), det.fraction)


def totoal(expo, mv, det, fx, g):
    # print('g', g)
    if fx.ty == 'sns':
        return quad(snsrates, det.erMin, det.erMax, args=(mv, det, fx, g))[0] * \
               expo * JoulePerKg * GeVPerJoule * 24 * 60 * 60 / dot(det.m, det.fraction)
    return quad(rates, det.erMin, det.erMax, args=(mv, det, fx, g))[0] * \
        expo * JoulePerKg * GeVPerJoule * 24 * 60 * 60 / dot(det.m, det.fraction)


def binned_events(era, erb, expo, mv, det, fx, g):
    if fx.ty == 'sns':
        return quad(snsrates, era, erb, args=(mv, det, fx, g,))[0] * \
               expo * JoulePerKg * GeVPerJoule * 24 * 60 * 60 / dot(det.m, det.fraction)
    return quad(rates, era, erb, args=(mv, det, fx, g))[0] * expo * JoulePerKg * GeVPerJoule * 24 * 60 * 60 / \
        dot(det.m, det.fraction)


def binned_background(era, erb, det, expo):
    return det.background * (erb - era) * expo * 1e6  # because dru is per kev


def ratesm(er, mv, det, fx, g):
    deno = 2 * sqrt(2) * gf * (2 * det.m * er + mv ** 2) * (mb ** 2) / (2 * det.m * er)
    qvs = (0.5 * det.z * (rho*(0.5-2*knu*ssw)+2*lul+2*lur+ldl+ldr + 2 * g['umm'] / deno + g['dmm'] / deno) +
           0.5 * det.n * (-0.5*rho+lul+lur+2*ldl+2*ldr + g['umm'] / deno + 2 * g['dmm'] / deno)) ** 2 + \
          (0.5 * det.z * (2 * g['uem'] / deno + g['dem'] / deno) +
           0.5 * det.n * (g['uem'] / deno + 2 * g['dem'] / deno)) ** 2 + \
          (0.5 * det.z * (2 * g['umt'] / deno + g['dmt'] / deno) +
           0.5 * det.n * (g['umt'] / deno + 2 * g['dmt'] / deno)) ** 2
    if fx.ty == 'sns':
        m = det.m
    else:
        m = dot(det.m, det.fraction)
    if fx.ty == 'snst':
        q = er
    else:
        q = sqrt(2 * det.m * er)
    return dot(2 / pi * (gf ** 2) * (2 * fx.numfint(er, m)- 2*er*fx.numfinv(er,m)+er*er*fx.numfinvs(er, m) - det.m * er * fx.numfinvs(er, m)) *
               det.m * qvs * formfsquared(q, det.z + det.n), det.fraction)


def ratesp(er, mv, det, fx, g):
    deno = 2 * sqrt(2) * gf * (2 * det.m * er + mv ** 2) * (mb ** 2) / (2 * det.m * er)
    qvs = (0.5 * det.z * (rho*(0.5-2*knu*ssw)+2*lul+2*lur+ldl+ldr + 2 * g['umm'] / deno + g['dmm'] / deno) +
           0.5 * det.n * (-0.5*rho+lul+lur+2*ldl+2*ldr + g['umm'] / deno + 2 * g['dmm'] / deno)) ** 2 + \
          (0.5 * det.z * (2 * g['uem'] / deno + g['dem'] / deno) +
           0.5 * det.n * (g['uem'] / deno + 2 * g['dem'] / deno)) ** 2 + \
          (0.5 * det.z * (2 * g['umt'] / deno + g['dmt'] / deno) +
           0.5 * det.n * (g['umt'] / deno + 2 * g['dmt'] / deno)) ** 2
    if fx.ty == 'sns':
        m = det.m
    else:
        m = dot(det.m, det.fraction)
    if fx.ty == 'snst':
        q = er
    else:
        q = sqrt(2 * det.m * er)
    return dot(2 / pi * (gf ** 2) * (2 * fx.nupfint(er, m)- 2*er*fx.nupfinv(er,m)+er*er*fx.nupfinvs(er, m) - det.m * er * fx.nupfinvs(er, m)) *
               det.m * qvs * formfsquared(q, det.z + det.n), det.fraction)


def snsrates(er, mv, det, fx, g):
    if det.ty == 'csi':
        return (rates(er, mv, det, fx, g) + ratesm(er, mv, det, fx, g) + ratesp(er, mv, det, fx, g)) * \
           0.331 * (1 + erf(0.248 * (er * 1e6 - 9.22)))
    return rates(er, mv, det, fx, g) + ratesm(er, mv, det, fx, g) + ratesp(er, mv, det, fx, g)


def rates_e(er, mv, det, fx, g):
    m = massofe
    deno = 2 * sqrt(2) * gf * (2 * m * er + mv ** 2)
    epls = (1 + (-0.5 + ssw) + g['elee'] / deno) ** 2 + (g['elem'] / deno) ** 2 + (g['elet'] / deno) ** 2
    eprs = (ssw + g['eree'] / deno) ** 2 + (g['erem'] / deno) ** 2 + (g['eret'] / deno) ** 2
    eplr = (1 + (-0.5 + ssw) + g['elee'] / deno) * (ssw + g['eree'] / deno) + (g['elem'] / deno) * (g['erem'] / deno) +\
           (g['elet'] / deno) * (g['eret'] / deno)
    return dot(2 / pi * (gf ** 2) * m * det.z *
               (epls * fx.fint(er, m) +
                eprs * (fx.fint(er, m) - 2 * er * fx.fintinv(er, m) + er ** 2 * fx.fintinvs(er, m)) -
                eplr * m * er * fx.fintinvs(er, m)), det.fraction)


def ratesp_e(er, mv, det, fx, g):
    m = massofe
    deno = 2 * sqrt(2) * gf * (2 * m * er + mv ** 2)
    epls = (1 + (-0.5 + ssw) + g['elee'] / deno) ** 2 + (g['elem'] / deno) ** 2 + (g['elet'] / deno) ** 2
    eprs = (ssw + g['eree'] / deno) ** 2 + (g['erem'] / deno) ** 2 + (g['eret'] / deno) ** 2
    eplr = (1 + (-0.5 + ssw) + g['elee'] / deno) * (ssw + g['eree'] / deno) + (g['elem'] / deno) * (g['erem'] / deno) +\
           (g['elet'] / deno) * (g['eret'] / deno)
    return dot(2 / pi * (gf ** 2) * m * det.z *
               (epls * fx.nupfint(er, m) +
                eprs * (fx.nupfint(er, m) - 2 * er * fx.nupfinv(er, m) + (er ** 2) * fx.nupfinvs(er, m)) -
                eplr * m * er * fx.nupfinvs(er, m)), det.fraction)


def ratesm_e(er, mv, det, fx, g):
    m = massofe
    deno = 2 * sqrt(2) * gf * (2 * m * er + mv ** 2)
    epls = (1 + (-0.5 + ssw) + g['elee'] / deno) ** 2 + (g['elem'] / deno) ** 2 + (g['elet'] / deno) ** 2
    eprs = (ssw + g['eree'] / deno) ** 2 + (g['erem'] / deno) ** 2 + (g['eret'] / deno) ** 2
    eplr = (1 + (-0.5 + ssw) + g['elee'] / deno) * (ssw + g['eree'] / deno) + (g['elem'] / deno) * (g['erem'] / deno) +\
           (g['elet'] / deno) * (g['eret'] / deno)
    return dot(2 / pi * (gf ** 2) * m * det.z *
               (epls * fx.numfint(er, m) + eprs * (fx.numfint(er, m) - 2 * er * fx.numfinv(er, m) + er ** 2 * fx.numfinvs(er, m)) -
                eplr * m * er * fx.numfinvs(er, m)), det.fraction)


def re(er, mv, det, fx, g):
        return rates_e(er, mv, det, fx, g) + ratesm_e(er, mv, det, fx, g) + ratesp_e(er, mv, det, fx, g)


def binned_events_e(era, erb, expo, mv, det, fx, g):
    if fx.ty == 'sns':
        return quad(re, era, erb, args=(mv, det, fx, g))[0] * \
               expo * JoulePerKg * GeVPerJoule * 24 * 60 * 60 / dot(det.m, det.fraction)
    return quad(rates_e, era, erb, args=(mv, det, fx, g))[0] * \
        expo * JoulePerKg * GeVPerJoule * 24 * 60 * 60 / dot(det.m, det.fraction)