import numpy as np
import lhapdf
import os
from .partonicXS import (
    gg_tt_SM,
    uu_tt_SM,
    dd_tt_SM,
    uu_tt_c8qt,
    uu_tt_c8qt2,
    dd_tt_c8qt,
    dd_tt_c8qt2,
    dd_tt_c8dt,
    dd_tt_c8dt2,
)

# constants
mt = 172.0
alphaS = 0.130
gs = np.sqrt(4 * np.pi * alphaS)
Lambda = 1000
gevm2tofb = 0.3894 * 1e12
Ecoll = 13000
Q = 91.188

try:
    pdf = lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)
except:
    os.system("lhapdf install NNPDF23_lo_as_0130_qed")
    pdf = lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)

gluonChannels = [(21, 21)]
upChannels = [(2, -2), (4, -4)]
downChannels = [(1, -1), (3, -3), (5, -5)]


def PDFconv(partXS, pdf, channels, x1, x2, Q):
    protonConv = 0.0
    for pid1, pid2 in channels:
        protonConv += partXS * pdf.xfxQ(pid1, x1, Q) * pdf.xfxQ(pid2, x2, Q)
        # if different particles, consider also the opposite combination
        if pid1 != pid2:
            protonConv += partXS * pdf.xfxQ(pid2, x1, Q) * pdf.xfxQ(pid1, x2, Q)

    return protonConv


def dsigma_dMttdytt_SM(Mtt, ytt):
    s = Mtt**2
    x1 = Mtt / Ecoll * np.exp(ytt)
    x2 = Mtt / Ecoll * np.exp(-ytt)

    if (Mtt > 2 * mt) and (np.abs(ytt) < np.log(Ecoll / Mtt)):

        gluonXS = PDFconv(gg_tt_SM(s, gs, mt), pdf, gluonChannels, x1, x2, Q)
        upXS = PDFconv(uu_tt_SM(s, gs, mt), pdf, upChannels, x1, x2, Q)
        downXS = PDFconv(dd_tt_SM(s, gs, mt), pdf, downChannels, x1, x2, Q)

        return gevm2tofb * 2 * Mtt / (Ecoll**2) * (gluonXS + upXS + downXS) / (x1 * x2)
    else:
        return 0.0


def dsigma_dMttdytt_c8qt(Mtt, ytt, c8qt):
    s = Mtt**2
    x1 = Mtt / Ecoll * np.exp(ytt)
    x2 = Mtt / Ecoll * np.exp(-ytt)

    if (Mtt > 2 * mt) and (np.abs(ytt) < np.log(Ecoll / Mtt)):

        upXS = PDFconv(uu_tt_c8qt(s, gs, mt, c8qt, Lambda), pdf, upChannels, x1, x2, Q)
        downXS = PDFconv(
            dd_tt_c8qt(s, gs, mt, c8qt, Lambda), pdf, downChannels, x1, x2, Q
        )

        return gevm2tofb * 2 * Mtt / (Ecoll**2) * (upXS + downXS) / (x1 * x2)
    else:
        return 0.0


def dsigma_dMttdytt_c8qt2(Mtt, ytt, c8qt):
    s = Mtt**2
    x1 = Mtt / Ecoll * np.exp(ytt)
    x2 = Mtt / Ecoll * np.exp(-ytt)

    if (Mtt > 2 * mt) and (np.abs(ytt) < np.log(Ecoll / Mtt)):

        upXS = PDFconv(uu_tt_c8qt2(s, gs, mt, c8qt, Lambda), pdf, upChannels, x1, x2, Q)
        downXS = PDFconv(
            dd_tt_c8qt2(s, gs, mt, c8qt, Lambda), pdf, downChannels, x1, x2, Q
        )

        return gevm2tofb * 2 * Mtt / (Ecoll**2) * (upXS + downXS) / (x1 * x2)
    else:
        return 0.0


def dsigma_dMttdytt_c8dt(Mtt, ytt, c8dt):
    s = Mtt**2
    x1 = Mtt / Ecoll * np.exp(ytt)
    x2 = Mtt / Ecoll * np.exp(-ytt)

    if (Mtt > 2 * mt) and (np.abs(ytt) < np.log(Ecoll / Mtt)):

        downXS = PDFconv(
            dd_tt_c8dt(s, gs, mt, c8dt, Lambda), pdf, downChannels, x1, x2, Q
        )

        return gevm2tofb * 2 * Mtt / (Ecoll**2) * (downXS) / (x1 * x2)
    else:
        return 0.0


def dsigma_dMttdytt_c8dt2(Mtt, ytt, c8dt):
    s = Mtt**2
    x1 = Mtt / Ecoll * np.exp(ytt)
    x2 = Mtt / Ecoll * np.exp(-ytt)

    if (Mtt > 2 * mt) and (np.abs(ytt) < np.log(Ecoll / Mtt)):
        downXS = PDFconv(
            dd_tt_c8dt2(s, gs, mt, c8dt, Lambda), pdf, downChannels, x1, x2, Q
        )

        return gevm2tofb * 2 * Mtt / (Ecoll**2) * (downXS) / (x1 * x2)
    else:
        return 0.0


def dsigma_dMttdytt_SMEFT(Mtt, ytt, c8qt, c8dt):
    return (
        dsigma_dMttdytt_SM(Mtt, ytt)
        + dsigma_dMttdytt_c8qt(Mtt, ytt, c8qt)
        + dsigma_dMttdytt_c8qt2(Mtt, ytt, c8qt)
        + dsigma_dMttdytt_c8dt(Mtt, ytt, c8dt)
        + dsigma_dMttdytt_c8dt2(Mtt, ytt, c8dt)
    )


if __name__ == "__main__":
    Mtt = 1000
    ytt = 2
    c8qt = 0
    c8dt = 0
    print(f"Cross section: {dsigma_dMttdytt_SM(Mtt, ytt)}")
