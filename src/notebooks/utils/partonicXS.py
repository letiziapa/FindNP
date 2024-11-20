import numpy as np


def gg_tt_SM(s, gs, mt):
    return (
        -0.0006510416666666666
        * (
            gs**4
            * np.sqrt(1 - (4 * mt**2) / s)
            * (
                248 * mt**2
                + 56 * s
                - (
                    64
                    * (mt**4 + 4 * mt**2 * s + s**2)
                    * np.arctanh(np.sqrt(1 - (4 * mt**2) / s))
                )
                / np.sqrt(s * (-4 * mt**2 + s))
            )
        )
        / (np.pi * s**2)
    )


def uu_tt_SM(s, gs, mt):
    return (gs**4 * np.sqrt(-4 * mt**2 + s) * (2 * mt**2 + s)) / (54.0 * np.pi * s**2.5)


def dd_tt_SM(s, gs, mt):
    return (gs**4 * np.sqrt(-4 * mt**2 + s) * (2 * mt**2 + s)) / (54.0 * np.pi * s**2.5)


def uu_tt_c8qt(s, gs, mt, c8qt, Lambda):
    return (
        c8qt
        * (gs**2 * np.sqrt(-4 * mt**2 + s) * (2 * mt**2 + s))
        / (108.0 * Lambda**2 * np.pi * s**1.5)
    )


def uu_tt_c8qt2(s, gs, mt, c8qt, Lambda):
    return (
        c8qt**2
        * (np.sqrt(1 - (4 * mt**2) / s) * (-(mt**2) + s))
        / (216.0 * Lambda**4 * np.pi)
    )


def dd_tt_c8qt(s, gs, mt, c8qt, Lambda):
    return (
        c8qt
        * (gs**2 * np.sqrt(-4 * mt**2 + s) * (2 * mt**2 + s))
        / (108.0 * Lambda**2 * np.pi * s**1.5)
    )


def dd_tt_c8qt2(s, gs, mt, c8qt, Lambda):
    return (
        c8qt**2
        * (np.sqrt(1 - (4 * mt**2) / s) * (-(mt**2) + s))
        / (216.0 * Lambda**4 * np.pi)
    )


def dd_tt_c8dt(s, gs, mt, c8dt, Lambda):
    return (
        c8dt
        * (gs**2 * np.sqrt(-4 * mt**2 + s) * (2 * mt**2 + s))
        / (108.0 * Lambda**2 * np.pi * s**1.5)
    )


def dd_tt_c8dt2(s, gs, mt, c8dt, Lambda):
    return (
        c8dt**2
        * (np.sqrt(1 - (4 * mt**2) / s) * (-(mt**2) + s))
        / (216.0 * Lambda**4 * np.pi)
    )
