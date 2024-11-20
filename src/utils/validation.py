from utils.tt_prod_ML4EFT import dsigma_dmtt_dy
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join("..")))
from utils.protonXS import dsigma_dMttdytt_SM, dsigma_dMttdytt_SMEFT


Mtt = np.linspace(400, 1000, 100)
ytt = np.linspace(-2.5, 2.5, 100)

for i in range(len(Mtt)):
    for j in range(len(ytt)):
        mine = dsigma_dMttdytt_SM(Mtt[i], ytt[j])
        # Tranform TeV in GeV and pb in fb
        ML4EFT = dsigma_dmtt_dy(ytt[j], Mtt[i] / 1e3) * 1e-3 * 1e3

        assert np.isclose(mine, ML4EFT)

print("All tests passed!")

# This test is done by removing the downXS from the c8qt dependence
# so that it behaves as ctu8
# also the quadratic term is removed in the sum of the cross section for the linear case
# while for the quadratic case, the linear term is removed


# c8qt = -2
# for i in range(len(Mtt)):
#     for j in range(len(ytt)):
#         mine = dsigma_dMttdytt_SMEFT(Mtt[i], ytt[j], c8qt, 0)
#         ML4EFT = dsigma_dmtt_dy(ytt[j], Mtt[i] / 1e3, (0, c8qt), order="quad") * 1e-3

#         assert np.isclose(mine, ML4EFT)
