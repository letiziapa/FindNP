import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from utils.funcs import MLP, parton_data_preprocessing
from utils.protonXS import dsigma_dMttdytt_SM, dsigma_dMttdytt_c8qt

np.random.seed(2)
print('\nComparison between the NN result and the analytical evaluations for c8qt\n')

palette = sns.color_palette('Set2')
sns.set_palette(palette)
plt.rcParams['text.usetex'] = False
p = {'size': 17, 'family': 'cmr10'}

path_to_MC = 'data/PartonLevel/MonteCarlo_PL'

X, Xsm, Xc8qt, y = parton_data_preprocessing('c8qt', path_to_MC)

model = MLP(input_size = 2)
model.load_state_dict(torch.load('weights/PartonLevel_model_weights_c8qt.pth'))

Xsm = torch.tensor(Xsm, dtype=torch.float32)
Xc8qt = torch.tensor(Xc8qt, dtype=torch.float32)

# tot XS in fb
sm = pd.read_json(path_to_MC + '/sm.json')
c8qt = pd.read_json(path_to_MC + '/c8qt.json')
totXSSM = sum(sm['weights'])
totXSc8qt = sum(c8qt['weights'])

print('Calculating g(x,c) analytically...\n')

def g(Mtt, ytt):
    return 1 / (
        1
        + (dsigma_dMttdytt_c8qt(Mtt, ytt, 1) / totXSc8qt)
        / (dsigma_dMttdytt_SM(Mtt, ytt) / totXSSM)
    )

sm_ytt = np.array(sm["ytt"])
sm_mtt = np.array(sm["mtt"])
c8qt_ytt = np.array(c8qt["ytt"])
c8qt_mtt = np.array(c8qt["mtt"])

g_sm = [g(mtt, ytt) for mtt, ytt in zip(sm_mtt, sm_ytt)]
g_c8qt = [g(mtt, ytt) for mtt, ytt in zip(c8qt_mtt, c8qt_ytt)]

print('Done! \n')

### uncomment to plot and save the results
# print('Plotting results... \n')

# plt.figure(figsize=(10, 6))
# plt.hist(g_sm, bins=100, density=True, range=(0, 1), histtype="step", label="SM")
# plt.hist(g_c8qt, bins=100, density=True, range=(0, 1), histtype="step", label="$c^{(8)}_{qt}$")
# plt.hist(model(Xsm).detach().numpy(), range = (0,1),bins = 100, density= True,alpha = 0.5,label="SM (NN)")
# plt.hist(model(Xc8qt).detach().numpy(), range = (0,1),bins = 100,density = True,  alpha = 0.5, label="$c^{(8)}_{qt}$ (NN)")
# plt.title('Comparison between analytical results and Neural Network output', fontdict = p)
# plt.xticks(**p)    
# plt.yticks(**p)
# plt.xlabel('g(x,c)', fontdict = p)
# plt.legend(prop = p)
# plt.savefig('plots/DecisionBoundary_c8qt_PartonLevel.png')
# print('Saved plot to plots/DecisionBoundary_c8qt_PartonLevel.png')
# #plt.show()