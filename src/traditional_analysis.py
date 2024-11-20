import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultranest import ReactiveNestedSampler
from ultranest.plot import cornerplot
import seaborn as sns
import warnings
from scipy.optimize import minimize
import scipy.stats
warnings.filterwarnings('ignore')
from utils.funcs import plot_confidence_regions, find_nsigma
np.random.seed(0)

plt.rcParams['text.usetex'] = False
p = {'size': 17, 'family': 'cmr10'}

LUMINOSITY = 300

print('Binned analysis using the transverse momentum of the positive lepton (ptl+) observable')
#transverse momentum of the positive lepton (binning)
bins_ptl = [20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 460, 500, 550, 600, 800]


df = pd.read_json('data/data.json')
sm = pd.read_json('data/MonteCarlo/SM.json')
c8dt = pd.read_json('data/MonteCarlo/c8dt.json')
c8dt2 = pd.read_json('data/MonteCarlo/c8dt2.json')
c8qt= pd.read_json('data/MonteCarlo/c8qt.json')
c8qt2 = pd.read_json('data/MonteCarlo/c8qt2.json')

#cross section is derived from values of the weights
weights = sm['weights'].values * LUMINOSITY

#EFT datasets
weightsc8dt = c8dt['weights'].values * LUMINOSITY
weightsc8dt2 = c8dt2['weights'].values * LUMINOSITY
weightsc8qt = c8qt['weights'].values * LUMINOSITY
weightsc8qt2 = c8qt2['weights'].values * LUMINOSITY

Tc8dt, _ = np.histogram(c8dt['ptl+'], weights=weightsc8dt, bins=bins_ptl)
Tc8dt2, _ = np.histogram(c8dt2['ptl+'], weights=weightsc8dt2, bins=bins_ptl)
Tc8qt, _ = np.histogram(c8qt['ptl+'], weights=weightsc8qt, bins=bins_ptl)
Tc8qt2, _ = np.histogram(c8qt2['ptl+'], weights=weightsc8qt2, bins=bins_ptl)


#cross section is derived from values of the weights
weights = sm['weights'].values * LUMINOSITY

#TRANSVERSE MOMENTUM OF THE POSITIVE LEPTON

#experimental values X
Xi_ptl, _ = np.histogram(df['ptl+'], bins=bins_ptl)
#errors
sigmai_ptl = np.sqrt(Xi_ptl)
#theoretical values T
Ti_ptl, _ = np.histogram(sm["ptl+"], weights=weights, bins=bins_ptl)

plt.figure(figsize = (8, 5))
plt.title('Transverse momentum of the positive lepton (SM only)', fontsize=15, fontname = 'cmr10'  )
plt.errorbar(bins_ptl[:-1], Xi_ptl, yerr=sigmai_ptl, fmt='.', label='Data', color='midnightblue')
plt.step(bins_ptl[:-1], Ti_ptl, where='mid', label='MC', color = 'tab:red')
plt.xticks(fontsize = 13, fontname = 'cmr10')
plt.yticks(fontsize = 13, fontname = 'cmr10')
plt.xlabel('$p_{tl}^+$', fontsize = 13, fontname = 'cmr10')
plt.ylabel('Events', fontsize = 13, fontname = 'cmr10')
plt.legend(fontsize = 15)
plt.savefig('plots/ptl_pos.pdf')
print('Saved plot of ptl+ to plots/ptl_pos.pdf')

#chi squared is calculated from X values from the data and T values from MC simulation
def chi2_sm(Xi, Ti, sigmai):
    return np.sum((Xi-Ti)**2/sigmai**2)

chi2_ptl = chi2_sm(Xi_ptl, Ti_ptl, sigmai_ptl)

#define log-likelihood function to be used in the nested sampling algorithm
def log_likelihood(params):
    C8dt, C8qt = params
    expected_data = Ti_ptl + C8dt*Tc8dt + (C8dt**2)*Tc8dt2 + C8qt*Tc8qt + (C8qt**2)*Tc8qt2
    chi_squared = np.sum(((Xi_ptl - expected_data) / sigmai_ptl)**2)
    return -0.5 * chi_squared

#prior function (assuming flat priors and assuming the coefficients take values between -2 and 2)
def prior_tranform(params): 
    return params * 4 - 2

print('Running UltrAnest to sample from the posterior')
#define parameters name and ultranest settings
params = [r"$c^{8}_{dt}$", r"$c^{8}_{qt}$"]
sampler = ReactiveNestedSampler(params, log_likelihood, transform=prior_tranform)

#run ultranest to sample from the posterior
result = sampler.run(min_num_live_points=1000)

#get the samples
samples = result['samples']

sampler.print_results()
#determine the maximum likelihood points
max_likelihood = result["maximum_likelihood"]["point"]
C8dt_best, C8qt_best = max_likelihood


### uncomment to visualise corner plot
# plt.figure(figsize=(12, 10))
# fig = cornerplot(result)
# plt.suptitle('Corner plot for the Wilson coefficients ($p_{tl}^+$ observable)', y = 1.03, fontsize = 18, fontname = 'cmr10')
# plt.savefig('plots/corner_plot_binned.png', bbox_inches='tight')
# print('Saved corner plot to plots/corner_plot_binned.png')

#calculate the significance of deviation from the SM
#calculate the log-likelihood assuming SM coefficients (0,0)
chi2SM = -2.0 * log_likelihood([0.0, 0.0])

#calculate the log-likelihood using the best-fit coefficient values
chi2Best = -2.0 * result["maximum_likelihood"]["logl"]

#Delta chi2 at 95% confidence has to be bigger than 5.991 (critical chi2)
print(f'Chi2 tension of the SM: {chi2SM - chi2Best:.3f}')

#calculate the critical chi2 at 2 degrees of freedom
print("Critical value with 2 dofs, 68%:", scipy.stats.chi2.isf(0.32, 2))
print("Critical value with 2 dofs, 95%:", scipy.stats.chi2.isf(0.05, 2))

print("--------------------------------------")

#calculate the p-value and the corresponding number of standard deviations
pvalue = 1 - scipy.stats.chi2.cdf(chi2SM - chi2Best, 2)
print(f'p-value:{pvalue:.4f}')
nsigma = scipy.optimize.fsolve(find_nsigma(pvalue), 0)

print(f'The SM has a tension of {nsigma[0]:.5f} sigma.')

#95% credible intervals
lower_bound_C8dt, upper_bound_C8dt = np.percentile(samples[:, 0], [2.5, 97.5])
lower_bound_C8qt, upper_bound_C8qt = np.percentile(samples[:, 1], [2.5, 97.5])

print(f"95% credible interval for C8dt: [{lower_bound_C8dt:.4f}, {upper_bound_C8dt:.4f}]")
print(f"95% credible interval for C8qt: [{lower_bound_C8qt:.4f}, {upper_bound_C8qt:.4f}]")

print(f"Maximum likelihood estimate for C8dt: {C8dt_best:.4f}")
print(f"Maximum likelihood estimate for C8qt: {C8qt_best:.4f}")

confidence_levels = [68,95,99]
##uncomment to visualise confidence regions
#plot_confidence_regions(samples, params, confidence_levels, save = True, path = 'plots/confidence_regions_binned.png')
