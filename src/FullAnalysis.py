import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultranest import ReactiveNestedSampler
import seaborn as sns
import warnings
import torch
import scipy
from utils.funcs import complete_data_preprocessing, MLP, plot_confidence_regions, find_nsigma
from utils.protonXS  import dsigma_dMttdytt_SM, dsigma_dMttdytt_c8dt
from ultranest.plot import cornerplot

warnings.filterwarnings('ignore')


np.random.seed(0)

palette = sns.color_palette('Set2')
sns.set_palette(palette)
plt.rcParams['text.usetex'] = False
p = {'size': 17, 'family': 'cmr10'}

print('Implementation of the Nested Sampling algorithm using Machine Learning models')

LUMINOSITY = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_json('data/data.json')
sm = pd.read_json('data/MonteCarlo/SM.json')
c8dt = pd.read_json('data/MonteCarlo/c8dt.json')
c8dt2 = pd.read_json('data/MonteCarlo/c8dt2.json')
c8qt= pd.read_json('data/MonteCarlo/c8qt.json')
c8qt2 = pd.read_json('data/MonteCarlo/c8qt2.json')

totXSSM = sum(sm['weights'])
totXSc8qt = sum(c8qt['weights'])
totXSc8qt2 = sum(c8qt2['weights'])
totXSc8dt = sum(c8dt['weights'])
totXSc8dt2 = sum(c8dt2['weights'])

features = ['ptl+', 'ptl-', 'etal+', 'etal-', 'ptll', 'mll',
'deltaPhill', 'deltaEtall', 'ptb_lead', 'ptb_trail', 'ptbb', 'mbb',
'ST', 'MET']

#load and scale the data
Xc8dt, Xc8dt_sm, Xsm, Xdata, y = complete_data_preprocessing(c8dt, sm, data, features)
Xc8qt, Xc8qt_sm, _, _, _ = complete_data_preprocessing(c8qt, sm, data, features)
Xc8qt2, Xc8qt2_sm, _, _, _ = complete_data_preprocessing(c8qt2, sm, data, features)
Xc8dt2, Xc8dt2_sm, _, _, _ = complete_data_preprocessing(c8dt2, sm, data, features)

#load the trained models
model_c8dt = MLP(input_size=14).to(device)
model_c8qt = MLP(input_size=14).to(device)
model_c8qt2 = MLP(input_size=14).to(device)
model_c8dt2 = MLP(input_size=14).to(device)

model_c8dt.load_state_dict(torch.load('weights/Full_model_weights_c8dt.pth'))
model_c8qt.load_state_dict(torch.load('weights/Full_model_weights_c8qt.pth'))
model_c8qt2.load_state_dict(torch.load('weights/Full_model_weights_c8qt2.pth'))
model_c8dt2.load_state_dict(torch.load('weights/Full_model_weights_c8dt2.pth'))

### uncomment to show the decision boundaries calculated from the NN
### change model and input coefficient (Xc8dt, Xc8qt, Xc8dt2, Xc8qt2) accordingly
# plt.figure(figsize=(7.5, 5.5))
# plt.hist(model_c8dt(Xsm).detach().numpy(), range = (0,1),bins = 100, density= True,alpha = 0.5,label='SM')
# plt.hist(model_c8dt(Xc8dt).detach().numpy(), range = (0,1),bins = 100, density= True,alpha = 0.5,label='$c^{(8)}_{dt}$')
# plt.title('Decision boundary as predicted by the Neural Network', fontdict = p)
# plt.xticks(**p)
# plt.yticks(**p)
# plt.xlabel('g(x,c)', fontdict = p)
# plt.legend(prop=p, loc = 'upper center')
# #plt.savefig('plots/decision_boundary_c8dt_CompleteAnalysis.png')
# plt.show()

#calculate cross-section ratios
def r_c8dt(x):
    return (1 / (model_c8dt(x).detach().numpy())) -1

def r_c8dt2(x):
    return (1 / (model_c8dt2(x).detach().numpy()))-1

def r_c8qt(x):
    return (1 / (model_c8qt(x).detach().numpy()))-1

def r_c8qt2(x):
    return (1 / (model_c8qt2(x).detach().numpy()))-1

#calculate the logarithm of the argument between brackets in Equation 33 in the report
def log_ev(x, c8dt, c8qt):
    aux = np.maximum(c8dt * (totXSc8dt/totXSSM) * r_c8dt(x)
        + c8qt * (totXSc8qt/totXSSM) * r_c8qt(x)
        + c8dt**2 * (totXSc8dt2/totXSSM) * r_c8dt2(x)
        + c8qt**2 * (totXSc8qt2/totXSSM) * r_c8qt2(x) ,  -0.999999)
    
    result = 1 + aux

    return np.log(result)

#calculate the sum over all events of the logarithm
def log_total_ev(c8dt, c8qt):

    x = Xdata
    total_ev = sum(log_ev(x, c8dt, c8qt))

    return total_ev

#calculate nu_tot in Equation 33
def nu(c8dt, c8qt):
    return (totXSSM + c8dt*totXSc8dt + c8qt*totXSc8qt + c8dt**2*totXSc8dt2 + c8qt**2*totXSc8qt2)*LUMINOSITY

#construct the full log-likelihood function to be used in NS
def log_likelihood(params):
    c8dt, c8qt = params 

    result = - nu(c8dt, c8qt) + log_total_ev(c8dt, c8qt)
    result = np.squeeze(result)
    
    return result

#define the prior transform
def prior_volume(method = 'uniform', min_val= -2.0, max_val= 2.0):
    if method == 'uniform':
        def prior_transform(params):
            return params * (max_val - min_val) + min_val
   
        return prior_transform
    
#choose a prior range (in this case narrow)
min_prior_val = -2
max_prior_val = 2
prior = prior_volume('uniform', min_prior_val, max_prior_val)
print(f'Prior range:[{min_prior_val},{max_prior_val}]')
#specify the name of the parameters
params = [r"$c^{8}_{dt}$", r"$c^{8}_{qt}$"]

#run the sampler
sampler = ReactiveNestedSampler(params, log_likelihood, transform=prior)
result = sampler.run(min_num_live_points=1000)

#get samples from the posterior
samples = result['samples']
sampler.print_results()

max_likelihood = result["maximum_likelihood"]["point"]
C8dt_best, C8qt_best = max_likelihood


#calculate the significance of deviation from the SM
#calculate the log-likelihood assuming SM coefficients (0,0)
chi2SM = -2.0 * log_likelihood([0.0, 0.0])
#calculate the log-likelihood using the best-fit coefficient values
chi2Best = -2.0 * result["maximum_likelihood"]["logl"]

#Delta chi2 at 95% confidence has to be bigger than 5.991 (critical chi2)
print(f'Chi2 tension of the SM: {chi2SM - chi2Best:.3f}')

#calculate the critical chi2 with 2 degrees of freedom
print("Critical value with 2 dofs, 68%:", scipy.stats.chi2.isf(0.32, 2))
print("Critical value with 2 dofs, 95%:", scipy.stats.chi2.isf(0.05, 2))

print("--------------------------------------")


pvalue = 1 - scipy.stats.chi2.cdf(chi2SM - chi2Best, 2)
nsigma = scipy.optimize.fsolve(find_nsigma(pvalue), 0)

print(f'The SM has a tension of {nsigma[0]:.5f} sigma.')

### uncomment to visualise and save the corner plot
# plt.figure(figsize=(12, 10))
# cornerplot(result)
# plt.suptitle(f'Corner plot for the Wilson coefficients\nUniform prior range [{min_prior_val}, {max_prior_val}]', y = 1.1, fontsize = 18, fontname = 'cmr10')
# plt.savefig('plots/cornerplot_FullAnalysis.png', bbox_inches='tight')

#calculate the 95% credible intervals
lower_bound_C8dt, upper_bound_C8dt = np.percentile(samples[:, 0], [2.5, 97.5])
lower_bound_C8qt, upper_bound_C8qt = np.percentile(samples[:, 1], [2.5, 97.5])

print(f'Prior range:[{min_prior_val},{max_prior_val}]')
print(f"95% credible interval for C8dt: [{lower_bound_C8dt:.4f}, {upper_bound_C8dt:.4f}]")
print(f"95% credible interval for C8qt: [{lower_bound_C8qt:.4f}, {upper_bound_C8qt:.4f}]")

print(f"Maximum likelihood estimate for C8dt: {C8dt_best:.4f}")
print(f"Maximum likelihood estimate for C8qt: {C8qt_best:.4f}")
### uncomment to visualise and save the confidence regions
confidence_levels = [68,95,99]
#plot_confidence_regions(samples, params,confidence_levels, save = True, path = 'plots/CR_FullAnalysis.png')


