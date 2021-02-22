# First steps with Leaspy

Welcome for the second practical session of the day!

## Objectives : 
- Learn to use Leaspy methods


## The set-up

As before, if you have followed the [installation details](https://gitlab.com/icm-institute/aramislab/disease-course-mapping-solutions) carefully, you should 

- be running this notebook in the `leaspy_tutorial` virtual environment
- having all the needed packages already installed

<span style='color: #a13203; font-weight: 600;'>💬 Question 1 💬</span> Run the following command lines

import os
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import main classes from Leaspy package
from leaspy import Leaspy, Data, AlgorithmSettings, IndividualParameters#, __watermark__

# Watermark trace with all packages versions
#print("\n".join([f'{pkg}: {pkg_version}' for pkg, pkg_version in __watermark__.items()]))

## Part I: Data

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> Data that can be used as a leaspy input should have the following format.

This result in multiple rows per subject. The input format **_MUST_** follow the following rules:
- A column named `ID`: corresponds to the subject indices
- A columns named `TIME`: corresponds to the subject's age at the corresponding visit
- One column per feature
- Each row is a visit, therefore the concatenation of the subject ID, the patient age at which the corresponding visit occured, and then the feature values

Concerning the features' values, as we are using a logistic model, they **_MUST_**:
- Be between 0 and 1
- In average increase with time for each subject (normal states correspond to values near 0 and pathological states to values near 1)

Moreover, to calibrate the progression model, we highly recommend to keep subjects that have been seen at least two times. You probably noticed that there are NaN: do not worry, Leaspy can handle them ;)

<span style='color: #a13203; font-weight: 600;'>💬 Question 2 💬</span> __Run the following lines to load the data.__

data_path = os.path.join(os.getcwd(),'..', "data/TP2_leaspy_beginner/")
df = pd.read_csv(data_path + 'simulated_data-corrected.csv')
df = df.set_index(["ID","TIME"])
df.head()

<span style='color: #a13203; font-weight: 600;'>💬 Question 3 💬</span> Does the data set seem to have the good format?

Your answer: ...

<span style='color: #a13203; font-weight: 600;'>💬 Question 4 💬</span> How many patients are there in the dataset?

# To complete

# –––––––––––––––– #
# –––– Answer –––– #
# –––––––––––––––– #

n_subjects = df.index.get_level_values('ID').unique().shape[0]
print(f'{n_subjects} subjects in the dataset.')

<span style='color: #a13203; font-weight: 600;'>💬 Question 5 💬</span> Create a training test that contains the first 160 patients and a testing set the rest. Each set will only contain the following features:
- MDS1_total
- MDS2_total
- MDS3_off_total

Help : Be careful, one patient is not one line ...

# To complete

df_train = ######################
df_test = ######################

# –––––––––––––––– #
# –––– Answer –––– #
# –––––––––––––––– #

df_train = df.loc[:'GS-160'][["MDS1_total", "MDS2_total", "MDS3_off_total"]]
df_test = df.loc['GS-161':][["MDS1_total", "MDS2_total", "MDS3_off_total"]]

### Leaspy's `Data` container


<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> _Leaspy_ comes with its own data containers. The one used in a daily basis is `Data`. You can load your data from a csv with it `Data.from_csv_file` or from a DataFrame `Data.from_dataframe`.

<span style='color: #a13203; font-weight: 600;'>💬 Question 6 💬</span> Run the following lines to convert DataFrame into Data object.

data_train = Data.from_dataframe(df_train)
data_test = Data.from_dataframe(df_test)

## Part II : Instantiate a `Leaspy` object

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> Before creating a leaspy object, you need to choose the type of progression shape you want to give to your data. The available models are the following:
- linear 
- logistic 

with the possibility to enforce a _parallelism_ between the features. **_Parallelism_** imposes that all the features have the same average pace of progression.

Once that is done, you just have to call `Leaspy('model_name')`. The dedicated names are  :
- `univariate_linear`
- `linear`
- `univariate_logistic`
- `logistic`
- `logistic_parallel`
- `lme_model`
- `constant_model`

<span style='color: #a13203; font-weight: 600;'>💬 Question 7 💬</span> We choose a logistic model. Run the following line to instantiate the leaspy object.

leaspy = Leaspy("logistic", source_dimension=2)

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> `Leaspy` object contains all the main methods provided by the software. With this object, you can:
- **calibrate** a model
- **personalize** a model to individual data (basically you infer the random effects with a gradient descent)
- **estimate** the features values of subjects at given ages based on your calibrated model and their individual parameters
- **simulate** synthetic subjects base on your calibrated model, a collection of individual parameters and data
- **load** and **save** a model

<span style='color: #a13203; font-weight: 600;'>💬 Question 8 💬</span> Check it out by running the following line

? Leaspy

<span style='color: #a13203; font-weight: 600;'>💬 Question 9 💬</span> This `Leaspy` object comes with an handy attribute for vizualization. Let's have a look on the data that will be used to calibrate our model

leaspy.model.dimension

ax = leaspy.plotting.patient_observations(data_train, alpha=.7, figsize=(14, 6))
ax.set_ylim(0,.8)
ax.grid()
plt.show()

Well... not so engaging, right? Let's see what Leaspy can do for you.

## Part III : Choose your algorithms

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> Once you choosed your model, you need to choose an algorithm to calibrate it.

To run any algorithm, you need to specify the settings of the related algorithm thanks to the `AlgorithmSettings` object. To ease Leaspy's usage for new users, we specified default values for each algorithm. Therefore, the name of the algorithm used is enough to run it. The one you need to fit your progression model is `mcmc_saem`, which stands for <a href="https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo" target="_blank">Markov chain Monte Carlo</a> - Stochastic Approximation of <a href="https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm" target="_blank">Expectation Maximization</a>. 

<span style='color: #a13203; font-weight: 600;'>💬 Question 10 💬</span> Run the following line to instanciate a `AlgorithmSettings` object.

algo_settings = AlgorithmSettings('mcmc_saem', 
                                  n_iter=3000,           # n_iter defines the number of iterations
                                  loss='MSE_diag_noise', # estimate the residual noise scaling per feature
                                  progress_bar=True)     # To display a nice progression bar during calibration

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> You can specify many more settings that are left by default for now. You can also save and load an `AlgorithmSettings` object in a json file.

<span style='color: #a13203; font-weight: 600;'>💬 Question 11 💬</span> Run the following line to get more informations.

? AlgorithmSettings

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> It is often usefull, even if it is optional to store the different logs of the model during the iterations. You can use the following method with the path of the folder where the logs will be stored.

<span style='color: #a13203; font-weight: 600;'>💬 Question 12 💬</span> Run the following lines.

algo_settings.set_logs(
    path='logs',          # Creates a logs file ; if existing, ask if rewrite it
    plot_periodicity=50,  # Saves the values to display in pdf every 50 iterations
    save_periodicity=10,  # Saves the values in csv files every 10 iterations
    console_print_periodicity=None,  # If = N, it display logs in the console/terminal every N iterations
    overwrite_logs_folder=True       # Default behaviour raise an error if the folder already exists.
)

## Part IV : Fit your model

<span style='color: #a13203; font-weight: 600;'>💬 Question 13 💬</span> Run the following lines to fit the model.

leaspy.fit(data_train, algorithm_settings=algo_settings)

#y_ij = f(model) + epsilon, epsilon ~ N(0, \sigma)

# –––––––––––––––– #
# –––– Answer –––– #
# –––––––––––––––– #

leaspy = Leaspy.load('outputs/model_parameters.json')

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> This might take several minutes, so let's discuss about the _keyword argument_ `source_dimension`. This parameters depend on the number of variable you want the model to learn: it can go from 1 to the number of variables. If it is not set by the user the default value is $\sqrt{N_{features}}$ as it has been shown empirically to give good results. You will learn more about the mathematical formulation of the model below (part V). 

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> Before assuming that the model is estimated, you have to check that the convergence went well. For that, you can look  the at the convergence during the iterations. To do so, you can explore the `logs` folder (in the same folder than this jupyter notebook) that shows the model convergence during the iterations. The first thing to look at is probably the `plots/convergence_1.pdf` and `plots/convergence_2.pdf` files : a run has had enough iterations to converge if the last 20 or even 30% of the iterations were stable for all the parameters. If not, you should provably re-run it with more iterations. 

from IPython.display import IFrame
IFrame('./logs/plots/convergence_2.pdf', width=990, height=670)

<span style='color: #a13203; font-weight: 600;'>💬 Question 14 💬</span> __Check out the parameters of the model that are stored here__

leaspy.model.parameters

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span>  Parameters are probably not straightfoward for now. The most important one is probably `noise_std`. It corresponds to the standard deviation of the Gaussian errors (one per feature). The smallest, the better - up to the lower bound which is the intrinsic noise in the data. Note that usually, cognitive measurements have an intrinsic error (computed on test-retest exams) between 5% and 10%.


<span style='color: #a13203; font-weight: 600;'>💬 Question 15 💬</span> Let's display `noise_std`

noise = leaspy.model.parameters['noise_std']
features = leaspy.model.features

print('Standard deviation of the residual noise for the feature:')
for n, f in zip(noise, features):
    print(f'- {f}: {n*100:.2f}%')

<span style='color: #a13203; font-weight: 600;'>💬 Question 16 💬</span> Save the model with the command below

leaspy.save("outputs/model_parameters.json", indent=2)

<span style='color: #a13203; font-weight: 600;'>💬 Question 17 💬</span> Load the model with the command below

leaspy = Leaspy.load('outputs/model_parameters.json')

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> Now that we have sufficient evidence that the model has converged, let's output what the average progression looks like! 

First, let's detail a bit what we are going to represent. We are going to display a trajectory: it corresponds to the temporal progression of the biomarkers. There is not only one trajectory for a cohort, as each subject has his or her own specific trajectory, meaning his or her disease progression. Each of these individual trajectories rely on individual parameters that are subject-specific. We will see those individual parameters a bit later, do not worry. For now, let's stick to the _average_ trajectory.

So what does the average trajectory corresponds to? The average trajectory correspond to a _virtual patient_ whose individual parameters are the average individual parameters. And these averages are already estimated during the calibration.

<span style='color: #a13203; font-weight: 600;'>💬 Question 18 💬</span> Let's plot the average trajectory

ax = leaspy.plotting.average_trajectory(alpha=1, figsize=(14,6))
ax.grid()
plt.show()

## Part V : Personalize the model to individual data

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> The personalization procedure allows to estimate the individual parameters that allows to modify the average progression to individual observations. The variations from the average trajectory to the individual one are encoded within three individual parameters : 
- $\alpha_i = \exp(\xi_i)$ : the acceleration factor, that modulates the speed of progression : $\alpha_i > 1$ means faster, $\alpha_i < 1$ means slower than the average progression
- $\tau_i$ : the time shift which delays the progression in a given number of years. It has to be compared to  `tau_mean` $ = \bar{\tau} $  which is in the model parameters above. In fact, $ \tau_i \sim \mathcal{N}( \bar{\tau}, \sigma_{\tau}^2)$ , so $\tau_i > \bar{\tau}$ means that the patient has a disease that starts later than average, while $\tau_i < \bar{\tau}$ means that the patient has a disease that starts earlier than average
- $w_i = (w_1, ..., w_N)$ ($N$ being the number of features) : the space-shift  which might, for a given individual, change the ordering of the conversion of the different features, compared to the mean trajectory.

In a nutshell, the $k$-th feature at the $j$-th visit of subject $i$, which occurs at time $t_{ij}$ writes: 

$$y_{ijk} = f_\theta ( w_{ik}+ \exp(\xi_i) * (t_{ij} - \tau_i) ) + \epsilon_{ijk}$$

With:
- $\theta$ being the population parameters, infered during calibration of the model,
- $f_\nu$ a parametric family of trajectories depending of model type,
- $\epsilon_{ijk}$ an independent normally distributed error term.

This writing is not exactly correct but helps understand the role of each individual parameters.

**[ Advanced ]** Remember the `sources`, or the `source_dimension`? Well, $w_i$ is not estimated directly, but rather thanks to a Independant Component Analysis, such that $w_i = A s_i$ where $s_i$ is of dimension $N_s$ = `source_dimension`. See associated papers for further details.

Now, let's estimate these individual parameters. The procedure relies on the `scipy_minimize` algorithm (gradient descent) that you have to define (or to load from an appropriate json file) :

<span style='color: #a13203; font-weight: 600;'>💬 Question 19 💬</span> First set the parameters

settings_personalization = AlgorithmSettings('scipy_minimize', progress_bar=True, use_jacobian=True)

<span style='color: #a13203; font-weight: 600;'>💬 Question 20 💬</span> Then use the second most important function of leaspy : `leaspy.personalize`. It estimates the individual parameters for the data you provide:

?leaspy.personalize

ip = leaspy.personalize(data_test, settings_personalization)

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> Note here that you can personalize your model on patients that have only one visit! And you don't have to use the same `data` as previously. It is especially useful, and important, in order to validate your model!

<span style='color: #a13203; font-weight: 600;'>💬 Question 20b 💬</span> Once the personalization is done, check the different functions that the `IndividualParameters` provides (you can save and load them, transform them to dataframes, etc) :

?ip

<span style='color: #a13203; font-weight: 600;'>💬 Question 21 💬</span> Plot the test data, but with reparametrized ages instead of real ages

# Plot the test data with individually reparametrized ages
ax = leaspy.plotting.patient_observations_reparametrized(data_test, ip, 
                                                         alpha=.7, linestyle='-', 
                                                         #patients_idx=list(data_test.individuals.keys())[:4],
                                                         figsize=(14, 6))
ax.grid()
plt.show()

Remember the raw plotting of values during question 9? Better, no?

# Plot the test data with individually with true ages
ax = leaspy.plotting.patient_observations(data_test,
                                          alpha=.7, linestyle='-', 
                                          #patients_idx=list(data_test.individuals.keys())[:4],
                                          figsize=(14, 6))
ax.grid()
plt.show()

Now, let's see what you can do with the individual parameters.

## Part VI : Impute missing values & predict individual trajectories

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> Together with the population parameters, the individual parameters entirely defines the individual trajectory, and thus, the biomarker values at any time. So you can reconstruct the individual biomarkers at different ages. 

You can reconstruct your observations at seen ages, i.e. at visits that have been used to personalize the model. There are two reasons you might want to do that:
- see how well the model fitted individual data
- impute missing values: as Leaspy handles missing values, it can then reconstruct them (note that this reconstruction will be noiseless)


The third very important function - after `leaspy.fit` and `leaspy.personalize` - is `leaspy.estimate`. Given some individual parameters and timepoints, the function estimates the values of the biomarkers at the given timepoints which derive from the individual trajectory encoded thanks to the individual parameters.

<span style='color: #a13203; font-weight: 600;'>💬 Question 22 💬</span> Check out the documentation

?leaspy.estimate

<span style='color: #a13203; font-weight: 600;'>💬 Question 23 💬</span> Before running `leaspy.estimate`, let's first retrieve the observations of subject 'GS-187' in the initial dataset. Get also his/her individual parameters as shown here:

observations = df_test.loc['GS-187']
print(f'Seen ages: {observations.index.values}')
print("Individual Parameters : ", ip['GS-187'])

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> The `estimate` first argument is a dictionary, so that you can estimate the trajectory of multiple individuals simultaneously (as long as the individual parameters of all your queried patients are in `ip`.

<span style='color: #a13203; font-weight: 600;'>💬 Question 24 💬</span> Now, let's estimate the trajectory for this patient.

timepoints = np.linspace(60, 100, 100)
reconstruction = leaspy.estimate({'GS-187': timepoints}, ip)

def plot_trajectory(timepoints, reconstruction, observations=None):

    if observations is not None:
        ages = observations.index.values
    
    plt.figure(figsize=(14, 6))
    plt.grid()
    plt.ylim(0, .75)
    plt.ylabel('Biomarker normalized value')
    plt.xlim(60, 100)
    plt.xlabel('Patient age')
    colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']
    
    for c, name, val in zip(colors, leaspy.model.features, reconstruction.T):
        plt.plot(timepoints, val, label=name, c=c, linewidth=3)
        if observations is not None:
            plt.plot(ages, observations[name], c=c, marker='o', markersize=12)
        
    plt.legend()
    plt.show()
                                
plot_trajectory(timepoints, reconstruction['GS-187'], observations)

# Or with plotting object
ax = leaspy.plotting.patient_trajectories(data_test, ip,
                                          patients_idx=['GS-187','GS-180'],
                                          labels=['MDS1','MDS2', 'MDS3 (off)'],
                                          #reparametrized_ages=True, # check sources effect
                                          
                                          # plot kwargs
                                          #color=['#003f5c', '#7a5195', '#ef5675', '#ffa600'],
                                          alpha=1, linestyle='-', linewidth=2,
                                          #marker=None,
                                          markersize=8, obs_alpha=.5, #obs_ls=':', 
                                          figsize=(16, 6),
                                          factor_past=.5,
                                          factor_future=5, # future extrapolation
                                          )
ax.grid()
#ax.set_ylim(0, .75)
ax.set_xlim(45, 120)
plt.show()

# Grasp source effects
ax = leaspy.plotting.patient_trajectories(data_test, ip,
                                          patients_idx='all',
                                          labels=['MDS1','MDS2', 'MDS3 (off)'],
                                          reparametrized_ages=True, # check sources effect
                                          
                                          # plot kwargs
                                          alpha=1, linestyle='-', linewidth=1,
                                          marker=None,
                                          figsize=(16, 6),
                                          factor_past=0,
                                          factor_future=0, # no extrapolation (future) nor past
                                          )
ax.grid()
#ax.set_ylim(0, .75)
#ax.set_xlim(45, 120)
plt.show()

## Part VII : Leaspy application - Cofactor analysis

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> Besides prediction, the individual parameters are interesting in the sense that they provide meaningful and interesting insights about the disease progression. For that reason, these individual parameters can be correlated to other cofactors. Let's consider that you have a covariate _Cofactor 1_ that encodes a genetic status: 1 if a specific mutation is present, 0 otherwise. 

<span style='color: #a13203; font-weight: 600;'>💬 Question 25 💬</span> Now, let's see if this mutation has an effect on the disease progression:

import seaborn as sns

# —— Convert individual parameters to dataframe
df_ip = ip.to_dataframe()

# —— Join the cofactors to individual parameters
cofactor = pd.read_csv(data_path + "cof_leaspy1.csv", index_col=['ID'])
df_ip = df_ip.join(cofactor.replace({'MUTATION':{0: 'Non-carrier', 1: 'Carrier'}}))

_, ax = plt.subplots(1, 2, figsize=(14, 6))

# —— Plot the time shifts in carriers and non-carriers
ax[0].set_title('Time shift histogram')
sns.histplot(data=df_ip, x='tau', hue='MUTATION', bins=15, ax=ax[0], stat='count', common_norm=False, kde=True)

# —— Plot the acceleration factor in carriers and non-carriers
ax[1].set_title('Log-Acceleration factor histogram')
sns.histplot(data=df_ip, x='xi', hue='MUTATION', bins=15, ax=ax[1], stat='count', common_norm=False, kde=True)

plt.show()

# __ Joint density (tau, xi) __
g = sns.jointplot(data=df_ip, x="tau", y="xi", hue="MUTATION", height=6)
g.plot_joint(sns.kdeplot, zorder=0, levels=8, bw_adjust=1.5)
g.ax_joint.grid();

 <span style='color: #a13203; font-weight: 600;'>💬 Question 26 💬</span> __Now, check your hypothesis with statistical tests__

# Shortcuts of df_ip for 2 sub-populationscarriers = df_ip[df_ip['MUTATION'] == 'Carrier']non_carriers = df_ip[df_ip['MUTATION'] == 'Non-carrier']
# —— Student t-test (under the asumption of a gaussian distribution only)
print(stats.ttest_ind(carriers['tau'], non_carriers['tau']))
print(stats.ttest_ind(carriers['xi'], non_carriers['xi']))

# —— Mann-withney t-test
print(stats.mannwhitneyu(carriers['tau'], non_carriers['tau']))
print(stats.mannwhitneyu(carriers['xi'], non_carriers['xi']))

## Part VIII : Data Simulation

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> Now that you are able to predict the evolution of a patient and use it to analyse cofactors, you might want to simulate a new one thanks to the information that you have learned. To do so you can use the last method of leaspy that we will study : `simulate`.

<span style='color: #a13203; font-weight: 600;'>💬 Question 27 💬</span> Have a look to the function

?leaspy.simulate

<span style='color: #015e75; font-weight: 600;'>ℹ️ Information ℹ️</span> To use the fuction we will first extract the individual parameters using personalize with `mode_real` option. The simulate function learns the joined distribution of the individual parameters and baseline age of the subjects
present in ``individual_parameters`` and ``data`` respectively to sample new patients from this joined distribution.

<span style='color: #a13203; font-weight: 600;'>💬 Question 28 💬</span> Define the settings for the personalization and get the individual parameters

settings_ip_simulate = AlgorithmSettings('mode_real')
individual_params = leaspy.personalize(data_test, settings_ip_simulate)

<span style='color: #a13203; font-weight: 600;'>💬 Question 29 💬</span> Define your algorithm for the simulation and simulate individuals from previously obtained individual parameters and dataset

settings_simulate = AlgorithmSettings('simulation')
simulated_data = leaspy.simulate(individual_params, data_test, settings_simulate)

<span style='color: #a13203; font-weight: 600;'>💬 Question 30 💬</span> Access to the individual parameters of one individual that you have created

print(simulated_data.get_patient_individual_parameters("Generated_subject_001"))

<span style='color: #a13203; font-weight: 600;'>💬 Question 31 💬</span> Plot the joint distribution of individual parameters (tau, xi) for simulated individuals that you have created

# Create a dataframe with individual parameters from both real & simulated individuals
df_ip_both = pd.concat({
    'estimated': individual_params.to_dataframe(),
    'simulated': simulated_data.get_dataframe_individual_parameters()
}, names=['Origin'])

g = sns.jointplot(data=df_ip_both, x='tau', y='xi', hue='Origin', height=8, 
                  marginal_kws=dict(common_norm=False))
g.plot_joint(sns.kdeplot, zorder=0, levels=8, bw_adjust=2.)
g.ax_joint.grid();

<span style='color: #a13203; font-weight: 600;'>💬 Question 32 💬</span> __Plot some simulated individual trajectories__

ax = leaspy.plotting.patient_observations(simulated_data.data, alpha=.7, figsize=(14, 4),
                                          patients_idx=[f'Generated_subject_{i:03}' for i in [1,2,7,42]])
ax.grid()
plt.show()