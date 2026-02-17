import bilby
from bilby.core.prior import PriorDict, Uniform, Constraint
import numpy as np
import matplotlib.pyplot as plt
import dill
import os
from Bayes_HEP.Design_Points import plots as Plots



def run_calibration(x, y_data_results, y_data_errors, priors, Emulators, output_dir, nburn, nwalkers=10, npool=5, Samples=1000, scalers=None):
    class GaussianLikelihood(bilby.Likelihood):
        def __init__(self, x, y, sigma, emulator, em_type):
            self.x = x
            self.y = y
            self.sigma = np.asarray(sigma)
            self.N = len(x)
            self.emulator = emulator
            self.em_type = em_type
            super().__init__()
            self.parameters = dict()

        def log_likelihood(self):
            params = np.array([self.parameters[key] for key in self.parameters])
            try:
                if self.em_type =='surmise':
                    prediction = self.emulator.predict(self.x, params)
                    model = np.squeeze(prediction.mean().T)
                    var = np.squeeze(prediction.var().T)
                    error = np.sqrt(var)
                elif self.em_type == 'scikit': 
                    combined_result=[]
                    params =np.atleast_1d(params)
                    repeated = np.tile(params, (self.x.shape[0], 1))
                    combined_result.append(np.hstack((self.x, repeated)))
                    combined_result = np.vstack(combined_result)
                    model, error = self.emulator.predict(combined_result, return_std=True)
                    var = error**2
                else:
                    print(f"[Unknown emulator type] {self.em_type}")
                    return -np.inf

                var_safe = self.sigma**2 + error**2 + 1e-10
                sigma_safe = np.sqrt(var_safe)
                res = self.y - model

                log_l = -0.5 * np.sum((res / sigma_safe)**2 + np.log(2 * np.pi * sigma_safe**2))
                if np.isnan(log_l) or np.isinf(log_l):
                    print("[Invalid LogL] NaN or inf detected in log-likelihood")
                    return -np.inf

                return log_l

            except Exception as e:
                print(f"[Likelihood Exception] {e}")
                return -np.inf


    samplers = dict(
        #bilby_mcmc=dict(nsamples=10, L1steps=20, ntemps=10, printdt=10,),
        #dynesty=dict(npoints=50, sample="acceptance-walk", naccept=20),
        #pymultinest=dict(nlive=50),
        #nestle=dict(nlive=500),
        emcee=dict(nwalkers=nwalkers, iterations= Samples, nburn=nburn),
        #ptemcee=dict(ntemps=10, nwalkers=20, nsamples=10),
    )

    results= dict()
    samples_results = {}
    map_params = {}
    emu_pos = {}

    min_samples = float('inf')  # Initialize to a large number

    for system in x.keys(): 
        samples_results[system] = {}
        map_params[system] = {}
        for em_type in Emulators:
            Results={}
            emulator = Emulators[em_type][system]

            if scalers is not None:
                y_data_scaled = scalers[system].transform(y_data_results[system].reshape(-1, 1)).flatten()
                sigma_scaled = y_data_errors[system] / scalers[system].scale_
            else:
                y_data_scaled = y_data_results[system]
                sigma_scaled = y_data_errors[system]


            likelihood = GaussianLikelihood(x[system], y_data_scaled, sigma_scaled, emulator, em_type)
            
            pos0 = None

            # Try to load existing pos0
            pos_path = f"{output_dir}/calibration/pos0/{system}__{em_type}_pos.dill"
            if os.path.exists(pos_path):
                with open(pos_path, "rb") as f:
                    pos0 = dill.load(f)
                    print(f"[Loaded pos0] for {system} {em_type}")

            for sampler in samplers:
                Results = bilby.core.sampler.run_sampler(
                    likelihood=likelihood,
                    priors=priors,
                    sampler=sampler,
                    outdir=f"{output_dir}/calibration/{em_type}/{system}_results",
                    label=sampler,
                    pos0=pos0,
                    npool=npool,
                    resume=True,
                    clean=False,
                    verbose=False,
                    **samplers[sampler]
                )
                results[em_type + '_' + sampler] = Results

                if sampler == 'emcee':
                    samples_results[system][em_type] = Results.samples
                    #Find MAP 
                    posterior = Results.posterior
                    posterior["log_posterior"] = posterior["log_likelihood"] + posterior["log_prior"]
                    idx = posterior["log_posterior"].idxmax()
                    param_cols = [col for col in posterior.columns if not col.startswith('log_')]
                    map_params[system][em_type] = posterior.loc[idx, param_cols].to_dict()
                    
                    if system not in emu_pos:
                        emu_pos[system] = {}

                    emu_pos[system][em_type] = Results.samples[-nwalkers:]

            with open(f"{output_dir}/calibration/pos0/{system}__{em_type}_pos.dill", "wb") as f:
                    dill.dump(emu_pos[system][em_type], f)

            with open(f"{output_dir}/calibration/samples/{system}__{em_type}.dill", "wb") as f:
                    dill.dump(samples_results[system][em_type], f)
            
            with open(f"{output_dir}/calibration/samples/{system}__{em_type}_map.dill", "wb") as f:
                    dill.dump(map_params[system][em_type], f)

            fig = bilby.core.result.plot_multiple(list(results.values()), labels=list(results.keys()), save=True, outdir=f'{system}.png')
            plt.suptitle(system, fontsize=16)
            plt.savefig(f"{output_dir}/plots/calibration/calibration_{system}", bbox_inches='tight')
            plt.close(fig)

            min_samples = min(len(samples_results[system][em_type]), min_samples)

    return results, samples_results, min_samples, map_params

def load_samples(output_dir, x, Emulators):
    samples_results = {}
    map_params = {}
    min_samples = float('inf')

    for system in x.keys():
        samples_results[system] = {}
        map_params[system] = {}
        for em_type in Emulators:
            filepath = f"{output_dir}/calibration/samples/{system}__{em_type}.dill"
            mapfile = f"{output_dir}/calibration/samples/{system}__{em_type}_map.dill"
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    samples_results[system][em_type] = dill.load(f)
                min_samples = min(len(samples_results[system][em_type]), min_samples)
            else:
                print(f"Warning: No samples found for {system} with emulator {em_type}. Skipping.")

            # Load the MAP params if they exist
            if os.path.exists(mapfile):
                with open(mapfile, "rb") as f:
                    map_params[system][em_type] = dill.load(f)
            else:
                map_params[system][em_type] = None

    return samples_results, min_samples, map_params

def get_traces(output_dir, x, samples_results, Emulators, parameter_names, percent):
    last_15 = {}
    for system in x.keys():
        last_15[system] = {}
        for em_type in Emulators:
            # Extract the last 15% of samples
            last_15[system][em_type] = samples_results[system][em_type][int((1 - percent) * len(samples_results[system][em_type])):]
            
            # Apply thinning: select every 10th sample
            thin = last_15[system][em_type][::10]
            
            # Plot the thinned samples
            Plots.plot_trace(thin, parameter_names, ' Trace '+ system + ' ' + em_type)
            plt.savefig(f"{output_dir}/plots/trace/traceplot_{system}_{em_type}.png", bbox_inches='tight')
            plt.close()  
