from Bayes_HEP.Design_Points import reader as Reader
from Bayes_HEP.Design_Points import design_points as DesignPoints
from Bayes_HEP.Design_Points import plots as Plots
from Bayes_HEP.Design_Points import data_pred as DataPred
from Bayes_HEP.Emulation import emulation as Emulation
from Bayes_HEP.Calibration import calibration as Calibration
from Bayes_HEP.Design_Points import rivet_html_parser as RivetParser

import os
import shutil
import matplotlib.pyplot as plt
import glob
import dill
    
###########################################################
################### SCRIPT PARAMETERS #####################
import argparse
import os

def str2bool(x):
    return str(x).lower() in ['true', '1', 'yes', 'y']

parser = argparse.ArgumentParser(description="Run Bayesian Emulator and Calibration workflow.")

parser.add_argument("--work_dir", type=str, default=None,
    help="Top-level working directory (default: $WORKDIR or /workdir)")
parser.add_argument("--main_dir", type=str, default=None,
    help="Project main directory (default: <work_dir>/New_Project)")
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--clear_output", type=str2bool, default=True)
parser.add_argument("--clean_input", type=str2bool, default=True)
parser.add_argument("--Coll_System", nargs="+", default=["pp_7000"],
    help="List of collision systems (e.g. pp_7000 pPb_5020)")
parser.add_argument("--model", type=str, default="pythia8")
parser.add_argument("--train_size", type=int, default=80,
    help="Percentage of design points for training")
parser.add_argument("--validation_size", type=int, default=20,
    help="Percentage of design points for validation")
parser.add_argument("--Train_Surmise", type=str2bool, default=True)
parser.add_argument("--Train_Scikit", type=str2bool, default=True)
parser.add_argument("--Load_Surmise", type=str2bool, default=True)
parser.add_argument("--Load_Scikit", type=str2bool, default=True)
parser.add_argument("--scaler_type", type=str, default=None,
    help="Type of scaler: StandardScaler, MinMaxScaler, RobustScaler, or None")
parser.add_argument("--PCA", type=str2bool, default=True)
parser.add_argument("--Run_Calibration", type=str2bool, default=True)
parser.add_argument("--nwalkers", type=int, default=50)
parser.add_argument("--npool", type=int, default=5)
parser.add_argument("--Samples", type=int, default=100)
parser.add_argument("--nburn", type=int, default=50)
parser.add_argument("--percent", type=float, default=0.15,
    help="Get traces for the last percentage of samples")
parser.add_argument("--Load_Calibration", type=str2bool, default=True)
parser.add_argument("--size", type=int, default=1000,
    help="Number of samples for results")
parser.add_argument("--Result_plots", type=str2bool, default=True)

args = parser.parse_args()

# Set/override variables using args
work_dir = args.work_dir or os.environ.get('WORKDIR', '/workdir')
main_dir = args.main_dir or f"{work_dir}/New_Project"
seed = args.seed
clear_output = args.clear_output
clean_input = args.clean_input 
Coll_System = args.Coll_System
model = args.model
train_size = args.train_size
validation_size = args.validation_size
Train_Surmise = args.Train_Surmise
Train_Scikit = args.Train_Scikit
Load_Surmise = args.Load_Surmise
Load_Scikit = args.Load_Scikit
scaler_type = args.scaler_type
PCA = args.PCA
Run_Calibration = args.Run_Calibration
nwalkers = args.nwalkers
npool = args.npool
Samples = args.Samples
nburn = args.nburn
percent = args.percent
Load_Calibration = args.Load_Calibration
size = args.size
Result_plots = args.Result_plots

Scale = scaler_type is not None

###########################################################
###########################################################
output_dir = f"{main_dir}/output"
if clear_output and os.path.exists(output_dir):
    print(f"Clearing output directory: {output_dir}")
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + "/plots", exist_ok=True)

############## Design Points ####################
print("Loading design points from input directory.")

index_files = DataPred.get_design_index(main_dir)
merged_Design_file = f"Design__Rivet__Merged.dat"
merged_output_file = f'{main_dir}/input/Design/{merged_Design_file}'

if not os.path.exists(merged_output_file):
    shutil.copy(f"{main_dir}/input/Rivet/parameter_prior_list.dat", merged_output_file)
    existing_rows = DataPred.get_existing_design_points(index_files)

    with open(merged_output_file, 'a') as f:
        f.write(f"\n\n# Total Design Points Merged = {len(existing_rows)}")
        f.write('\n' + "# Design point indices (row index): " + ' '.join(str(i) for i in range(len(existing_rows))) + '\n')   
        f.write("\n".join(existing_rows) + "\n")
    print(f"➕ Appended {len(existing_rows)} design points to {merged_output_file}")
print(f"Loading {merged_Design_file} from input directory.") 

RawDesign = Reader.ReadDesign(f'{main_dir}/input/Design/{merged_Design_file}')
priors, parameter_names, dim= DesignPoints.get_prior(RawDesign)
train_points, validation_points, train_indices, validation_indices = DesignPoints.load_data(train_size, validation_size, RawDesign['Design'], priors, seed)

Plots.plot_design_points(train_points, validation_points, priors)
plt.suptitle(f"Design Point Parameter Space", fontsize=18)
plt.savefig(f"{output_dir}/plots/Design_Points.png")
plt.show()

print("Loading input directory.")

prediction_dir, data_dir = f"{main_dir}/input/Prediction", f"{main_dir}/input/Data"
DG_predictions_files = glob.glob(f"{prediction_dir}/*.dat")
merged_dir = f"{main_dir}/input/Prediction_Merged"
os.makedirs(merged_dir, exist_ok=True)
DataPred.group_histograms_by_design(DG_predictions_files, merged_dir)

Data = {}
Predictions = {}
all_data = {}
n_hist = {}

for system in Coll_System:
    System, Energy = system.split('_')[0], system.split('_')[1]  
    sys = System + Energy   

    prediction_files = sorted(glob.glob(os.path.join(merged_dir, f"Prediction__{model}__{Energy}__{System}__*__values.dat")))
    data_files = sorted(glob.glob(os.path.join(data_dir, f"Data__{Energy}__{System}__*.dat")))

    all_predictions  = [Reader.ReadPrediction(f) for f in prediction_files]
    all_data[sys] = [Reader.ReadData(f) for f in data_files]

    if clean_input: DataPred.zeros_nan_remover(main_dir, all_predictions) 

    n_hist[sys] = len(prediction_files)

    x, x_errors, y_data_results, y_data_errors = DataPred.get_data(all_data[sys], sys)
    y_train_results, y_train_errors, y_val_results, y_val_errors = DataPred.get_predictions(all_predictions, train_indices, validation_indices, sys)

print("Data and predictions loaded successfully.")

######### Emulators ########
Emulators = {}
PredictionVal = {}
PredictionTrain = {}
scalers = {} if Scale else None
os.makedirs(output_dir + "/emulator", exist_ok=True)

if Scale:
    print("Scaling training and validation data.")

    for system in x.keys():  

        scaler = scaler_type()
        
        # Fit on experimental data
        scaler.fit(y_data_results[system].reshape(-1, 1))
        
        # Transform training data
        y_train_results_scaled = np.zeros_like(y_train_results[system])
        for i in range(y_train_results[system].shape[0]):  
            y_train_results_scaled[i, :] = scaler.transform(y_train_results[system][i, :].reshape(-1, 1)).flatten()
        y_train_results[system] = y_train_results_scaled
        
        # Transform validation data
        y_val_results_scaled = np.zeros_like(y_val_results[system])
        for i in range(y_val_results[system].shape[0]):  
            y_val_results_scaled[i, :] = scaler.transform(y_val_results[system][i, :].reshape(-1, 1)).flatten()
        y_val_results[system] = y_val_results_scaled
        
        # Store scaler
        scalers[system] = scaler
        
######### Surmise Emulator ########
if Train_Surmise:
    print("Training Surmise emulators.")
    method_type = 'PCGP' if PCA else 'indGP'
    Emulators['surmise'], PredictionVal['surmise_val'], PredictionTrain['surmise_train'] = Emulation.train_surmise(Emulators, x, y_train_results, train_points, validation_points, output_dir, method_type)
elif Load_Surmise:
    print("Loading Surmise emulator.")
    Emulators['surmise'] = {}
    Emulators['surmise'], PredictionVal['surmise_val'], PredictionTrain['surmise_train'] = Emulation.load_surmise(Emulators['surmise'], x, train_points, validation_points, output_dir)

######## Scikit-learn Emulator ########
if Train_Scikit:
    print("Training Scikit-learn emulator.")
    method_type = 'GP'
    if PCA:
        print("PCA is not supported for Scikit-learn emulator. Using standard Gaussian Process.") 
        
    Emulators['scikit'], PredictionVal['scikit_val'], PredictionTrain['scikit_train'] = Emulation.train_scikit(Emulators, x, y_train_results, train_points, validation_points, output_dir, method_type)
elif Load_Scikit:
    print("Loading Scikit-learn emulator.")
    Emulators['scikit'] = {}
    Emulators['scikit'], PredictionVal['scikit_val'], PredictionTrain['scikit_train'] = Emulation.load_scikit(Emulators['scikit'], x, train_points, validation_points, output_dir)

os.makedirs(f"{output_dir}/plots/emulators/", exist_ok=True)

if Train_Surmise or Load_Surmise or Train_Scikit or Load_Scikit:
    Plots.plot_rmse_comparison(y_train_results, y_val_results, PredictionTrain, PredictionVal, output_dir)
    
########### Calibration ###########
if Run_Calibration:
    print("Running calibration.")
    os.makedirs(f"{output_dir}/calibration/samples/", exist_ok=True)
    os.makedirs(f"{output_dir}/calibration/pos0/", exist_ok=True)
    os.makedirs(f"{output_dir}/plots/calibration/", exist_ok=True)  
    os.makedirs(f"{output_dir}/plots/trace/", exist_ok=True)

    results, samples_results, min_samples, map_params = Calibration.run_calibration(x, y_data_results, y_data_errors, priors, Emulators, output_dir, nburn, nwalkers, npool, Samples, scalers)
    
    Calibration.get_traces(output_dir, x, samples_results, Emulators, parameter_names, percent) 

if Load_Calibration:
    print("Calibration not performed. Loading Samples.")
    samples_results, min_samples, map_params= Calibration.load_samples(output_dir, x, Emulators)

########### Results ###########
if Result_plots:
    print("Generating results plots.")

    if min_samples < size:
        print(f"Warning: Minimum samples ({min_samples}) is less than requested size ({size}). Adjusting size to {min_samples}.")
        size = min_samples

    Plots.results(size, x, all_data, samples_results, y_data_results, y_data_errors, Emulators, n_hist, output_dir, scalers)
    
print("done")