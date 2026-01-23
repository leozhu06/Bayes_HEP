import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from matplotlib.lines import Line2D
import pandas as pd
            
def plot_design_points(train_points, validation_points, priors):
    param_names = list(priors.keys())  # Extract parameter names from priors
    num_params = len(param_names)  # Get the number of parameters

    fig, axes = plt.subplots(num_params, num_params, figsize=(15, 15))
    axes = axes.flatten()

    # Initialize variables to collect handles and labels for the legend
    handles, labels = None, None

    for i in range(num_params):
        for j in range(num_params):
            ax = axes[i * num_params + j]

            if j > i:
                ax.axis('off')  # Hide upper triangle plots
                continue

            if i == j:
                # Histogram for diagonal elements
                ax.hist(train_points[:, i], bins=20, alpha=0.5, label='Train', color='blue')
                ax.hist(validation_points[:, i], bins=20, alpha=0.5, label='Validation', color='orange')
                ax.set_ylabel('Frequency')

                # Collect handles and labels for the legend once
                if handles is None and labels is None:
                    handles, labels = ax.get_legend_handles_labels()
            else:
                # Scatter plot for lower triangle elements
                ax.scatter(train_points[:, j], train_points[:, i], alpha=0.5, label='Train', color='blue')
                ax.scatter(validation_points[:, j], validation_points[:, i], alpha=0.5, label='Validation', color='orange')

            # Set x and y labels dynamically
            if i == num_params - 1:  # Set x-axis label for the bottom row
                ax.set_xlabel(param_names[j], fontsize=14)
                ax.tick_params(axis='x', labelrotation=45)

            if j == 0:  # Set y-axis label for the first column
                ax.set_ylabel(param_names[i], fontsize=14)

    # Set a global legend
    fig.legend(handles, labels, fontsize=20, loc='upper right', bbox_to_anchor=(.95, .95))

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing


def remove_outliers(data):
    """Removes outliers using the IQR method."""
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]
 

def plot_combined_box_rmse(ax, y_true, predictions_dict, label):
    y_true = np.asarray(y_true).reshape(-1,1)

    rmse_filtered_list = []
    tick_labels = []

    for emulator_name, pred in predictions_dict.items():
        pred = np.asarray(pred).reshape(-1,1)

        rmse = np.sqrt((pred - y_true) ** 2)
        rmse_filtered = remove_outliers(rmse.flatten())
        rmse_filtered_list.append(rmse_filtered)
        tick_labels.append(emulator_name)

    ax.boxplot(rmse_filtered_list, tick_labels=tick_labels)
    ax.set_title(f'{label}', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)


def plot_rmse_comparison(y_train_results, y_val_results, PredictionTrain, PredictionVal, output_dir):
    systems = list(y_train_results.keys())
    emulator_names = list(PredictionTrain.keys())  

    # Parse emulator names from keys
    emulator_types = sorted(set(name.split('_')[0] for name in emulator_names))

    ncols = len(systems)
    fig, axes = plt.subplots(2, ncols, figsize=(10 * ncols, 16), sharey='row')

    if ncols == 1:  # Ensure axes is always 2D
        axes = axes.reshape(2, 1)

    for i, system in enumerate(systems):
        label = system.replace("_", "\n")

        # Collect emulator predictions for this system
        train_preds = {emu: PredictionTrain[f"{emu}_train"][system] for emu in emulator_types}
        val_preds = {emu: PredictionVal[f"{emu}_val"][system] for emu in emulator_types}

        plot_combined_box_rmse(axes[0, i], y_train_results[system], train_preds, label=label)
        plot_combined_box_rmse(axes[1, i], y_val_results[system], val_preds, label=label)

    axes[0, 0].set_ylabel('Training RMSE', fontsize=30)
    axes[1, 0].set_ylabel('Validation RMSE', fontsize=30)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/emulators/RMSE_Comparison.png")

    
def plot_trace(samples, parameter_names, title):
    num_params = samples.shape[1]
    fig, axes = plt.subplots(num_params, 1, figsize=(10, 2 * num_params), sharex=True)
    
    for i, param in enumerate(parameter_names):
        axes[i].plot(samples[:, i])
        axes[i].set_ylabel(param)
        axes[i].set_xlabel('Iteration')
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])


def results(size, x, all_data, samples_results, y_data_results, y_data_errors, Emulators, n_hist, output_dir, scalers=None):
    
    for i, system in enumerate(x.keys()):

        for em_type in Emulators:
            # Draw samples from the emulator
            rows = np.random.choice(samples_results[system][em_type].shape[0], size=size, replace=False)
            samples = samples_results[system][em_type][rows]
            
            #Make predictions using the emulator
            if em_type == 'surmise':
                sur_post = Emulators[em_type][system].predict(x[system], samples).mean()

                if scalers is not None:
                    sur_post = scalers[system].inverse_transform(sur_post)

            elif em_type == 'scikit':
                combined_result=[]

                for sample in samples:  
                    sample = np.atleast_1d(sample)
                    repeated = np.tile(sample, (x[system].shape[0], 1))  
                    combined_result.append(np.hstack((x[system], repeated)))
                combined_result = np.vstack(combined_result)
                post = Emulators[em_type][system].predict(combined_result)  
                post = np.asarray(post).reshape(-1)
                sk_post = post.reshape(samples.shape[0], x[system].shape[0]).T  

        for j in range(n_hist[system]):
            num_systems = len(x.keys())
            fig, ax = plt.subplots(figsize=(10, 8))


            handles, labels = [], []
            legend_added_for = set() 
        
            x_param = x[system][x[system][:, 0] == j]
            x_val = x_param[:, 1]

            start = sum(len(x[system][x[system][:, 0] == k]) for k in range(j))
            nbins = len(x_param)
            end = start + nbins

        
            for em_type in Emulators:

                if em_type == 'surmise':
                    post = sur_post[start:end, :] 
                elif em_type == 'scikit':
                    post = sk_post[start:end, :]  
                    
                observable     = all_data[system][j]["Observable"]
                subobservable  = all_data[system][j]["Subobservable"]
                experiment     = all_data[system][j]["Experiment"].upper()
                energy         = all_data[system][j]["Energy"]
                inspire        = all_data[system][j]["Inspire"]
                histogram      = all_data[system][j]["Histogram"]
                
                upper = np.percentile(post.T, 97.5, axis=0)
                lower = np.percentile(post.T, 2.5, axis=0)
                median = np.percentile(post.T, 50, axis=0)
                
                # Plot predictions
                line_pred = ax.plot(x_val, median, label=f'{em_type} Prediction', alpha=0.7)
                ax.fill_between(x_val, lower, upper, alpha=0.3)

                if em_type not in legend_added_for:
                    handles.append(line_pred[0])
                    labels.append(f'Calibrated Prediction {em_type.capitalize()}')
                    legend_added_for.add(em_type)

            # Plot experimental error bars
            line_err = ax.errorbar(
                x_val, y_data_results[system][start:end], yerr=[y_data_errors[system][start:end], y_data_errors[system][start:end]],
                color='black', fmt='o', markersize=5, capsize=3, label='stat + sys error'
            )
            
            if 'data' not in legend_added_for:
                handles.append(line_err)
                labels.append('stat + sys error')
                legend_added_for.add('data')

            if isinstance(subobservable, (list, tuple)):
                subobservable = " ".join(subobservable).strip()
            if isinstance(observable, (list, tuple)):
                observable = " ".join(observable).strip()
                
            ax.set_ylabel(observable, fontsize=24)
            ax.set_xlabel(subobservable, fontsize=24)
            ax.set_title(f"Predictions for {system} Collisions", fontsize=24)

            os.makedirs(f"{output_dir}/plots/results/{inspire}", exist_ok=True)

            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.01, 1), fontsize=16)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(f"{output_dir}/plots/results/{inspire}/{histogram}", bbox_inches='tight')
            plt.close(fig)