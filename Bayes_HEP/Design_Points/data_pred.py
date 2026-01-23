from Bayes_HEP.Design_Points import reader as Reader
import numpy as np
import os
import glob


x, x_errors, y_data_results, y_data_errors, y_train_results, y_train_errors, y_val_results, y_val_errors  = {},  {}, {}, {}, {}, {}, {}, {}

def get_system(filename):
    if "200__AuAu" in filename:
        return "AuAu200"
    elif "2760__PbPb" in filename:
        return "PbPb2760"
    elif "5020__PbPb" in filename:
        return "PbPb5020"
    return None


def get_data(all_data, system):
    for i, data_entry in enumerate(all_data):
    #for data_entry in all_data:
        #x_values = np.array(data_entry["Data"]["x"],i)
        x_values = np.column_stack((
            np.full(len(data_entry["Data"]["x"]), i),           # Λ column
            np.array(data_entry["Data"]["x"])                   # subobservable column
        ))
        x_errors_values = np.array(data_entry["Data"]["xerr"])
        y_data_values = np.array(data_entry["Data"]["y"])
        y_data_errors_values = np.array(data_entry["Data"]["yerr"])

        if system not in x:
            x[system] = x_values
            x_errors[system] = x_errors_values
            y_data_results[system] = y_data_values
            y_data_errors[system] = y_data_errors_values
        else:
            x[system] = np.concatenate((x[system], x_values))
            x_errors[system] = np.concatenate((x_errors[system], x_errors_values))
            y_data_results[system] = np.concatenate((y_data_results[system], y_data_values))
            y_data_errors[system] = np.concatenate((y_data_errors[system], y_data_errors_values))

    return x, x_errors, y_data_results, y_data_errors

def get_predictions(all_predictions, train_indices, validation_indices, system):
    for prediction in all_predictions:
        if "values" not in prediction["FileName"]:
            continue

        prediction_values = np.array(prediction["Prediction"])
        prediction_errors = np.array(prediction["PredictionErrors"])

        if system not in y_train_results:
            y_train_results[system] = prediction_values[train_indices]
            y_train_errors[system] = prediction_errors[train_indices]
            y_val_results[system] = prediction_values[validation_indices]
            y_val_errors[system] = prediction_errors[validation_indices]
        else:
            y_train_results[system] = np.hstack((y_train_results[system], prediction_values[train_indices]))
            y_train_errors[system] = np.hstack((y_train_errors[system], prediction_errors[train_indices]))
            y_val_results[system] = np.hstack((y_val_results[system], prediction_values[validation_indices]))
            y_val_errors[system] = np.hstack((y_val_errors[system], prediction_errors[validation_indices]))
         
    return y_train_results, y_train_errors, y_val_results, y_val_errors


### Leo ####
def get_design_index(main_dir):
    index_files = glob.glob(f"{main_dir}/input/Design/Design__Rivet__*.dat")
    index_files = [file for file in index_files if "Merged" not in file]
    index_numbers = [int(file.split("__")[-1].split(".")[0]) for file in index_files]
    return index_files

def get_max_design_index(main_dir):
    index_files = glob.glob(f"{main_dir}/input/Design/Design__Rivet__*.dat")
    index_files = [file for file in index_files if "Merged" not in file]
    index_numbers = [int(file.split("__")[-1].split(".")[0]) for file in index_files]
    max_index = max(index_numbers) if index_numbers else 0
    return index_files, max_index

def get_existing_design_points(index_files):
    existing_rows = []
    for oldfile in index_files:
        with open(oldfile) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                existing_rows.append(line)
    return existing_rows

def group_histograms_by_design(DG_predictions_files, merged_dir):
    hist_groups = {}
    for f in DG_predictions_files:
        parts = os.path.basename(f).split("__")
        parts = [p for p in parts if not p.startswith("DG")]
        key = "__".join(parts)
        hist_groups.setdefault(key, []).append(f)

    for key, DG_list in hist_groups.items():
        DG_list.sort(
            key=lambda p: int(
                next(
                    (t.split("_")[1] for t in os.path.basename(p).split("__") if t.startswith("DG_")),
                    "-1", 
                )
            )
        )

        headers = [line for line in open(DG_list[0]) if line.startswith("#")][:-1]
        data = [np.loadtxt(f) for f in DG_list]
        merged = np.column_stack(data)
        with open(f"{merged_dir}/{key}", "w") as f:
            f.writelines(headers)
            f.write("# " + " ".join(f"design_point{i+1}" for i in range(merged.shape[1])) + "\n")
            np.savetxt(f, merged, fmt="%.6e")

### Corey ####
def zeros_nan_remover(main_dir, all_predictions):
    
    prediction_dir = f"{main_dir}/input/Prediction_Merged"
    data_dir = f"{main_dir}/input/Data"

    removal_report_list = []

    prediction_files = os.listdir(prediction_dir)
    prediction_files.sort()
    prediction_files_path = [f'{prediction_dir}/{prediction_files}' for prediction_files in prediction_files]

    data_files = os.listdir(data_dir)
    data_files.sort()
    data_files_path = [f'{data_dir}/{data_files}' for data_files in data_files]

    all_predictions = [Reader.ReadPrediction(f) for f in prediction_files_path]

    #loop over all files, designpoints, and data points to see how many unfilled points and put it into line_delete list
    for file_number in range(len(all_predictions)):
        missing = 0
        design_point_number = 0
        line_delete = []

        for design_point in all_predictions[file_number]['Prediction']:
            design_point_number += 1
            data_point_number = 0 

            for data_point in design_point:
                data_point_number += 1

                if data_point == 0:
                    missing +=1
                    line_delete.append(data_point_number)

                if np.isnan(data_point):
                    missing +=1
                    line_delete.append(data_point_number)

        unique_line_delete = list(set(line_delete))
        unique_line_delete.sort()

        #create list of lines from file pre-removal
        with open(prediction_files_path[file_number], 'r') as f:
            file_list = []
            for line in f:
                file_list.append(line)

        #remove rows based on the line_delete list
        for x in unique_line_delete:
            file_list.pop(x-(data_point_number+1))

        #rewrite prediction file without the rows with zeros and nan
        with open(prediction_files_path[file_number], 'w') as f:
            f.writelines(file_list)
        
        split_path = prediction_files[file_number].split('__')

        if split_path[-1].split('.')[0] == 'values':

            print(f'{split_path[4]} {split_path[5]}     ROWS REMOVED: {len(unique_line_delete)}     TOTAL UNFILLED VALUES: {missing}')
            print(f'INDEX OF DATA POINTS REMOVED: {unique_line_delete}')
            print('----------------------------------------------------------------------------------')
        
            removal_report_list.append(f'{split_path[4]} {split_path[5]}     ROWS REMOVED: {len(unique_line_delete)}     TOTAL UNFILLED VALUES: {missing}\n')
            removal_report_list.append(f'INDEX OF DATA POINTS REMOVED: {unique_line_delete}\n\n')

            #remove lines from the data files
            
            with open(data_files_path[int((file_number-1)/2)], 'r') as f:
                file_list = []
                for line in f:
                    file_list.append(line)
            for x in unique_line_delete:
                file_list.pop(x-(data_point_number+1))
            with open(data_files_path[int((file_number-1)/2)], 'w') as f:
                f.writelines(file_list)

    #create removal report file 
    with open('removal_report.txt', 'w') as f:
        f.writelines(removal_report_list)
