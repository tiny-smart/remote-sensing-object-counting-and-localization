import os
import re
import numpy as np
import matplotlib.pyplot as plt

folder_path = ['VIS']
flag_all=['NULL_PLEASE_ENTER_YOUR_MESSAGE']


def acc(gt_values, pred_values):
    acc=0
    wacc=0
    abs_errors = np.abs(pred_values - gt_values)
    #mean_abs_error = np.mean(pred_values)
    mean_gt=np.mean(gt_values)
    
    j=0
    err=0
    for i in range(len(abs_errors)):
        if gt_values[i]>1 and gt_values[i]<=8000:
            #and gt_values[i]<=200
            err+=abs_errors[i]/gt_values[i]
            #print(abs_errors[i]/gt_values[i])
            j+=1
    err=(err+0.00000001)/(j+0.00000001)
    print("Accurcy:",1-err)
    
    err=0
    grd=0
    for i in range(len(abs_errors)):
        if gt_values[i]>1 and gt_values[i]<=8000:
            #and gt_values[i]<=200
            err+=abs(pred_values[i]-gt_values[i])
            grd+=gt_values[i]
            #print(abs_errors[i]/gt_values[i])
    err=(err+0.00000001)/(grd+0.00000001)
    print("WeightedAccurcy:",1-err)
    
def show(gt_values, pred_values):
    A = np.vstack([gt_values, np.ones(len(gt_values))]).T
    a, b = np.linalg.lstsq(A,pred_values, rcond=None)[0]
    
    
    squared_errors = (gt_values - pred_values) ** 2
    mean_gt_values = np.mean(gt_values)
    ss_total = np.sum((gt_values - mean_gt_values) ** 2)
    ss_residual = np.sum(squared_errors)
    r2 = 1 - (ss_residual / ss_total)
    
    plt.scatter(gt_values, pred_values, label='Data')
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.title('GT vs Pred Scatter Plot')
    plt.grid(True)
    # Assuming gt_train is a list or numpy array
    pred_values = [a * x + b for x in gt_values]
    plt.plot(gt_values, pred_values, color='red', label=f'Pred = {a:.2f}*GT + {b:.2f}')
    plt.text(0.5, 0.1, f'Correlation Coefficient: {r2:.2f}', transform=plt.gca().transAxes, fontsize=10)
    plt.legend()
    #path='/data/ctf/workfile/Z_PHOTOS/'+flag_all[i]+'.png'
    #plt.savefig(path)
    plt.show()    


def calculate_metrics(gt_values, pred_values):
    # Ensure the inputs are numpy arrays
    gt_values = np.array(gt_values)
    pred_values = np.array(pred_values)
    
    # Calculate the number of samples
    n = len(gt_values)
    
    # Calculate MAE
    abs_errors = np.abs(gt_values - pred_values)
    mae = np.sum(abs_errors) / n
    
    # Calculate RMSE
    squared_errors = (gt_values - pred_values) ** 2
    mse = np.sum(squared_errors) / n
    rmse = np.sqrt(mse)
    
    # Calculate rMAE
    mean_abs_gt_values = np.mean(np.abs(gt_values))
    rmae = mae / mean_abs_gt_values
    
    # Calculate rMSE
    #rmse = np.sqrt(mse)
    rmse_normalized = rmse / mean_abs_gt_values
    
    # Calculate R^2
    mean_gt_values = np.mean(gt_values)
    ss_total = np.sum((gt_values - mean_gt_values) ** 2)
    ss_residual = np.sum(squared_errors)
    r2 = 1 - (ss_residual / ss_total)
    
    return mae, rmse, rmae, rmse_normalized, r2
for i in range(1):
    pattern = re.compile(r'(\d+)_gt(\d+)_pred(\d+)\.jpg')
    gt_values = []
    pred_values = []
    flag = 0
    for filename in os.listdir(folder_path[i]):
        if filename.endswith('.jpg'):

            match = pattern.match(filename)
            if match:

                gt = int(match.group(2))
                pred = int(match.group(3))
                if flag == 1:

                    if abs(gt - pred) <= 30:
                        gt_values.append(gt)
                        pred_values.append(pred)
                else:
                    gt_values.append(gt)
                    pred_values.append(pred)

    gt_values = np.array(gt_values)###############xiugaiweizijideshujv
    pred_values = np.array(pred_values)###############xiugaiweizijideshujv
    #########################################################
    mae, rmse, rmae, rmse_normalized, r2 = calculate_metrics(gt_values, pred_values)
    print("#############################")
    print("###Now:",flag_all[i],end="###")
    print()
    acc(gt_values, pred_values)

    print(f"MAE: {mae}")
    print(f"rMAE: {rmae}")
    print(f"RMSE: {rmse}")
    
    print(f"rMSE: {rmse_normalized}")
    print(f"R^2: {r2}")
    
    
    print("#############################")
    show(gt_values, pred_values)
    
    
    
    
    



    
    
    