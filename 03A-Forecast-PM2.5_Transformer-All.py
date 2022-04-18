import torch

import pickle
import time

import copy

import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import os

MODEL_FOLDER = "./tstrans_models/"
synthetic = False

from sklearn.preprocessing import MinMaxScaler

def create_in2out_sequences(input_data, window_size):
    in2out_seq = []
    L = len(input_data)
    
    for i in range(L-window_size):
        train_seq = input_data[i:i+window_size]
        train_label = input_data[i+window_size:i+window_size+output_window]
        in2out_seq.append((train_seq ,train_label))
        
        # Shape
        # ---
        # train_seq - (window_size, )
        # train_label - (window_size, )
    
    return torch.FloatTensor(in2out_seq)

def load_dataset(synthetic=False, dataset=None):
    if synthetic:
        time       = np.arange(0, 400, 0.1)    
        amplitude  = np.sin(time) + np.sin(time*0.05) + \
                     np.sin(time*0.12) * np.random.normal(-0.2, 0.2, len(time))
        return amplitude
    return dataset

def create_dataset(synthetic=False, dataset=None):
    
    values = load_dataset(synthetic, dataset)
    samples = int(0.7 * len(values))
        
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    values = scaler.fit_transform(values.reshape(-1, 1)).reshape(-1)
    
    train_data = values[:samples]
    test_data  = values[samples:]
    
    train_data = create_in2out_sequences(train_data, input_window)
    train_data = train_data[:-output_window] # Don't think much. Just did to fix errors.
    
    test_data = create_in2out_sequences(test_data, input_window)
    test_data = test_data[:-output_window] # Don't think much. Just did to fix errors.
    
    return train_data.to(device), test_data.to(device), scaler 

def get_batch(source, i, batch_size):
    
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    
    input_ = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    
    # Shape
    # ---
    # input_ - (input_window, batch_size, 1)
    # target - (input_window, batch_size, 1)
    # I know this looks counterintuitive, but bear with me. 
    # URL : https://stackoverflow.com/questions/65451265/pytorch-why-batch-is-the-second-dimension-in-the-default-lstm
    
    return input_, target


def train(model, optimizer, criterion, train_data):
    
    model.train() 
    
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
#             print('| Epoch {:03d} | {:05d}/{:05d} batches | '
#                   'lr {:02.6f} | {:5.2f} ms | '
#                   'loss {:5.5f}'.format(
#                    epoch, 
#                    batch, len(train_data) // batch_size, 
#                    scheduler.get_lr()[0],
#                    elapsed * 1000 / log_interval,
#                    cur_loss))
                  
            total_loss = 0
            start_time = time.time()
            
def evaluate(model, data_source, criterion):
    model.eval()
    
    total_loss = 0.
    eval_batch_size = 1000
        
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = model(data)            
            total_loss += len(data[0])* criterion(output, targets).cpu().item()
    
    return total_loss / len(data_source)

def fit_ts_transformer(model, optimizer, scheduler, criterion, epochs,
                       train_data, val_data, verbose=False):
    
    best_model = None

    early_stop_flag = False
    min_val_loss = 1e9
    bad_val_loss_ctr = 0
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):

        if (early_stop_flag):
            break

        epoch_start_time = time.time()
        train(model, optimizer, criterion, train_data)

        val_loss = evaluate(model, val_data, criterion)
        
        if verbose:

            print('-' * 89)

            print('| End of epoch {:03d} | time: {:5.2f}s | valid loss {:5.5f}'.format(epoch, 
                   (time.time() - epoch_start_time), 
                   val_loss))

            print('-' * 89)

        scheduler.step() 

        if (min_val_loss > val_loss):
            best_model = copy.deepcopy(model.state_dict())
            min_val_loss = val_loss
            bad_val_loss_ctr = 0
        else:
            bad_val_loss_ctr += 1

        if bad_val_loss_ctr >= 20:
            early_stop_flag = True
            
    model.load_state_dict(best_model)

input_window  = 1
output_window = 1

batch_size = 64
epochs = 200

lr = 0.005 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_predictions(model, scaler, data_source, criterion):
    
    model.eval() 
    
    total_loss = 0.
    y_hat = torch.Tensor(0)    
    y_test = torch.Tensor(0)
    
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = model(data)            
            total_loss += criterion(output, target).item()
            y_hat = torch.cat((y_hat, output[-1].view(-1).cpu()), 0)
            y_test = torch.cat((y_test, target[-1].view(-1).cpu()), 0)
            
    y_hat  = scaler.inverse_transform(y_hat.unsqueeze(-1))
    y_test = scaler.inverse_transform(y_test.unsqueeze(-1))

    return model, y_hat, y_test

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 

def get_TSTransformerModel(scaler, train_data, val_data):
    
    from tstransformer import TransTS

    import torch
    import torch.nn as nn
    
    model = TransTS(input_window).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = nn.MSELoss()
    
    fit_ts_transformer(model, optimizer, scheduler, criterion, epochs, train_data, val_data, verbose=False)
    
    model, y_pred, y_test = get_predictions(model, scaler, val_data, criterion)
    
    score = {
        "r2_score": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred), 
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mean": np.mean(y_test)
    }
    
    return model, score, scaler, y_pred, y_test

MODEL_LIST = [
    ('TSTransformer', get_TSTransformerModel),
]

plt.rc('text', usetex=True)  
plt.rc('font', family='sans-serif')

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def line_format(label):
    if (label.day % 15 == 0):
        return f"{label.day}\n{label.month_name()[:3]}"

def perform_modelling(train_data, val_data, scaler, test_idx_values, station_data, 
                      station_id, dataset_name, show_graph=True):

    best = {}
    best_score = 9e9

    rows = []

    predictions = {}

    for name, model_fn in MODEL_LIST:
        details = {}
        print(f"{station_id} - Getting {name} ...")
        
        model, score, scaler, y_pred, y_test = model_fn(scaler, train_data, val_data)
        row = [f"{name}", score['mae'], score['rmse'], score['r2_score'], score['mean']]
        rows.append(row)

        predictions[name] = y_pred.reshape(-1)
        predictions['Actual'] = y_test.reshape(-1)
        
        details['station_name'] = station_data[0]
        details['station_id'] = station_id
        details['name'] = name + ""
        details['model'] = model
        details['scaler'] = scaler
        details['score'] = score       
        details['test_set_predictions'] = y_pred
        details['test_set'] = y_test
        
        with open(MODEL_FOLDER + f"{dataset_name}/{name}/{station_id}_pm25.pkl", "wb") as file:
            details_model = details['model']
            details_model.save(MODEL_FOLDER + f"{dataset_name}/{name}/{station_id}_pm25.h5")
            del details['model']
            pickle.dump(details, file, protocol=4)

        if score['rmse'] < best_score:
            best = details
            best['model'] = details_model
            best_score = score['rmse']

    model_dfs = pd.DataFrame(rows, columns=["model", "mae", "rmse", "r2_score", "mean"])
    
    test_idx_values += datetime.timedelta(days=1)
    predictions_df = pd.DataFrame(predictions, index=test_idx_values)
    predictions_df.to_pickle(MODEL_FOLDER + f"{dataset_name}/{station_id}_predictions.pkl")
    
    if show_graph:
        fig = plt.figure(figsize=(18, 4))
        ax = fig.gca()
        
        predictions_df[['Actual', best['name']]].plot(ax=fig.gca())
        plt.title(f"{station_id} | Test Set", fontsize=14)
        plt.ylabel('PM2.5')
#         plt.grid(ls='--')
        
        msg_rmse = f"RMSE: {best['score']['rmse'].round(3)}"
        msg_mae =  f"MAE:  {best['score']['mae'].round(3)}"
        msg_mean = f"MEAN: {best['score']['mean'].round(3)}"
        
        msg = msg_rmse + '\n' + msg_mae + '\n' + msg_mean 
        
        ax.set_xticks(predictions_df.index)
        ax.set_xticklabels(map(line_format, predictions_df.index), rotation=0,  ha="center");
        
        ax.text(0.475, 0.85, msg,
             bbox=dict(facecolor='white', alpha=1),
             horizontalalignment='left',
             verticalalignment='center',
             fontsize=12,
             transform=ax.transAxes)

    return best, model_dfs, predictions_df, ax  

import torch.multiprocessing as mp

def main():
    
    os.system(f"rm -rf {MODEL_FOLDER}")
    os.mkdir(f"{MODEL_FOLDER}")
    
    POLLUTANT = "PM2.5"
    datasets = [
                "delhi", 
                "seoul", 
                "skopje", 
                "ulaanbaatar"
               ]         
                  
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    manager = mp.Manager()    
    ax_list = manager.list()        
    pool = mp.Pool(4)
    
    for dataset_name in datasets:
    
        try:
            os.mkdir(MODEL_FOLDER + dataset_name)   
        except Exception as e:
            print(e)

        try:
            for name, model_fn in MODEL_LIST:
                os.mkdir(MODEL_FOLDER + f"{dataset_name}/{name}")
        except Exception as e:
            print(e)
        
        dataset = pickle.load(open(f"./Data/{dataset_name}_dataset.pkl", "rb"))
        pool.starmap(perform_task, [(idx, ax_list, dataset, dataset_name) for idx in range(len(dataset))])

    pool.close()    
    pickle.dump(list(ax_list), open("transformer_dataset_pics.pkl", "wb"), protocol=4)
    
def perform_task(idx, ax_list, dataset, dataset_name):

    station_dict = dataset[idx]
    station_df = station_dict['df']
    station_dict['df'].index.freq = 'D'
    station_data = [station_dict[key] for key in list(station_dict.keys())[:-1]]
    
    train_data, val_data, scaler = create_dataset(synthetic=synthetic, dataset=station_dict['df'].values)
    test_idx_values = station_dict['df'].index.to_pydatetime()[-val_data.shape[0]+1:]

    best, model_dfs, predictions_df, ax = perform_modelling(train_data, val_data, scaler, test_idx_values, station_data, station_data[1], dataset_name, show_graph=True)
    
    ax_list.append(ax)    
    
    
if __name__ == "__main__":
    main()