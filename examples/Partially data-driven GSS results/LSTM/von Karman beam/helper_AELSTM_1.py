import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import os
import scipy.io as sio   

def create_dataset(dataset, forcing_dim, look_back=50):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, forcing_dim:])
    return np.array(X), np.array(Y)


def sim_pred(model, X, steps, x_train, encoding_dim, **kwargs):
    y=[]
    X0=X[0:1,:,:]
    input_log=[X0]
    forcing_dim = x_train.shape[2] - encoding_dim
    
    for i in range(x_train.shape[1]):
        _ = model.predict(X0, verbose=0)
    
    stateful_layers = kwargs["stateful_layers"]
    for i in range(steps):
        y.append(model.predict(X0, verbose=0))
        
        # for layer in stateful_layers:
        #     layer.reset_states()
            
        X0=X0[:,1:,:]
        tempvec=X[i+1:i+2,-1,0:forcing_dim]
        tempvec=np.concatenate((tempvec,y[i]),axis=1)
        tempvec=np.reshape(tempvec,(1,1,x_train.shape[2]))
        X0=np.concatenate((X0,tempvec),axis=1)
        input_log.append(X0)
    y.append(model.predict(X0))
    
    return y, input_log

def sim_pred_noy(model,F, y0, steps, look_back):
    y=[]
    F0 = F[0:look_back]
    X0 = np.concatenate((F0, y0), axis=1)
    X0 = np.expand_dims(X0, axis=0)
    for i in range(steps):
        y.append(model.predict(X0))
        tempvecF = F[look_back+1+i:look_back+i+2]
        tempvec = np.concatenate((tempvecF,y[i]),axis=1)
        X = np.concatenate((X0.squeeze(), tempvec), axis=0)
        X0 = X[-look_back:]
        X0 = np.expand_dims(X0, axis=0)
        input_log.append(X0)
    y.append(model.predict(X0))
    return y, input_log


def get_lstm_dataset(x, y, look_back):
    dataset_lstm = np.concatenate((x, y), axis=1)
    # Combine x and y
    forcing_dim = x.shape[1]
    x_lstm, y_lstm = create_dataset(dataset_lstm, forcing_dim, look_back)
    return x_lstm, y_lstm


def plot_AE_res(x_ground_truth, x_AE_processed, title_str, n_NNMs=9, dt = 0.1):
    '''
    Plot the original and autoencoder-processed trajectories for the first 20 elements of the oscillator chain.
    The function plots the trajectories for elements 1-10 and 11-20 separately.
    
    Parameters:
    x_ground_truth: np.array
        The original trajectory data.
    x_AE_processed: np.array
        The autoencoder-processed trajectory data.
    title_str: str
        The title of the plot.
    '''
    
    t = np.arange(0, x_ground_truth.shape[0]*dt, dt)
    
        
    def plot_10_AE_res(x_ground_truth, x_AE_processed, start_idx, t):
        plt.figure(start_idx, figsize=(13, 11))
        for i in range(1, 11):
            plt.subplot(5, 2, i)
            if start_idx + i-1 >= x_ground_truth.shape[1]:
                break
            plt.plot(t,x_ground_truth[:, start_idx + i-1], label='Original trajectory')
            plt.plot(t,x_AE_processed[:, start_idx + i-1], label='Processed trajectory', linestyle='--', alpha=0.9)
            #plt.legend(loc="upper right")
            
            # Format the plot
            fig, ax = plt.gcf(), plt.gca()
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xlim(0, t[-1])
            
        fig = plt.gcf()
        handles, labels = [], []
        for ax in fig.get_axes():
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)

        plt.suptitle(f'Original vs. Autoencoder-Processed Displacements\n{n_NNMs} NNMs (elements {start_idx+1}-{start_idx+10})\n{title_str}', fontsize=16)
    
    for i in range(0, x_ground_truth.shape[1], 10):
        plot_10_AE_res(x_ground_truth, x_AE_processed, i, t)
    

    plt.show()

def nmte(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the normalized mean trajectory error (NMTE) between
    y_true and y_pred, each of shape (N_time, D_dim).
    """
   
    N, D = y_true.shape
    y_norms = np.linalg.norm(y_true, axis=1)         
    y_max = y_norms.max()                    
    err = np.linalg.norm(y_true - y_pred, axis=1)    

    return err.sum() / (N * y_max)




def load_data(file, data_type, **kwargs):
    """
    Load a trajectory/forcing .mat file used for AE/LSTM training.

    Parameters
    ----------
    file : str
        Path to a Matlab .mat file (typically under data/), e.g., data/trajectory_<case>.mat
        produced by traj_VonKarmanBeam.m or data/SSM_details_forcing.mat. Must exist before
        loading.
    data_type : {"generated", "SSM_anchor"}
        - "generated": expects keys 't', 'F_ext', and 'y' from a generated trajectory file
          (time vector, external forcing history, and full-state response). If kwargs['tspan']
          is provided, it is checked against the stored time vector.
        - "SSM_anchor": expects keys 'Forcing_history' and 'Xwq' from an SSM anchor run
          (forcing history and modal/physical coordinates). If kwargs['tspan'] is provided, its
          length must match the forcing rows.
    **kwargs : optional
        tspan (np.ndarray): Optional time vector used for consistency checks when loading.

    Returns
    -------
    force : np.ndarray or None
        External forcing history with time along axis 0. None if the file is missing or malformed.
    full_model : np.ndarray or None
        State/coordinate trajectories with time along axis 0. None if the file is missing or malformed.
    """
    
       
    if not os.path.isfile(file):
        print(f'No .mat file found in the data directory for {file}')
        return None, None
    
    data = sio.loadmat(file)
    
    if data_type == "generated":
        tspan_ = data['t'].flatten()
        force = data['F_ext']
        full_model = data['y']
        
        if kwargs.get('tspan') is not None:
            assert (tspan_ == kwargs['tspan']).all(), "Time span mismatch between data and tspan"
        print(f'Loaded data from {file} successfully')
        
    elif data_type == "SSM_anchor":
        if 'Forcing_history' not in data or 'Xwq' not in data:
            print(f'Missing keys in the .mat file: {file}')
            return None, None
        force = data['Forcing_history'] # Potentially adapt if you change the keys used in matlab for saving
        full_model = data['Xwq']
        print(f'Loaded data from {file} successfully')
        
        if full_model.shape[0] < full_model.shape[1]:
            full_model = full_model.T
        if force.ndim == 2 and force.shape[0] < force.shape[1]:
            force = force.T
        
        if kwargs.get('tspan') is not None:
            assert len(kwargs['tspan']) == force.shape[0], "Time span length mismatch with force data"
    else:
        raise ValueError(f"Unknown data type: {data_type}, expected 'generated' or 'SSM_anchor'")
    
        
    return force, full_model

