import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator   
import os
import pickle

def create_dataset(dataset, forcing_dim, look_back=50):
    """
    Create LSTM training dataset with sequences and targets.
    
    Splits time series data into input sequences (X) and corresponding output targets (Y)
    for LSTM training using a sliding window approach.
    
    Parameters
    ----------
    dataset : np.ndarray
        Time series data of shape (n_timesteps, n_features) where features include both
        forcing dimensions and state variables.
    forcing_dim : int
        Number of forcing dimensions at the start of each feature vector. These will be
        excluded from the output targets Y.
    look_back : int, optional
        Length of the input sequence window (default=50). Each X sample will contain
        look_back consecutive timesteps.
    
    Returns
    -------
    X : np.ndarray
        Input sequences of shape (n_samples, look_back, n_features).
    Y : np.ndarray
        Output targets of shape (n_samples, n_features - forcing_dim), corresponding to
        the state at timestep look_back+1.
    """
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, forcing_dim:])
    return np.array(X), np.array(Y)


def sim_pred(model, X, steps, x_train, encoding_dim, **kwargs):
    """
    Simulate predictions using an LSTM model with autoregressive feedback.
    
    Performs multi-step ahead prediction by feeding the model's outputs back as inputs
    for subsequent predictions. Useful for long-term trajectory forecasting.
    
    Parameters
    ----------
    model : keras.Model
        Trained LSTM model for prediction. Should accept inputs of shape
        (batch_size, look_back, n_features) and output shape (batch_size, encoding_dim).
    X : np.ndarray
        Input sequence data of shape (n_timesteps, look_back, n_features) containing both
        forcing and state variables.
    steps : int
        Number of prediction steps to simulate forward in time.
    x_train : np.ndarray
        Training data of shape (n_samples, look_back, n_features), used to determine
        the forcing dimension and for model warm-up.
    encoding_dim : int
        Dimension of the encoded/latent state variables that the model predicts.
    **kwargs : dict
        Additional keyword arguments. Should include 'stateful_layers' (list of stateful
        layers in the model, currently not used but reserved for state reset).
    
    Returns
    -------
    y : list of np.ndarray
        List of predictions, each of shape (1, encoding_dim), containing the predicted
        state at each time step.
    input_log : list of np.ndarray
        Log of input sequences used at each prediction step, each of shape
        (1, look_back, n_features).
    """
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

def sim_pred_noy(model, F, y0, steps, look_back):
    """
    Simulate predictions with external forcing and initial state.
    
    Performs multi-step ahead prediction by combining external forcing data with
    predicted states in an autoregressive manner.
    
    Parameters
    ----------
    model : keras.Model
        Trained LSTM model for prediction. Should accept inputs of shape
        (batch_size, look_back, n_features).
    F : np.ndarray
        External forcing data of shape (n_timesteps, n_forcing_features) containing
        the forcing signal for all prediction steps.
    y0 : np.ndarray
        Initial state of shape (look_back, n_state_features) representing the initial
        conditions for the state variables.
    steps : int
        Number of prediction steps to simulate forward in time.
    look_back : int
        Length of the input sequence window used by the model.
    
    Returns
    -------
    y : list of np.ndarray
        List of predictions, each containing the predicted state at each time step.
    input_log : list of np.ndarray
        Log of input sequences used at each prediction step, each of shape
        (1, look_back, n_features).
    """
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
    """
    Prepare LSTM dataset from forcing and state data.
    
    Combines forcing (x) and state (y) data, then creates sequential input-output pairs
    suitable for LSTM training.
    
    Parameters
    ----------
    x : np.ndarray
        Forcing/input data of shape (n_timesteps, n_forcing_features) containing external
        forcing signals or input variables.
    y : np.ndarray
        State/output data of shape (n_timesteps, n_state_features) containing the system
        state variables to be predicted.
    look_back : int
        Length of the input sequence window for LSTM training.
    
    Returns
    -------
    x_lstm : np.ndarray
        Input sequences of shape (n_samples, look_back, n_forcing_features + n_state_features).
    y_lstm : np.ndarray
        Output targets of shape (n_samples, n_state_features), representing states at
        timestep look_back+1.
    """
    dataset_lstm = np.concatenate((x, y), axis=1)
    # Combine x and y
    forcing_dim = x.shape[1]
    x_lstm, y_lstm = create_dataset(dataset_lstm, forcing_dim, look_back)
    return x_lstm, y_lstm


def plot_AE_res(x_ground_truth, x_AE_processed, title_str, n_NNMs=9, dt=0.1):
    """
    Plot the original and autoencoder-processed trajectories for the oscillator chain.
    
    Creates subplots comparing ground truth and autoencoder-processed trajectories for
    each element of the oscillator chain. Plots are organized in groups of 10 elements.
    
    Parameters
    ----------
    x_ground_truth : np.ndarray
        Original trajectory data of shape (n_timesteps, n_elements) containing the true
        displacement values for each oscillator element over time.
    x_AE_processed : np.ndarray
        Autoencoder-processed trajectory data of shape (n_timesteps, n_elements) containing
        the reconstructed displacement values after encoding and decoding.
    title_str : str
        Title string to be included in the plot suptitle, typically describing the test
        case or parameter configuration.
    n_NNMs : int, optional
        Number of nonlinear normal modes (NNMs) used in the autoencoder (default=9).
        This is displayed in the plot title.
    dt : float, optional
        Time step size in seconds (default=0.1). Used to create the time axis for plotting.
    
    Returns
    -------
    None
        Displays matplotlib figures but does not return any value.
    """
    
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
    
def load_data(file_path):
    """
    Load pickled data from file and convert to numpy array.
    
    Reads a pickle file, converts the data to a numpy array, and removes singleton
    dimensions. Provides error handling for missing files.
    
    Parameters
    ----------
    file_path : str
        Path to the pickle file containing the data. Should be a valid file path
        to a .pkl or similar pickle file.
    
    Returns
    -------
    data : np.ndarray
        Loaded data as a numpy array with singleton dimensions removed. The shape
        depends on the original data structure stored in the pickle file.
    
    Raises
    ------
    FileNotFoundError
        If the specified file_path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        data = np.array(data).squeeze()
        print(f"Data loaded from {file_path} successfully.")
    return data
    
