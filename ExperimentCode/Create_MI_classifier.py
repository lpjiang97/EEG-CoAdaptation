from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import signal
from mne.filter import filter_data

import scipy.signal as scisig
import numpy as np
import pandas as pd
import datetime
import pickle
import glob
import csv
import mne

def LoadEEGData(filename, EEGdevice):
    """ This function converts a single .easy file (from NIC2) to an easy-to-use dataframe.
    Uses both the .easy file and .info file (containing metadata)
    
    ---- Input ----
    filename: string containing the .easy filepath
    
    ---- Output ----
    df: dataframe containing all the EEG, accelerometer, and event marker data
    fs: sampling rate for the EEG data (Hz)
    fs_accel: sampling rate for the accelerometer data (Hz)
    
    """
    if EEGdevice == 7:
        x = 1
    elif EEGdevice == 8:
        # Read in the .easy file
        df = pd.read_csv(filename, delimiter='\t', header=None)

        # Get metadata from the .info file
        fname = filename[:-5] + '.info'
        with open(fname) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        # Get the channel names
        channel_info = [x for x in content if 'Channel ' in x]
        channel_names = []
        for ch in range(len(channel_info)):
            channel_names.append(channel_info[ch].split(': ')[1])

        channel_names.append('X')
        channel_names.append('Y')
        channel_names.append('Z')
        channel_names.append('STI 014')
        channel_names.append('DateTime')

        # Get sampling rates
        sampling_rates = [x for x in content if 'sampling rate: ' in x]
        fs_all = []
        for freq in range(len(sampling_rates)):
            tmp = sampling_rates[freq].split(': ')[1].split(' ')[0]
            if tmp in ['N/A']:
                print('Skipping N/A')
            else:
                fs_all.append(float(sampling_rates[freq].split(': ')[1].split(' ')[0]))

        # Store sampling rates
        fs = fs_all[0]
        fs_accel = fs_all[1]

        # Assign the column names
        df.columns = channel_names
    
    # Return dataframe and sampling rates
    return df, fs, fs_accel

def LoadBehavioralData(filename_behavioral):
    """
    This function loads behavioral data for the motor screening task and formats it to use in this script
    """
    behavioralData = pd.read_csv(filename_behavioral, ',')
    behavioralData = behavioralData.transpose()
    behavioralHeader = behavioralData.iloc[0]
    behavioralData = behavioralData.iloc[2:]
    behavioralData.columns = behavioralHeader
    
    return behavioralData

def SyncTriggerPulses(EEGdata, EEGdevice, fs, behavioralData):
    """
    This function returns the indices for events of interest
    """
    
    if EEGdevice == 7:
        print('Put code here')
    elif EEGdevice == 8:
        # Store where the values in trigger are equal to 8 (the audio trigger input channel number)
        index_trigger = np.where(EEGdata['STI 014']!=0)
        index_trigger = index_trigger[0]
        
        # Check number of trials
        num_of_trials = behavioralData.shape[0]
        if num_of_trials > len(index_trigger):
            num_of_trials = num_of_trials - 1
            num_trials_removed = 1
        else:
            num_trials_removed = 0
        
        trialLength = int(behavioralData['trialLength'][1])

        # Get trial timing
        t_trial_start = list()
        t_trial_end = list()

        # Creating lists of all trigger start and end locations
        for i in range(0,num_of_trials):
            t_trial_start.append(index_trigger[i])
            t_trial_end.append(index_trigger[i] + int(trialLength*fs))

        # Save rest period epochs as well as trials for comparison
        t_rest_start = list()
        t_rest_end = list()

        for i in range(num_of_trials-1):
            t_rest_start.append(t_trial_end[i])
            t_rest_end.append(t_trial_start[i+1])
    
    return num_of_trials, t_trial_start, t_trial_end, t_rest_start, t_rest_end

def EpochData(EEGdata, t_trial_start, t_trial_end):
    """
    This function epochs the data
    """
    
    if EEGdevice == 7:
        channels = EEGdata.columns[1:8]
    elif EEGdevice == 8:
        channels = EEGdata.columns[0:8]
    
    epochs = []
    epochs_norm = []

    for trial in range(0,len(t_trial_start)):
        t_start = t_trial_start[trial]
        t_end = t_trial_end[trial]

        # Baseline
        if trial == 0:
            tb_start = t_trial_start[trial] - np.round(1.5*fs)
            tb_end = t_trial_start[trial]
        else:
            tb_start = t_trial_end[trial-1]
            tb_end = t_trial_start[trial]
            
        baseline = EEGdata.loc[tb_start:tb_end][channels]
        
        # Store epoch
        tmp = (EEGdata.loc[t_start:t_end][channels] - np.mean(baseline))/np.std(baseline)
        epochs_norm.append(tmp)
        epochs.append(EEGdata.loc[t_start:t_end][channels])
    
    return epochs, epochs_norm

def CutEpochs(epochs, fs, trial_type):
    epochs_cut = []
    trial_type_cut = list()

    num_of_epochs = len(epochs)
    full_epoch_length = np.shape(epochs[0])[0]
    cut_epoch_length = int(1.750*fs)

    sliding_window_starts = np.floor(np.linspace(1*fs, full_epoch_length - cut_epoch_length, 10))

    # For each epoch
    for epochOfInt in range(0,num_of_epochs):
        # Reset the index within each epoch temporarily
        reset_index = epochs[epochOfInt].reset_index(drop=True)

        # Sliding window of 750 ms
        for new_start in sliding_window_starts:
            tmp = reset_index.loc[new_start:new_start+cut_epoch_length]
            epochs_cut.append(tmp)
            trial_type_cut.append(trial_type[epochOfInt])
    
    num_of_trials_cut = len(trial_type_cut)
    
    return epochs_cut, trial_type_cut, num_of_trials_cut

def OrganizeTrials(behavioralData):
    """
    Organizes trials
    """
    
    # Create lists for each trial type
    trialL = list()
    trialR = list()
    i = 0

    for letter in behavioralData['trialType']:
        if letter == 'L':
            trialL.append(i)
        elif letter == 'R':
            trialR.append(i)
        i += 1
    
    # Create a single list that includes which trial is which (L = 0, R = 1)
    trial_type = list()
    i = 0

    for letter in behavioralData['trialType']:
        if letter == 'L':
            trial_type.append(0)
        elif letter == 'R':
            trial_type.append(1)
        i += 1
    
    print('np.shape(trial_type): ' + str(np.shape(trial_type)))

    return trial_type, trialL, trialR

def ExtractFeatures(epochs, num_of_trials, channelsToUse, ds_factor):
    """
    Extract signal features of interest
    """
    
    # Get the summed delta power for each trial
    alpha_power = dict.fromkeys(channelsToUse)
    beta_power = dict.fromkeys(channelsToUse)
    ds_f = ds_factor # downsampling factor

    for chanOfInt in channelsToUse:
        tmp_alpha = list()
        tmp_beta = list()

        for trial in range(0, num_of_trials):
            f, Pxx_den = signal.welch(signal.decimate(epochs[trial][chanOfInt],ds_f), fs/ds_f, scaling='spectrum')
            alpha_idx = np.where(np.logical_and(np.round(f) > 8, np.round(f) <= 12))
            tmp_alpha.append(np.sum(Pxx_den[alpha_idx]))

            beta_idx = np.where(np.logical_and(np.round(f) > 13, np.round(f) <= 30))
            tmp_beta.append(np.sum(Pxx_den[beta_idx]))

        alpha_power[chanOfInt] = tmp_alpha
        beta_power[chanOfInt] = tmp_beta
    
    return alpha_power, beta_power

def TrainDecoder(X, y):
    """
    Trains the decoder on ALL the data (does not split into test and train because this is all train)
    """
    # preprocess dataset, split into training and test part
    args = np.arange(len(X))
    np.random.shuffle(args)
    X = [X[i] for i in args]
    y = [y[i] for i in args]
    X_not_scaled = X
    X = StandardScaler().fit_transform(X)

    # Resample to account for imbalance
    method = SMOTE(kind='regular')
    X_balanced, y_balanced = method.fit_sample(X, y)

    # Determine model parameters
    activations = ['relu','tanh']
    alphas = np.logspace(-6, 3, 10)
    solvers = ['lbfgs','sgd']
    hyper_params = {"activation":activations, "alpha":alphas, "solver":solvers}
    grid = GridSearchCV(MLPClassifier(learning_rate='constant', random_state=1), param_grid=hyper_params, cv=KFold(n_splits=5), verbose=True)
    grid.fit(X_balanced, y_balanced)

    # Fit the model
    clf = grid.best_estimator_
    clf.fit(X_balanced,y_balanced)

    """
    # Determine model parameters
    activations = ['relu','tanh']
    alphas = np.logspace(-6, 3, 10)
    solvers = ['lbfgs','sgd']
    hyper_params = {"activation":activations, "alpha":alphas, "solver":solvers}
    grid = GridSearchCV(MLPClassifier(learning_rate='constant', random_state=1), param_grid=hyper_params, cv=KFold(n_splits=5), verbose=True)
    grid.fit(X, y)

    # Fit the model
    clf = grid.best_estimator_
    clf.fit(X,y)
    """
    
    return clf, X, X_not_scaled, y

def SaveDecoderAndData(clf, X, X_not_scaled, y, subjID):
    """
    Save the decoder and the data it was trained/tested on
    """
    time_to_save = datetime.datetime.now().isoformat()
    time_to_save = time_to_save.replace('T','-')
    time_to_save = time_to_save.replace(':','-')
    
    model = clf
    model_file = 'Models/' + subjID + '_MI_classifier_' + time_to_save[:19] + '.sav'
    pickle.dump(model, open(model_file, 'wb'))
    
    filepath_export_data = 'Models/' + subjID + '_data_for_MI_classifier_' + time_to_save[:19] + '.npz'
    np.savez_compressed(filepath_export_data, subjID=subjID, X=X, X_not_scaled=X_not_scaled, y=y)

if __name__ == "__main__":
    print()
    print('-------------------------------------------------------------------')
    print('---If you want to use automatic file selection,                 ---')
    print('---move your EEG data files (.easy, .info) for motor screening  ---')
    print('---into the SaveData folder in this directory!                  ---')
    print('-------------------------------------------------------------------')
    print()
    subjID = input('Enter subject ID: ')
    EEGdevice = int(input('Enter EEG device (7 for DSI-7, 8 for Enobio): ')) # 7 for DSI-7, 8 for Enobio
    if EEGdevice == 7:
        print('DSI-7 selected')
    elif EEGdevice == 8:
        print('Enobio selected')
    else:
        raise ValueError('Invalid EEG device number')
    
    findPaths = input('Manually enter in file paths (y/n)?: ')
    if findPaths in ['y', 'Y', 'yes', 'Yes', 'YES']:
        # Let user manually define file paths
        filename_eeg = input('Enter EEG data file path: ')
        filename_behavioral = input('Enter behavioral data file path: ')
    elif findPaths in ['n', 'N', 'no', 'No', 'NO']:
        # Automatically find the most recent files
        eeg_files = glob.glob('SaveData/*' + subjID + '_Motor*.easy')
        filename_eeg = eeg_files[-1] # load the most recent eeg file
        print(filename_eeg)

        behav_files = glob.glob('SaveData/' + subjID + '_Motor_Screening_*.csv')
        filename_behavioral = behav_files[-1] # load the most recent behavioral file
        print(filename_behavioral)

    # Load EEG data
    EEGdata, fs, fs_accel = LoadEEGData(filename_eeg, EEGdevice)

    # Load behavioral data
    behavioralData = LoadBehavioralData(filename_behavioral)

    # Sync up trigger pulses
    num_of_trials, t_trial_start, t_trial_end, t_rest_start, t_rest_end = SyncTriggerPulses(EEGdata, EEGdevice, fs, behavioralData)

    # Filter the data
    if EEGdevice == 7:
        channels = EEGdata.columns[1:8]
    elif EEGdevice == 8:
        channels = EEGdata.columns[0:8]
    EEGdata_filt = EEGdata.copy()
    eeg_data = EEGdata[channels].values * 1.0 # multiply by 1.0 to convert int to float
    filtered = filter_data(eeg_data.T, sfreq=fs, l_freq=1, h_freq=40)
    EEGdata_filt[channels] = filtered.T

    # Epoch the data
    epochs, epochs_norm = EpochData(EEGdata_filt, t_trial_start, t_trial_end)
    
    # Organize trial types
    trial_type, trialL, trialR = OrganizeTrials(behavioralData)

    # Cut epochs
    epochs_cut, trial_type_cut, num_of_trials_cut = CutEpochs(epochs_norm, fs, trial_type)

    # Get signal features
    alpha_power, beta_power = ExtractFeatures(epochs_cut, num_of_trials_cut, ['C3','C4'], 1)
    motor_features = [alpha_power['C3'], alpha_power['C4'], beta_power['C3'], beta_power['C4']]
    motor_features = np.transpose(motor_features)

    # Train model
    print(np.shape(motor_features))
    print(np.shape(trial_type_cut))
    print(num_of_trials)
    clf, X, X_not_scaled, y = TrainDecoder(motor_features, trial_type_cut)

    # Save decoder and data it was trained/tested on
    SaveDecoderAndData(clf, X, X_not_scaled, y, subjID)