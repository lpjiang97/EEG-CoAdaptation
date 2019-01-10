from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from mne.filter import filter_data

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import scipy.signal as scisig
import numpy as np
import pandas as pd
import csv
import pickle
import datetime

def ConvertEasyToDataFrame(filename):
    """ This function converts a single .easy file (from NIC2) to an easy-to-use dataframe.
    Uses both the .easy file and .info file (containing metadata)
    
    ---- Input ----
    filename: string containing the .easy filepath
    
    ---- Output ----
    df: dataframe containing all the EEG, accelerometer, and event marker data
    fs: sampling rate for the EEG data (Hz)
    fs_accel: sampling rate for the accelerometer data (Hz)
    
    """
    import pandas as pd
    
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
        fs_all.append(float(sampling_rates[freq].split(': ')[1].split(' ')[0]))

    # Store sampling rates
    fs = fs_all[0]
    fs_accel = fs_all[1]
    
    # Assign the column names
    df.columns = channel_names
    
    # Return dataframe and sampling rates
    return df, fs, fs_accel

def PlotAverageSpectrogram(epochs, trials, chanOfInt, fs, ds_factor):
    """
    Input
    epochs: your data epochs (raw)
    trials: list or array of trials you wish to include in the average (e.g., all trials of type A)
    chanOfInt: name of the channel of interest
    fs: sampling rate
    ds_factor: downsampling factor
    """

    Sxx_all = list()

    for trialOfInt in trials:
        f, t, Sxx = scisig.spectrogram(scisig.decimate(epochs[trialOfInt][chanOfInt], ds_factor), fs/ds_factor, nperseg=10)
        # nperseg determines size of time bins, number of time bins = fs/nperseg - 1
        # nperseg = number of segments per second, so 10 = 100 ms chunks
        Sxx_all.append(Sxx)

    Sxx_avg = np.mean(Sxx_all,0)
    
    return f, t, Sxx_avg

if __name__ == "__main__":
    subjID = input('Subject ID: ')
    EEGdevice = int(input('7 if DSI-7, 8 if Enobio: '))
    filepath_EEG = input('EEG data file path (.easy or .csv): ')
    filepath_behavioral = input('Behavioral data filepath (.csv): ')

    ## Load in data from the EEG file
    if EEGdevice == 7:
        EEGmeta = list()
        # Open and read ('r') csv file with raw data
        with open(filepath_EEG, 'r') as csvfile:
            for row in range(0,15):
                EEGmeta.append(csvfile.readline())

        # Save sampling frequency
        fs = int(EEGmeta[1].split(',')[1])

        # Skip the first 15 rows that contain the meta data
        EEGdata = pd.read_csv(filepath_EEG,',',skiprows=15)

    elif EEGdevice == 8:
        EEGdata, fs, fs_accel = ConvertEasyToDataFrame(filepath_EEG)

    else:
       raise ValueError('Invalid EEG device number')

    # Load behavioral data
    behavioralData = pd.read_csv(filepath_behavioral, ',')
    behavioralData = behavioralData.transpose()

    # Saving header information
    behavioralHeader = behavioralData.iloc[0]

    behavioralData = behavioralData.iloc[2:]
    behavioralData.columns = behavioralHeader

    ## Get event markers
    print('Get event markers')
    if EEGdevice == 7:
        # Uses audio pulses
        # Store where the values in trigger are equal to 8 (the audio trigger input channel number)
        index_trigger = np.where(EEGdata['Trigger']==8)

        # Initial peak/jump detection (note that it isn't clean all the time so just take initial)
        index_trigger = np.where(np.diff(EEGdata['Trigger'])>1) #use EEGdata['Trigger'] to make sure we get initial
        
        # Delete suspicious trigger pulses that are closer than 4 seconds to one another
        index_trigger = np.delete(index_trigger, np.where(np.diff(index_trigger[0]) < (4*fs)))

        # Remove initial sync pulse (there were two in this case)
        index_trigger = index_trigger[2:]

        # Check number of trials
        num_of_trials = behavioralData.shape[0]
        channels = EEGdata.columns[1:8]

        # If the number of trials is greater than number of pulses,
        # Likely last trial pulse was not sent, so estimate it
        if np.shape(index_trigger)[0] < num_of_trials:
            index_trigger = np.append(index_trigger, int(index_trigger[-1]+np.mean(np.diff(index_trigger))))

    elif EEGdevice == 8:
        # Use LSL event markers
        index_trigger = np.where(EEGdata['STI 014']!=0)
        index_trigger = index_trigger[0]
        num_of_trials = behavioralData.shape[0]
        channels = EEGdata.columns[0:7]
        
        if num_of_trials > len(index_trigger):
            # Task was run with first trial sending pulse of value 0
            # so remove the first trial since we don't know when it starts
            num_of_trials = num_of_trials - 1
    
    ## Separate into epochs
    print('Separate into epochs')
    trialLength = int(behavioralData['trialLength'][0])

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
    
    ## Cleaning EEG data
    print('Clean EEG data')
    EEGdata_filt = EEGdata.copy()
    eeg_data = EEGdata[channels].values * 1.0 # multiply by 1.0 to convert int to float
    filtered = filter_data(eeg_data.T, sfreq=fs, l_freq=1, h_freq=40)
    EEGdata_filt[channels] = filtered.T

    epochs_raw = []
    epochs_filt = []

    for trial in range(0,len(t_trial_start)):
        epochs_raw.append(EEGdata.loc[t_trial_start[trial]:t_trial_end[trial]][channels])
        epochs_filt.append(EEGdata_filt.loc[t_trial_start[trial]:t_trial_end[trial]][channels])
    
    ## Organize trials
    print('Organize trials')
    # Create lists for each trial type
    trialL = list()
    trialR = list()
    i = 0

    for letter in behavioralData['trialType'][1:]:
        if letter == 'L':
            trialL.append(i)
        elif letter == 'R':
            trialR.append(i)
        elif letter == 'B':
            trialR.append(i)
        elif letter == 'D':
            trialR.append(i)
        i += 1
    
    # Create a single list that includes which trial is which
    trial_type = list()
    i = 0

    for letter in behavioralData['trialType'][1:]:
        if letter == 'L':
            trial_type.append(0)
        elif letter == 'R':
            trial_type.append(1)
        elif letter == 'B':
            trial_type.append(0)
        elif letter == 'D':
            trial_type.append(1)
        i += 1
    
    ## Store the data
    print('Store the data')
    # Store spectrogram data for each trial, for each channel
    ds_factor = 4 # downsampling factor
    channels_to_use = ['C3', 'C4'] # to use all channels, use channels
    epochs_Sxx = [[]]*len(channels_to_use)
    i = 0

    for chanOfInt in channels_to_use:
        tmp = list()
        for trial in range(0,len(trial_type)):
            f, t, Sxx = scisig.spectrogram(scisig.decimate(epochs_raw[trial][chanOfInt].T, ds_factor), fs/ds_factor, nperseg=10)
            tmp.append(Sxx.flatten('C')) # if you want all frequency data
            #tmp.append(Sxx[1]) # if you just want second frequency bin (10-20 Hz if you do ds_factor = 3)
        epochs_Sxx[i] = tmp
        i += 1

    # Format for classification
    print('Format data for classification')
    epochs_collapsed = []

    # Concatenate all epochs such that shape is (num of trials x (flattened channels and spec data))
    # So this should be 100 x 2310 (which is 7 channels * 330 spectrogram data points)
    num_of_channels = np.shape(epochs_Sxx)[0]
    num_of_trials = np.shape(epochs_Sxx)[1]
    num_of_specData = np.shape(epochs_Sxx)[2]

    epochs_collapsed = np.asarray(epochs_Sxx).reshape(num_of_trials, num_of_channels * num_of_specData)

    h = .02  # step size in the mesh

    # preprocess dataset, split into training and test part
    print('Preprocess data for classification')
    X = [np.mean(epochs_collapsed,1),np.std(epochs_collapsed,1)]
    X = np.asarray(X).T
    y = trial_type
    args = np.arange(len(X))
    np.random.shuffle(args)
    X = [X[i] for i in args]
    y = [y[i] for i in args]
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))


    ## Classify
    print('Classify')
    name = "Random Forest"
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('Score: ' + str(score))

    # Save model
    print('Save model')
    model = clf
    time_to_save = datetime.datetime.now().isoformat()
    time_to_save = time_to_save.replace('T','-')
    time_to_save = time_to_save.replace(':','-')
    model_file = subjID + '_MI_classifier_' + time_to_save[:19] + '.sav'
    pickle.dump(model, open(model_file, 'wb'))

    # Save training/testing examples
    print('Save training/testing examples')
    filepath_export_data = 'data_for_training_' + time_to_save[:19] + '.npz'
    np.savez_compressed(filepath_export_data, X=X, y=y)