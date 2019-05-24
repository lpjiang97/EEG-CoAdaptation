""" 
One dimensional (1D) center-out Brain-Computer Interface (BCI) task

In this task, subjects imagine moving their left hand to move the cursor left,
and imagine moving their right hand to move the cursor right.

In each trial, the subject attemps to move the cursor into the target by
appropriately moving the cursor either left or right towards it.

The cursor will move once per second, using EEG data from the preceeding X ms.

Once the cursor hits the target, the trial will end. If, however, the subject
is not able to get the cursor into the target before X seconds, the trial will
end automatically.

Each trial is separated by an Inter-trial Interval (ITI) randomly lasting between
1.75 to 2.25 seconds.

WARNING: This code assumes C3 is channel X and C4 is channel in the DSI-7,
and assumes that C3 is channel X and C4 is channel X in the Enobio!!
 (change description in intro print out once you determine this)
  (and change this in game class setup)


TO ADD:
+ keyboard close out
"""

from CCDLUtil.EEGInterface.DSI.DSIInterface import DSIStreamer
import sys
from sys import platform
from psychopy import visual, core, event
from sklearn.preprocessing import StandardScaler
from random import shuffle, uniform, randrange, random
from mne.filter import filter_data
from datetime import datetime
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
from PIL import Image
from scipy import signal
import scipy.signal as scisig
import json
import numpy as np
import pandas as pd
import pickle
import glob
import time
import arcade

USE_SOUND = platform == "win32"
if USE_SOUND: from psychopy import sound

moveCount = 0
def increment():
    global moveCount
    moveCount += 1

# Row and column count should be an odd number so there's a center
ROW_COUNT = 5
COLUMN_COUNT = 17
WIDTH = 75
HEIGHT = WIDTH
MARGIN = 1

SCREEN_WIDTH = (WIDTH + MARGIN) * COLUMN_COUNT + MARGIN
SCREEN_HEIGHT = (HEIGHT + MARGIN) * ROW_COUNT + MARGIN

def butter_bandpass(low, high, fs, order=5):
    """
    Wrapper function for the scipy butter
    :param low: Frequency to filter above
    :param high: Frequency to filter below
    :param fs: Sampling rate (hz)
    :param order: Order of filter to use (default = 5)
    :return: Numerator (b) and denominator (a) polynomials of the IIR filter
    """
    nyq = 0.5 * fs
    b, a = scisig.butter(order, [low / nyq, high / nyq], btype='band')
    return b, a

def butter_bandpass_filter(data, low, high, fs, order=5):
    """
    Filters passed data with a bandpass butter function
    :param data: data to be bandpass filtered
    :param low: Frequency to filter above
    :param high: Frequency to filter below
    :param fs: Sampling rate (hz)
    :param order: Order of filter to use (default = 5)
    :return: filtered data (and modifies original data).
    """
    b, a = butter_bandpass(low, high, fs, order=order)
    data = data - np.mean(data)
    return scisig.lfilter(b, a, data)

def play_sound(value, sec=0.2, octave=4):
    if USE_SOUND:
        sound.init(rate=44100, stereo=True, buffer=128)
        sync_beep = sound.Sound(value=value, secs=sec, octave=octave, loops=0)
        sync_beep.play()
        return sync_beep
    else:
        sys.stdout.write('\\' + value)
        sys.stdout.flush()
        return None

def connectEEG(EEGdevice, outlet):
    if EEGdevice == 7:
        # Send audio sync pulse to DSI-streamer
        play_sound('C')  
    elif EEGdevice == 8:
        # Send pulse through LSL
        outlet.push_sample([111])
    
    # Initialize the clock once connection/recording is established (tic)
    clock = core.Clock()
    initial_timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    return (clock, initial_timestamp)

def loadModel(subjID):
    """
    Loads the most recent version of the MI classifier
    """
    models = glob.glob('Models/' + subjID + '_MI_classifier_*')
    model_file = models[-1] # load the most recent model
    loaded_model = pickle.load(open(model_file, 'rb'))

    # Load data associated with that model for standardization purposes
    data_files = glob.glob('Models/' + subjID + '_data_for_MI_classifier_*')
    data_file = data_files[-1] # load the most recent model
    loaded_data = np.load(data_file)
    X_loaded = loaded_data['X_not_scaled']

    return loaded_model, X_loaded

class Player(arcade.Sprite):
    """ Player sprite """

    def update(self):
        self.center_x += self.change_x
        self.center_y += self.change_y

        if self.left < 0:
            self.left = 0 + MARGIN
        elif self.right > SCREEN_WIDTH - 1:
            self.right = SCREEN_WIDTH - MARGIN

        if self.bottom < 0:
            self.bottom = 0 + MARGIN
        elif self.top > SCREEN_HEIGHT - 1:
            self.top = SCREEN_HEIGHT - MARGIN

class MyGame(arcade.Window):
    """ Main application class. """

    def __init__(self, width, height, taskData, col_names, EEGdevice, outlet, streamer, alpha_baseline_open, alpha_baseline_closed, oz_channel=0):
        # Call the parent class initializer
        super().__init__(width, height)

        # Variables for sync with LSL
        self.streamer = streamer
        self.alpha_open = alpha_baseline_open
        self.alpha_closed = alpha_baseline_closed
        self.oz_channel = oz_channel
        self.EEGdevice = EEGdevice
        self.outlet = outlet

        if self.EEGdevice == 7:
            self.fs = 300
            self.channels = 7
        elif self.EEGdevice == 8:
            self.fs = 500
            self.channels = 8
        
        self.nyq = 0.5 * self.fs


        # Store behavioral data
        self.taskData = taskData
        self.taskData_col_names = col_names

        # Variables that will hold sprite lists
        self.player_list = None
        self.target_list = None
        self.ITI_list = None

        # Set up the player info
        self.player_sprite = None
        self.target_sprite = None
        self.score = 0

        # Track the current state of what key is pressed
        self.move_left = False
        self.move_right = False
        
        # Determine when to move and when to end a trial
        self.trialTime = time.time()
        self.trialLength = 20
        self.firstTrial = True

        # Determine when to end the task
        self.trialNum = 0
        self.trialType = [0,0,0,0,0,0,1,1,1,1,1,1] #0 for left, 1 for right, 12 trials total
        np.random.shuffle(self.trialType)

        # Keep of track of if you're in an ISI
        self.inISI = False

        # Set the background color
        arcade.set_background_color(arcade.color.BLACK)
    
    def create_target(self):
        # Close task if end of trials
        print('trialNum: ' + str(self.trialNum))
        print('len(trialType): ' + str(len(self.trialType)))
        if self.trialNum >= len(self.trialType):
            arcade.close_window()
        else:
            # Create the target
            im = Image.open('Images/target.png')
            img_width, img_height = im.size
            scaling_factor = min(WIDTH/img_width, HEIGHT/img_height)
            self.target_sprite = arcade.Sprite('Images/target.png', scale = scaling_factor)

            # Place to either the left or right
            if self.trialType[self.trialNum] > 0.5:
                # Target on the right
                self.target_sprite.center_x = (WIDTH + MARGIN) * (COLUMN_COUNT-1) - WIDTH/2 - MARGIN
                self.target_sprite.center_y = (HEIGHT + MARGIN) * (ROW_COUNT)/2 + MARGIN
            else:
                # Target on the left
                self.target_sprite.center_x = (WIDTH + MARGIN) * (1) + WIDTH/2 + MARGIN
                self.target_sprite.center_y = (HEIGHT + MARGIN) * (ROW_COUNT)/2 + MARGIN

            self.target_list.append(self.target_sprite)
        self.trialNum += 1

    def setup(self):
        """ Set up the game and initialize variables. """
        # Create the sprite list
        self.player_list = arcade.SpriteList()
        self.target_list = arcade.SpriteList()
        self.ITI_list = arcade.SpriteList()

        # Set up the player
        self.score = 0
        im = Image.open('Images/character.png')
        img_width, img_height = im.size
        scaling_factor = min(WIDTH/img_width, HEIGHT/img_height)
        self.player_sprite = Player('Images/character.png', scale = scaling_factor)
        self.player_sprite.center_x = (WIDTH + MARGIN) * (COLUMN_COUNT/2) + MARGIN/2
        self.player_sprite.center_y = (HEIGHT + MARGIN) * (ROW_COUNT/2) + MARGIN/2
        self.player_list.append(self.player_sprite)

        # Create the first target
        self.create_target()

        # Create placeholder for baseline voltage values
        self.baseline = []
        self.eeg_data = []
    
    def on_draw(self):
        """ Render the screen. """
        # This command has to happen before we start drawing
        arcade.start_render()

        # Draw all the sprites
        self.player_list.draw()
        self.target_list.draw()

        # Score text
        output = f"Score: {self.score}"
        arcade.draw_text(output, 10, 20, arcade.color.WHITE, 14)


    def classify_epoch(self):
        """
        Call this function in update to:
        1. Normalize the just-read-in data to previous ITI (baseline)
        2. Convert into usable features
        3. Feed into our model to get prediction
        4. Return direction prediction (use self.move_left, etc)
        """
        # Initialize our feature vector
        X = [0, 0, 0, 0]

        # Filter our data
        filtered = filter_data(self.eeg_data.T, sfreq=self.fs, l_freq=7, h_freq=31, verbose='ERROR')
        data = filtered.T
        
        """
        # If baseline is empty, just normalize to self (first trial)
        if len(self.baseline) > 0:
            tmp = (data - np.mean(self.baseline))/(np.std(self.baseline))
        else:
            tmp = (data - np.mean(data))/(np.std(data))
        """

        # Get alpha and beta power for C3 and C4
        # channels = [self.C3, self.C4]
        # cnt = 0
        # for chanOfInt in channels:
        #     ds_f = 1 # downsample factor
        #     f, Pxx_den = signal.welch(signal.decimate(tmp[:,chanOfInt], ds_f), self.fs/ds_f, scaling='spectrum')
        #     alpha_idx = np.where(np.logical_and(np.round(f) >= 8, np.round(f) <= 12))
        #     X[cnt] = np.sum(Pxx_den[alpha_idx])
        #     cnt += 1

        #     beta_idx = np.where(np.logical_and(np.round(f) >= 13, np.round(f) <= 30))
        #     X[cnt] = np.sum(Pxx_den[beta_idx])
        #     cnt += 1

        # get alpha power from live EEG
        Oz = data[:, self.oz_channel]
        Oz_filter = filter_data(Oz, self.fs, 8, 12, verbose='ERROR')
        power = (np.abs(scisig.hilbert(Oz_filter)) ** 2).mean()
        print('Power: ' + str(power))

        # Take care of any NaN from loose electrode (a!=a means NaN)
        for i in range(0,len(X)):
            if X[i] != X[i]:
                X[i] = 0
                print('WARNING: loose channel, setting feature value to 0')

        # Feed into decoder (0 = left, 1 = right)
        # Scale the features, and reshape for single example
        # print('X: ' + str(X))
        # X = np.asarray(X)
        # print('X (scaled): ' + str(X))

        open_diff = np.abs(power-self.alpha_open)
        closed_diff = np.abs(power-self.alpha_closed)
        if open_diff < closed_diff:
            y = 0
        else:
            y = 1

        #y = 0 if power < self.alpha else 1 # self.clf.predict([X[0]])

        print('y: ' + str(y))
        print('-----------------------------------------------------')

        if y == 0:
            self.move_left = True
            self.move_right = False
        elif y == 1:
            self.move_left = False
            self.move_right = True

    
    def update(self, delta_time):
        """ All the logic to move, and the game logic goes here. """

        # Start task with rest trial
        if self.firstTrial is True:
            self.display_rest()
            self.firstTrial = False

        # Calculate where to move the player sprite
        self.player_sprite.change_x = 0
        self.player_sprite.change_y = 0

        
        # Buffer data for 750 ms and process to determine which direction to go
        step_time = time.time() # want to take 1 second for step to update
        step_length = 1.750 # amount of time we want to use for calculation
        buffer = int(self.fs * step_length)
        data = np.zeros((buffer, self.channels))

        if self.EEGdevice == 7:
            self.streamer.clear_out_buffer()

        for idx in range(buffer):
            if self.EEGdevice == 7:
                # code to read in EEG from DSI-7 here
                tmp = 1 # just put something here to prevent indentation issues
                data[idx, :] = self.streamer.out_buffer_queue.get()[:-1]
            elif self.EEGdevice == 8:
                sample, timestamp = inlet.pull_sample()
                data[idx, :] = sample
        
        self.eeg_data = data
        self.classify_epoch() # TO DO
        
        # Move the player sprite
        while time.time() - step_time < 2:
            tmp = 1 # just do something until time is up

        pulseValue = 0

        if self.move_left is True:
            # Left (right hemi power high = left hand)
            self.player_sprite.set_position(
                self.player_sprite.center_x - (WIDTH + MARGIN),
                self.player_sprite.center_y
            )
            direction_moved = 'left'
            pulseValue = 1
        elif self.move_right is True:
            # Right
            self.player_sprite.set_position(
                self.player_sprite.center_x + WIDTH + MARGIN,
                self.player_sprite.center_y
            )
            direction_moved = 'right'
            pulseValue = 2
    
        # Call update on all sprites
        self.player_list.update()
        self.target_list.update()
        self.ITI_list.update()

        # Send pulses
        move_time = clock.getTime()
        if self.EEGdevice == 7:
        # Audio beep to send pulse to sync box
            if self.move_left is True:
                play_sound('C')
            elif self.move_right is True:
                play_sound('E')
        elif self.EEGdevice == 8:
            # Send pulse through LSL, 1 if left, 2 if right
            self.outlet.push_sample([pulseValue])
            if self.move_left is True:
                play_sound('C')
            elif self.move_right is True:
                play_sound('E')

        # Store task data
        taskData.loc[moveCount] = pd.Series([move_time, \
                    self.player_sprite.center_x, self.player_sprite.center_y, \
                    self.target_sprite.center_x, self.target_sprite.center_y, \
                    direction_moved, self.score], \
                    index=self.taskData_col_names)
        increment()

        # End trial if time is up
        if time.time() - self.trialTime >= self.trialLength:
            arcade.start_render()
            arcade.draw_rectangle_filled(center_x = self.player_sprite.center_x, center_y = self.player_sprite.center_y, width = WIDTH, height = HEIGHT, color = arcade.color.RED)
            arcade.finish_render()
            arcade.pause(1)
            # Play minor arpeggio downward
            play_sound(value='A', sec=0.1, octave=4)
            arcade.pause(0.1)
            play_sound(value='E', sec=0.1, octave=4)
            arcade.pause(0.1)
            play_sound(value='C', sec=0.1, octave=4)
            arcade.pause(0.1)
            play_sound(value='A', sec=0.1, octave=3)

            for target in self.target_list:
                target.kill()
            self.display_rest()
            self.create_target()

        # Generate list of all targets that collided with the player
        target_hit_list = arcade.check_for_collision_with_list(self.player_sprite, self.target_list)

        # Change color of cursor on top of target
        hitTarget = False
        for target in target_hit_list:
            hitTarget = True
        
        if hitTarget is True:
            arcade.start_render()
            arcade.draw_rectangle_filled(center_x = self.player_sprite.center_x, center_y = self.player_sprite.center_y, width = WIDTH, height = HEIGHT, color = arcade.color.GREEN)
            arcade.finish_render()
            arcade.pause(1)
            # Play major arpeggio upward
            play_sound(value='C', sec=0.1, octave=4)
            arcade.pause(0.1)
            play_sound(value='E', sec=0.1, octave=4)
            arcade.pause(0.1)
            play_sound(value='G', sec=0.1, octave=4)
            arcade.pause(0.1)
            play_sound(value='C', sec=0.1, octave=5)

        # Loop through each colliding sprite, removing it, and adding to the score
        for target in target_hit_list:
            target.kill()
            self.display_rest()
            self.score += 1
            self.create_target()
    
    def display_rest(self):
        self.inISI = True

        """ Displays a fixation cross for a predetermined amount of time for rest (inter-trial interval; ITI)"""
        center_x = (WIDTH + MARGIN) * (COLUMN_COUNT/2) + MARGIN/2
        center_y = (HEIGHT + MARGIN) * (ROW_COUNT/2) + MARGIN/2

        self.ITI_item = arcade.Sprite('Images/empty.png', image_height = SCREEN_HEIGHT, image_width = SCREEN_WIDTH, center_x = center_x, center_y = center_y)
        self.ITI_list.append(self.ITI_item)
        self.ITI_item = arcade.Sprite('Images/cross.png', center_x = center_x, center_y = center_y)
        self.ITI_list.append(self.ITI_item)

        # Cover whole window with a black box and add white fixation cross
        arcade.start_render()
        self.ITI_list.draw()
        arcade.finish_render()

        # Collect baseline data while waiting
        if self.EEGdevice == 7:
            # Audio beep to send pulse to sync box
            print('Need to send that rest is starting')
        elif self.EEGdevice == 8:
            # Send pulse through LSL, 1 if left, 2 if right
            pulseValue = 3 # indicate start of baseline
            self.outlet.push_sample([pulseValue])

        # Wait x number of seconds, then destroy the box and fixation cross
        rest_time = uniform(1.75,2.25)
        rest_start = time.time()

        step_length = rest_time # amount of time we want to use for calculation
        buffer = int(self.fs * step_length)

        data = np.zeros((buffer, self.channels))
        for idx in range(buffer):
            if self.EEGdevice == 7:
                # code to read in EEG from DSI-7 here
                tmp = 1 # just put something here to prevent indentation issues
            elif self.EEGdevice == 8:
                sample, timestamp = inlet.pull_sample()
                data[idx, :] = sample
        
        filtered = filter_data(data.T, sfreq=self.fs, l_freq=7, h_freq=31)
        self.baseline = filtered.T

        while time.time() - rest_start < rest_time:
            tmp = 1 # just do something until time is up


        #arcade.pause(rest_time)
        for item in self.ITI_list:
            item.kill()
        
        self.inISI = False

        # Reset cursor to center
        self.player_sprite.center_x = center_x
        self.player_sprite.center_y = center_y
        self.player_list.append(self.player_sprite)

        # Reset trial time-out timer
        self.trialTime = time.time()

        # Send pulse to signify end of rest period
        if self.EEGdevice == 7:
            # Audio beep to send pulse to sync box
            print('Need to send that rest is ending')
        elif self.EEGdevice == 8:
            # Send pulse through LSL, 1 if left, 2 if right
            pulseValue = 4 # indicate the end of baseline
            self.outlet.push_sample([pulseValue])
    
    def on_key_press(self, key, modifiers):
        if key == arcade.key.Q:
            arcade.close_window()
        if key == arcade.key.ESCAPE:
            arcade.close_window()
        

def saveData(subjID, taskParameters, taskData):
    # Check how many files are currently present for this subject
    fileNum = len(glob.glob('SaveData/alpha_BCI_' + subjID + '*.csv')) + 1

    # Save the task parameters to json file
    with open('SaveData/alpha_BCI_' + subjID + '_R' + str(fileNum) + '_Parameters.json', 'w') as fp:
        json.dump(taskParameters, fp)

    # Save the data to csv file
    file_name = ('SaveData/alpha_BCI_' + subjID + '_R' + str(fileNum) + '.csv')
    taskData.to_csv(file_name, encoding='utf-8')

    # Save the data as a json file
    taskData.to_json('SaveData/alpha_BCI_' + subjID + '_R' + str(fileNum) + '.json')


def get_baseline_alpha(streamer, EEGdevice=7, oz_channel=4, seconds=20, fs=300, verbose=True):
    '''Calculating alpha baseline with eyes open 

    args:
        - streamer: DSI/Enobio Streamer object 
        - buffer_length: the number of data points used for hilbert power 
        - seconds: base 
    '''
    if EEGdevice == 7:
        num_channels = 7
        # clear buffer
        streamer.clear_out_buffer()
        # save the power
        powers = []
        # start collecting for eyes open
        if verbose:
            print('Keep your eyes open!')
            print("Start calculating baseline...")
        data = np.zeros((seconds//2 * fs, num_channels))  
        for i in range(seconds//2 * fs):
            data[i] = streamer.out_buffer_queue.get()[:-1]
        # get Oz
        print(np.shape(data))
        Oz = data[:, oz_channel]
        Oz_filter = filter_data(Oz, fs, 8, 12, verbose='ERROR')
        power_open = (np.abs(scisig.hilbert(Oz_filter)) ** 2).mean()
        if verbose:
            print("Baseline calculation finished!")
            print('Baseline (eyes open): ' + str(power_open))

        # start collecting for eyes closed
        if verbose:
            print('Keep your eyes closed!')
            print("Start calculating baseline...")
        data = np.zeros((seconds//2 * fs, num_channels))  
        for i in range(seconds//2 * fs):
            data[i] = streamer.out_buffer_queue.get()[:-1]
        # get Oz
        Oz = data[:, oz_channel]
        Oz_filter = filter_data(Oz, fs, 8, 12, verbose='ERROR')
        power_closed = (np.abs(scisig.hilbert(Oz_filter)) ** 2).mean()
        if verbose:
            print("Baseline calculation finished!")
            print('Baseline (eyes closed): ' + str(power_closed))
        return power_open, power_closed
    elif EEGdevice == 8:
        num_channels = 8
        # TO DO, MAKE THIS WORK
        print('Just temporarily setting threshold values now')
        power_open = 10
        power_closed = 100
        return power_open, power_closed
    


if __name__ == "__main__":
    # Ask for subjectID
    print()
    print('-------------------------------------------------------------------')
    print('---Make sure volume is up and audio trigger cable is plugged in!---')
    print('-------------------------------------------------------------------')
    print()
    # subjID = input('Enter subject ID: ')
    subjID = input("Enter subject ID: ")

    # Ask for which system we are using
    EEGdevice = int(input('Enter EEG device (7 = DSI-7, 8 = Enobio): '))
    streamer = None

    if EEGdevice == 7:
        print('DSI-7 selected')
        oz_channel = 4 # actually PO7
        outlet = []
        print('starting to connect to DSI streamer')
        streamer = DSIStreamer(live=True, save_data=True)
        streamer.start_recording()
        streamer.start_saving_data("testing_bci_1d.csv")
        print('connected')
    elif EEGdevice == 8:
        print('Enobio selected')
        oz_channel = int(input('Enter channel number for Oz (first channel is channel 0): '))
        # Make sure names are set in NIC2 settings
        # Outlet for Lab Streaming Layer: LSLoutlet
        # Markers Lab Streaming Layer 1: LSLmarkers1
        info = StreamInfo('LSLmarkers1','Markers',1,0,'int32','LSLoutlet')
        outlet = StreamOutlet(info)

        # Connect to read EEG
        stream_name = 'LSLoutlet-EEG'
        streams = resolve_stream('type', 'EEG')

        try:
            for i in range (len(streams)):

                print(streams[i].name())

                if (streams[i].name() == stream_name):
                    index = i
                    print ("NIC stream available")

            print ("Connecting to NIC stream... \n")
            inlet = StreamInlet(streams[index])   

        except NameError:
            print ("Error: NIC stream not available\n\n\n")
    else:
        raise ValueError('Invalid EEG device number')

    # Ensure EEG is connected
    [clock, initial_timestamp] = connectEEG(EEGdevice, outlet)

    # Load our classifier
    # clf, X_loaded = loadModel(subjID)

    # Save task parameters
    taskParameters = {'initial_timestamp':initial_timestamp, \
        'column_count':COLUMN_COUNT,'row_count':ROW_COUNT,'width':WIDTH,\
        'height':HEIGHT,'margin':MARGIN,'subjID':subjID}

    # Create pandas dataframe to store behavioural data
    col_names = ['time', 'player_x', 'player_y', 'target_x', 'target_y', 'direction_moved','score']
    taskData = pd.DataFrame(columns = col_names)

    # before game, calculate base line
    print('starting to get baseline alpha???')
    alpha_baseline_open, alpha_baseline_closed = get_baseline_alpha(streamer, EEGdevice, oz_channel, seconds=20)
    #alpha_baseline = np.mean(alpha_baseline_open, alpha_baseline_closed)

    # Create the window to display
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, taskData, col_names, EEGdevice, outlet, streamer, alpha_baseline_open, alpha_baseline_closed, oz_channel)
    game.setup()
    arcade.run()

    # Once window is closed
    saveData(subjID, taskParameters, taskData)

    # Send end of task pulse when closing
    if EEGdevice == 7:
        # Send audio sync pulse to DSI-streamer
        print('End of task')  
        streamer.stop_recording()
    elif EEGdevice == 8:
        # Send pulse through LSL
        outlet.push_sample([999])
