"""
This is an alternate version of Motor_Screening.py where the stimulus is
a white box moving instead of a static arrow.

Subjects are asked to imagine that they are moving the box they see
with the hand associated with the direction (box moving left = left hand).
"""

import sys
from sys import platform
from psychopy import visual, core, event
from random import shuffle, uniform
from datetime import datetime
from pylsl import StreamInfo, StreamOutlet
import pandas as pd
import numpy as np
import glob
import time

USE_SOUND = platform == "win32"
if USE_SOUND: from psychopy import sound

trialCnt = 0

def increment():
    global trialCnt
    trialCnt += 1

def play_sound(value):
    if USE_SOUND:
        sound.init(rate=44100, stereo=True, buffer=128)
        sync_beep = sound.Sound(value=value, secs=0.2, octave=4, loops=0)
        sync_beep.play()
        return sync_beep
    else:
        sys.stdout.write('\a')
        sys.stdout.flush()
        return None

def connectEEG(EEGdevice, outlet):
    if EEGdevice == 7:
        # Send audio sync pulse to DSI-streamer
        play_sound('C')  
    else:
        # Send pulse through LSL
        outlet.push_sample([999])
    
    # Initialize the clock once connection/recording is established (tic)
    clock = core.Clock()
    initial_timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    return (clock, initial_timestamp)

def runBlock(subjID, trialType, taskData, win, clock, EEGdevice, outlet):
    # Run trial and ISI (block notation from DMN task code) 
    runTrial(subjID, trialType, taskData, win, clock, 'F', EEGdevice, outlet)
    ISIlength = uniform(1,2) # random value generated here
    runISI(ISIlength, win)

def runTrial(subjID, trialType, taskData, win, clock, tone, EEGdevice, outlet):
    posx = 0
    trialLength = 4
    time_trialStart = time.time()
    while time.time() - time_trialStart <= trialLength:
        rect = visual.Rect(
                win=win,
                units='pix',
                width=100,
                height=100,
                fillColor=[1,1,1],
            )
        rect.pos = [posx,0]
        # Determine what to display
        if trialType == 'L':
            stimulus = 'left'
        elif trialType == 'R':
            stimulus = 'right'

        #message.setText(stimulus)
        rect.draw()
        win.flip()

        # Fill in the dataframe
        if posx == 0:
            if EEGdevice == 7:
                beep = play_sound(tone)
                trialStart = clock.getTime()
                if beep is not None:
                    beep.play()
                else:
                    play_sound(tone)
            else:
                outlet.push_sample([trialCnt])
                trialStart = clock.getTime()
        
        # Determine what to display
        if trialType == 'L':
            posx = posx - 100
        elif trialType == 'R':
            posx = posx + 100

        # Allow an early termination
        allKeys = event.getKeys()
        for thisKey in allKeys:
            if thisKey in ['q', 'escape']:
                saveData(subjID, taskData, win)
        event.clearEvents()
        
        core.wait(1)

    # Save the task data
    taskData['trial_' + str(trialCnt)] = pd.Series([trialStart, stimulus, trialLength, trialType], \
    index=['trialStart','stimulus','trialLength','trialType'])

    increment() # increase trial counter

def runISI(ISIlength, win):
    message.setText('+')
    win.flip()
    core.wait(ISIlength)
    message.setText(' ')
    win.flip()

def saveData(subjID, taskData, win):
    # Check how many files are currently present for this subject
    fileNum = len(glob.glob('SaveData/' + subjID + '_Motor_Screening*.csv')) + 1

    # Save the data to csv file
    file_name = ('SaveData/' + subjID + '_Motor_Screening_R' + str(fileNum) + '.csv')
    taskData.to_csv(file_name, encoding='utf-8')

    # Save the data as a json file
    taskData.to_json('SaveData/' + subjID + '_Motor_Screening_R' + str(fileNum) + '.json')

    # Close the window
    win.close()
    core.quit()

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

    if EEGdevice == 7:
        print('DSI-7 selected')
        outlet = []
    elif EEGdevice == 8:
        print('Enobio selected')
        # Make sure names are set in NIC2 settings
        # Outlet for Lab Streaming Layer: LSLoutlet
        # Markers Lab Streaming Layer 1: LSLmarkers1
        info = StreamInfo('LSLmarkers1','Markers',1,0,'int32','LSLoutlet')
        outlet = StreamOutlet(info)
    else:
        raise ValueError('Invalid EEG device number')

    # Ensure EEG is connected
    [clock, initial_timestamp] = connectEEG(EEGdevice, outlet)

    # Create the window to display
    win = visual.Window(color=[-1,-1,-1], fullscr=True)

    # Initialize message here
    message = visual.TextStim(win, text='Watch the box move and imagine you are moving it')
    message.setAutoDraw(True)  # automatically draw every frame
    win.flip()
    core.wait(4)
    message.setText('If it moves left, imagine you are moving it with your left hand')
    win.flip()
    core.wait(4)
    message.setText('If it moves right, imagine you are moving it with your right hand')
    win.flip()
    core.wait(3)
    message.setText('Press any key to continue...')
    win.flip()
    event.waitKeys()
    message.setText(' ')
    win.flip()

    # Create pandas dataframe to store behavioural data
    d = {'initialize' : pd.Series([initial_timestamp, 0, 0, 'X'], \
    index=['trialStart','stimulus', 'trialLength', 'trialType'])}
    taskData = pd.DataFrame(d)

    # Loop to shuffle and run blocks
    """
    Trial Type L: Imagine moving your left hand
    Trial Type R: Imagine moving your right hand
    """
    trialType = ['L','R']
    numOfTrials = 100

    trialOrder = np.matlib.repmat(trialType, 1, int(numOfTrials/len(trialType)))
    shuffle(trialOrder[0])
    for currentTrial in trialOrder[0]:
        runBlock(subjID, currentTrial, taskData, win, clock, EEGdevice, outlet)

    saveData(subjID, taskData, win)