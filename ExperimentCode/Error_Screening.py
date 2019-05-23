""" 
This Error Screening script is just a simple 1D center-out task where subjects
need to press the left or right arrow key 3 times consecutively to move it one
cell over.

Occasionally, the cursor (what they are controlling) will move in the direction
opposite to what they pressed.

They have 7 seconds to reach the target from the start of each trial.
"""

import sys
from sys import platform
from psychopy import visual, core, event
from random import shuffle, uniform, randrange, random
from datetime import datetime
from pylsl import StreamInfo, StreamOutlet
from PIL import Image
import json
import pandas as pd
import glob
import time
import arcade

USE_SOUND = platform == "win32"
if USE_SOUND: from psychopy import sound

# Set error rate
ERROR_RATE = 0.20

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

    def __init__(self, width, height, taskData, col_names, EEGdevice, outlet):
        # Call the parent class initializer
        super().__init__(width, height)

        # Variables for sync with LSL
        self.EEGdevice = EEGdevice
        self.outlet = outlet

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
        self.left_pressed = False
        self.right_pressed = False
        self.can_move = True

        # Determine when to move and when to end a trial
        self.pressCountToMove = 3
        self.trialTime = time.time()
        self.trialLength = 7

        # Keep of track of if you're in an ISI
        self.inISI = False
        self.trialPressCount = 0

        # Set the background color
        arcade.set_background_color(arcade.color.BLACK)
    
    def create_target(self):
        # Create the target
        im = Image.open('Images/target.png')
        img_width, img_height = im.size
        scaling_factor = min(WIDTH/img_width, HEIGHT/img_height)
        self.target_sprite = arcade.Sprite('Images/target.png', scale = scaling_factor)

        # Place to either the left or right
        if random() > 0.5:
            # Target on the right
            self.target_sprite.center_x = (WIDTH + MARGIN) * (COLUMN_COUNT-1) - WIDTH/2 - MARGIN
            self.target_sprite.center_y = (HEIGHT + MARGIN) * (ROW_COUNT)/2 + MARGIN
        else:
            # Target on the left
            self.target_sprite.center_x = (WIDTH + MARGIN) * (1) + WIDTH/2 + MARGIN
            self.target_sprite.center_y = (HEIGHT + MARGIN) * (ROW_COUNT)/2 + MARGIN

        self.target_list.append(self.target_sprite)

    def setup(self):
        """ Set up the game and initialize variables. """
        # Create random variable to determine forced error
        self.reverseDirection = False

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
    
    def update(self, delta_time):
        """ All the logic to move, and the game logic goes here. """

        # Calculate where to move the player sprite
        self.player_sprite.change_x = 0
        self.player_sprite.change_y = 0

        if self.can_move is True:
            if self.left_pressed and not self.right_pressed:
                self.player_sprite.set_position(
                    self.player_sprite.center_x - (WIDTH + MARGIN),
                    self.player_sprite.center_y
                )
                self.can_move = False
            elif self.right_pressed and not self.left_pressed:
                self.player_sprite.set_position(
                    self.player_sprite.center_x + WIDTH + MARGIN,
                    self.player_sprite.center_y
                )
                self.can_move = False

        # Call update on all sprites
        self.player_list.update()
        self.target_list.update()
        self.ITI_list.update()

        # End trial if time is up
        if time.time() - self.trialTime >= self.trialLength:
            arcade.start_render()
            arcade.draw_rectangle_filled(center_x = self.player_sprite.center_x, center_y = self.player_sprite.center_y, width = WIDTH, height = HEIGHT, color = arcade.color.RED)
            arcade.finish_render()
            arcade.pause(1)

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

        # Loop through each colliding sprite, removing it, and adding to the score
        for target in target_hit_list:
            target.kill()
            self.display_rest()
            self.score += 1
            self.create_target()

    def on_key_press(self, key, modifiers):

        if self.inISI is False:
            """ Called whenever a left or right key is pressed. """
            if key == arcade.key.LEFT or key == arcade.key.RIGHT:
                move_time = clock.getTime()
                self.trialPressCount += 1

                # Move the player
                if self.reverseDirection is False:
                    # Everything functions normally
                    if key == arcade.key.LEFT:
                        self.left_pressed = True
                        pressed_key = 'left'
                    elif key == arcade.key.RIGHT:
                        self.right_pressed = True
                        pressed_key = 'right'
                    direction_moved = pressed_key
                else:
                    if self.EEGdevice == 7:
                        # Audio beep to send pulse to sync box
                        play_sound('C') 
                    else:
                        # Send pulse through LSL
                        self.outlet.push_sample([moveCount])

                    # Input is reversed
                    if key == arcade.key.LEFT:
                        self.right_pressed = True
                        pressed_key = 'left'
                        direction_moved = 'right'
                    elif key == arcade.key.RIGHT:
                        self.left_pressed = True
                        pressed_key = 'right'
                        direction_moved = 'left'

                # Add to behavioral data dataframe
                if self.trialPressCount == self.pressCountToMove:
                    taskData.loc[moveCount] = pd.Series([move_time, \
                        self.player_sprite.center_x, self.player_sprite.center_y, \
                        self.target_sprite.center_x, self.target_sprite.center_y, \
                        pressed_key, direction_moved, self.reverseDirection, self.score], \
                        index=self.taskData_col_names)
                    increment()
    
    def on_key_release(self, key, modifiers):
        if self.inISI is False:
            """ Called when the user releases left or right key. """
            if key == arcade.key.LEFT or key == arcade.key.RIGHT:
                if self.reverseDirection is False:
                    # Everything functions normally
                    if key == arcade.key.LEFT:
                        self.left_pressed = False
                        if self.trialPressCount == self.pressCountToMove:
                            self.can_move = True
                    elif key == arcade.key.RIGHT:
                        self.right_pressed = False
                        if self.trialPressCount == self.pressCountToMove:
                            self.can_move = True
                else:
                    # Input is reversed
                    if key == arcade.key.LEFT:
                        self.right_pressed = False
                        if self.trialPressCount == self.pressCountToMove:
                            self.can_move = True
                    elif key == arcade.key.RIGHT:
                        self.left_pressed = False
                        if self.trialPressCount == self.pressCountToMove:
                            self.can_move = True
                
                # Update random variable to determine forced error
                if self.can_move is True:
                    if random() > ERROR_RATE:
                        self.reverseDirection = False
                    else:
                        self.reverseDirection = True
                
                # Only move the cursor every 3 presses (self.pressCountToMove)
                if self.trialPressCount == self.pressCountToMove:
                    self.trialPressCount = 0
    
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

        # Wait x number of seconds, then destroy the box and fixation cross
        rest_time = uniform(1,2)
        arcade.pause(rest_time)
        for item in self.ITI_list:
            item.kill()
        
        self.inISI = False

        # Reset cursor to center
        self.player_sprite.center_x = center_x
        self.player_sprite.center_y = center_y
        self.player_list.append(self.player_sprite)

        # Reset trial time-out timer
        self.trialTime = time.time()
        

def saveData(subjID, taskParameters, taskData):
    # Check how many files are currently present for this subject
    fileNum = len(glob.glob('SaveData/Error_Screening_' + subjID + '*.csv')) + 1

    # Save the task parameters to json file
    with open('SaveData/Error_Screening_' + subjID + '_R' + str(fileNum) + '_Parameters.json', 'w') as fp:
        json.dump(taskParameters, fp)

    # Save the data to csv file
    file_name = ('SaveData/Error_Screening_' + subjID + '_R' + str(fileNum) + '.csv')
    taskData.to_csv(file_name, encoding='utf-8')

    # Save the data as a json file
    taskData.to_json('SaveData/Error_Screening_' + subjID + '_R' + str(fileNum) + '.json')


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

    # Save task parameters
    taskParameters = {'initial_timestamp':initial_timestamp, \
        'column_count':COLUMN_COUNT,'row_count':ROW_COUNT,'width':WIDTH,\
        'height':HEIGHT,'margin':MARGIN,'error_rate':ERROR_RATE,'subjID':subjID}

    # Create pandas dataframe to store behavioural data
    col_names = ['time', 'player_x', 'player_y', 'target_x', 'target_y', 'key_pressed','direction_moved','error_induced','score']
    taskData = pd.DataFrame(columns = col_names)

    # Create the window to display
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, taskData, col_names, EEGdevice, outlet)
    game.setup()
    arcade.run()

    # Once window is closed
    saveData(subjID, taskParameters, taskData)