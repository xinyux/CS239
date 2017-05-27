# SmartWatch as Sleep and Sleep Quality Detector

The goal of this project is to investigate how well a smartwatch can sense and model sleep qulity without significantly changes in people's behavior. We used Andriod app "Decent logger" to log sensor inputs, including accelerometer, magnetic_field, orientation, gyroscope, gravity, linear_acceleration, rotation_vector, significant_motion. The input is sampled at 200Hz. Everyday, a survey is collected from the participant, the survey collects sleep information (time go to bed, time wake up, sleep quality, how sleepy the participant feel next day). We then perform feature engeering to detect sleep and predict sleep quality. 


## Data collection

Due to battery life constrain, the participant is not asked to wear the smartwatch while sleeping, instead, the participant is asked to charge the watch and place it on bed. 


## Files

feature_eng.py : generate features for sleep/nonsleep classification 

model.py : train classification models and predict


## adb commands

Installation

`$ brew cask install android-platform-tools`

Go to adb shell

`$ adb shell`

Pull all session data out to current directory

`$ adb pull /mnt/sdcard/Sessions .`

