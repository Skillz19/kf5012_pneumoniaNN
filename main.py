# from data_loader import DataLoader
from load_data import LoadInputData
from model import ResNet50
from train_evaluate import TrainTest
# noinspection PyCompatibility
import configparser
from datetime import datetime
import getpass
import os
import numpy as np


# returns a formatted string variable of current time
def get_time_str():
    time = datetime.now().isoformat(timespec="minutes")
    time = time.replace('-', '_')
    time = time.replace(':', '_')
    time = time.replace('T', ' ')
    return time


# returns username to be used as prefix for naming files
def get_prefix():
    user = getpass.getuser()
    if user is None or len(user) == 0:
        user = 'model'
    return user


# returns a string for naming models each time the model is trained
def get_model_name(time):
    name = get_prefix() + '_' + time + ".h5"
    name = './models/' + name
    return name


# returns name for saving train test results to file
def get_result_name():
    name = get_prefix() + '_results.txt'
    name = './results/' + name
    return name


# create a directory if it is missing
def create_missing_directory(directory):
    path = os.path.dirname(directory)
    if not os.path.exists(path):
        os.makedirs(path)


# returns a string for saving learning curves plot at the end of training
def get_plot_name(time):
    name = get_prefix() + '_' + time + '.png'
    name = './plots/'+name
    return name


def get_settings_name(time):
    name = get_prefix() + '_' + time + '.ini'
    name = './settings/'+name
    return name


config = configparser.ConfigParser()
config.read('config.ini')

shape = tuple(map(int, config['DEFAULT']['shape'].strip('()').split(',')))
include_top = config.getboolean('DEFAULT', 'include_top')
weights = config['DEFAULT']['weights']
learning_rate = config.getfloat('DEFAULT', 'learning_rate')
batch_size = config.getint('DEFAULT', 'batch_size')
parent_dir = config['DEFAULT']['parent_dir']
rotation = config.getfloat('DEFAULT', 'rotation')
trainable_base = config.getboolean('DEFAULT', 'trainable_base')
flip = config['DEFAULT']['flip']
model_name = config['DEFAULT']['model_name']
epochs = config.getint('DEFAULT', 'epochs')
plot_learning_curve = config.getboolean('DEFAULT', 'plot_learning_curves')
prdi = config.getboolean('DEFAULT', 'print_random_dataset_image')
use_es = config.getboolean('DEFAULT', 'use_early_stopping')
es_patience = config.getint('DEFAULT', 'early_stop_patience')
es_monitor = config['DEFAULT']['early_stop_monitor']
buffered_prefetch = config.getboolean('DEFAULT', 'buffered_prefetch')
validation_split = config.getfloat('DEFAULT', 'validation_split')

# Create an instance of the DataLoader class
dl = LoadInputData(height=shape[0], width=shape[1], batch_size=batch_size, parent_dir=parent_dir,
                   val_split=validation_split, use_prefetch=buffered_prefetch)

# Load the data
train_ds, val_ds, test_ds = dl.load_data()
'''for image, label in train_ds:
    print(image)
    print(label)
    print(np.min(image), np.max(image))
    break
exit(0)
'''
# Print the details of the training dataset
dl.print_dataset_details(train_ds)
# Create an instance of the model class
rn50 = ResNet50(input_shape=shape, include_top=False, weights_src=weights, learning_rate=learning_rate,
                rotation=rotation, flip=flip, trainable_base=trainable_base, model_name=model_name)
# build custom model based on resnet 50
model = rn50.build_resnet50()

# generate variables name for storing files
time_str = get_time_str()
plot_fig_name = get_plot_name(time_str)
saved_model_name = get_model_name(time_str)

settings_name = get_settings_name(time_str)
# create settings directory if not exist
create_missing_directory(settings_name)

with open(settings_name, 'w') as configfile:
    config.write(configfile)

# Create an instance of the train and evaluate class
train = TrainTest(epochs, train_ds, test_ds, val_ds, plot_learning_curve, plot_fig_name, prdi,
                  use_es, es_patience, es_monitor)
# train and evaluate model
hist, result = train.train(model)

result_name = get_result_name()
create_missing_directory(result_name)

with open(result_name, 'a') as txt:
    txt.write('\nTrain time: '+time_str+' - '+result)

model.save(saved_model_name)
