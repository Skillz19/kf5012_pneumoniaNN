from data_loader import DataLoader
from model import ResNet50
from train_evaluate import TrainTest
# noinspection PyCompatibility
import configparser
from datetime import datetime
import getpass
import os

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
    name = get_prefix() + '_'+ time + ".h5"
    name = './models/' + name
    return name
# returns a string for saving learning curves plot at the end of training
def get_plot_name(time):
    name = get_prefix() + '_' + time + '.png'
    name = './plots/'+name
    return name
def get_settings_name(time):
    name = get_prefix() + '_'+ time +'.ini'
    name = './settings/'+name
    return name

config = configparser.ConfigParser()
config.read('config.ini')

shape = tuple(map(int, config['DEFAULT']['shape'].strip('()').split(',')))
include_top = config.getboolean('DEFAULT', 'include_top')
weights = config['DEFAULT']['weights']
learning_rate = config.getfloat('DEFAULT', 'learning_rate')
batch_size = config.getint('DEFAULT','batch_size')
parent_dir = config['DEFAULT']['parent_dir']
rotation = config.getfloat('DEFAULT','rotation')
trainable_base = config.getboolean('DEFAULT', 'trainable_base')
flip = config['DEFAULT']['flip']
model_name = config['DEFAULT']['model_name']
epochs = config.getint('DEFAULT','epochs')
plot_learning_curve = config.getboolean('DEFAULT','plot_learning_curves')
prdi = config.getboolean('DEFAULT', 'print_random_dataset_image')

# Create an instance of the DataLoader class
dl = DataLoader(height=shape[0], width=shape[1], batch_size=batch_size, parent_dir=parent_dir)

# Load the data
train_ds, val_ds, test_ds = dl.load_data()

# Print the details of the training dataset
dl.print_dataset_details(train_ds)

# Create an instance of the model class
rn50 = ResNet50(input_shape=shape, include_top=False,weights_src=weights,learning_rate=learning_rate,
                rotation=rotation,flip=flip,trainable_base=trainable_base,model_name=model_name)
# build custom model based on resnet 50
model = rn50.build_resnet50()

# generate variables name for storing files
time_str = get_time_str()
plot_fig_name = get_plot_name(time_str)
saved_model_name = get_model_name(time_str)

# create settings directory if not exist
settings_name = get_settings_name(time_str)
path = os.path.dirname(settings_name)
if not os.path.exists(path):
    os.makedirs(path)

with open(settings_name, 'w') as configfile:
  config.write(configfile)

# Create an instance of the train and evaluate class
train = TrainTest(epochs, train_ds, test_ds, val_ds, plot_learning_curve, plot_fig_name, prdi)
# train and evaluate model
train.train(model)

model.save(saved_model_name)
