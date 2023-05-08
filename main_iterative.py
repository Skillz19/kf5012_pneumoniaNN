from load_data import LoadInputData
from model import Network
from train_evaluate import TrainTest
# noinspection PyCompatibility
import configparser
from datetime import datetime
import getpass
import os
from enum_models import Models
from recorder_xlsx import Record
import gc
# noinspection PyUnresolvedReferences,PyPep8Naming
from tensorflow.keras import backend as K
import tensorflow as tf


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
    plot_name = get_prefix() + '_' + time + ".png"
    plot_name = './models/' + plot_name
    return name, plot_name


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


def get_record_name(time):
    name = get_prefix() + '_' + time + '.xlsx'
    name = './records/'+name
    return name


conf = tf.compat.v1.ConfigProto()
conf.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=conf))


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
# Print the details of the training dataset
dl.print_dataset_details(train_ds)


# test over different hyperparameters
custom_layer = (1, 2, 3)
create_missing_directory(record_name)
record_name = get_record_name(get_time_str())
record = Record(record_name)
for custom_layer in custom_layer:
    record.write_layer(custom_layer)
    record.write_header()
    # define different base models to train and test on
    base_models = [Models.ENUM_VGG16, Models.ENUM_VGG19, Models.ENUM_DENSENET121, Models.ENUM_NASNETLARGE]
    for base_model in base_models:
        record.write_base_model(base_model)
        activation_functions = ['relu'] if custom_layer == 3 else ['relu', 'leaky_relu', 'elu', 'gelu']
        # activation_functions = ['relu', 'leaky_relu', 'elu', 'gelu']
        best_activation = ''
        max_f1_score = 0
        for activation in activation_functions:
            # Create an instance of the model class
            nn = Network(input_shape=shape, include_top=False, weights_src=weights, learning_rate=learning_rate,
                         base_type=base_model, rotation=rotation, flip=flip, trainable_base=trainable_base,
                         model_name=model_name, custom_layer=custom_layer, activation=activation)
            # build custom model based on resnet 50
            model = nn.build_model()

            # generate variables name for storing files
            time_str = get_time_str()
            plot_fig_name = get_plot_name(time_str)
            saved_model_name, model_plot_name = get_model_name(time_str)

            settings_name = get_settings_name(time_str)
            # create settings directory if not exist
            create_missing_directory(settings_name)

            with open(settings_name, 'w') as configfile:
                config.write(configfile)

            # Create an instance of the train and evaluate class
            train = TrainTest(epochs, train_ds, test_ds, val_ds, plot_learning_curve, plot_fig_name, prdi,
                              use_es, es_patience, es_monitor)
            # train and evaluate model
            hist = train.train(model)
            test_loss, test_accuracy, test_precision, test_recall, test_f1_score = model.evaluate(test_ds)
            if test_f1_score > max_f1_score:
                max_f1_score = test_f1_score
                best_activation = activation

            result_name = get_result_name()
            create_missing_directory(result_name)

            msg = f' F1_Score/Accuracy/precision/recall/ scores on the test dataset: ' \
                  f'{test_f1_score:.3f}/{test_accuracy:.3f}/{test_precision:.3f}/{test_recall:.3f}' \
                  f' loss: {test_loss:.3f}'
            with open(result_name, 'a') as f:
                f.write('\n'+msg)

            f1_score = (hist.history['f1_score'][-1], hist.history['val_f1_score'][-1], test_f1_score)
            accuracy = (hist.history['accuracy'][-1], hist.history['val_accuracy'][-1], test_accuracy)
            precision = (hist.history['precision'][-1], hist.history['val_precision'][-1], test_precision)
            recall = (hist.history['recall'][-1], hist.history['val_recall'][-1], test_recall)
            loss = (hist.history['loss'][-1], hist.history['val_loss'][-1], test_loss)

            record.write_values(activation, f1_score, accuracy, precision, recall, loss)
            record.save()
            # clear memory
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            del model
            del train
            del hist
            del nn
            gc.collect()

        record.write_learn_row_title(best_activation)
        # run test again with the best activation function
        learn_rate = [0.0001, 0.0005, 0.005, 0.01, 0.05]
        for learn in learn_rate:
            # Create an instance of the model class
            nn = Network(input_shape=shape, include_top=False, weights_src=weights, learning_rate=learn,
                         base_type=base_model, rotation=rotation, flip=flip, trainable_base=trainable_base,
                         model_name=model_name, custom_layer=custom_layer, activation=best_activation)
            # build custom model based on resnet 50
            model = nn.build_model()

            # generate variables name for storing files
            time_str = get_time_str()
            plot_fig_name = get_plot_name(time_str)
            saved_model_name, model_plot_name = get_model_name(time_str)

            settings_name = get_settings_name(time_str)
            # create settings directory if not exist
            create_missing_directory(settings_name)

            with open(settings_name, 'w') as configfile:
                config.write(configfile)

            # Create an instance of the train and evaluate class
            train = TrainTest(epochs, train_ds, test_ds, val_ds, plot_learning_curve, plot_fig_name, prdi,
                              use_es, es_patience, es_monitor)
            # train and evaluate model
            hist = train.train(model)
            test_loss, test_accuracy, test_precision, test_recall, test_f1_score = model.evaluate(test_ds)

            result_name = get_result_name()
            create_missing_directory(result_name)

            msg = f' F1_Score/Accuracy/precision/recall/ scores on the test dataset: ' \
                  f'{test_f1_score:.3f}/{test_accuracy:.3f}/{test_precision:.3f}/{test_recall:.3f}' \
                  f' loss: {test_loss:.3f}'
            with open(result_name, 'a') as f:
                f.write('\n' + msg)

            f1_score = (hist.history['f1_score'][-1], hist.history['val_f1_score'][-1], test_f1_score)
            accuracy = (hist.history['accuracy'][-1], hist.history['val_accuracy'][-1], test_accuracy)
            precision = (hist.history['precision'][-1], hist.history['val_precision'][-1], test_precision)
            recall = (hist.history['recall'][-1], hist.history['val_recall'][-1], test_recall)
            loss = (hist.history['loss'][-1], hist.history['val_loss'][-1], test_loss)

            record.write_values(learn, f1_score, accuracy, precision, recall, loss)
            record.save()
            # clear memory
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            del model
            del train
            del hist
            del nn
            gc.collect()
record.save()
