from data_loader import DataLoader
from model import ResNet50
from train_evaluate import TrainTest
import configparser

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

# Create an instance of the train and evaluate class
train = TrainTest(epochs,train_ds,test_ds,val_ds,plot_learning_curve)
# train and evaluate model
train.train(model)
