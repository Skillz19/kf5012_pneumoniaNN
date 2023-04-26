import tensorflow as tf
# noinspection PyUnresolvedReferences
from tensorflow.keras import Input
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Sequential, Model
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, \
    RandomFlip, RandomRotation


class ResNet50:
    def __init__(self, input_shape, include_top, weights_src, learning_rate,
                 rotation=0.2, flip='horizontal', trainable_base=False,
                 model_name='harbottle_pneumonia'):
        self.include_top = include_top
        self.weights_src = weights_src
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.trainable_base = trainable_base
        self.rotation = rotation
        self.flip = flip
        self.name = model_name

    def build_data_augmentation(self):
        return Sequential([
            RandomFlip(self.flip),
            RandomRotation(self.rotation)
        ])

    def load_resnet50(self):
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
        # load the model
        base_model = tf.keras.applications.ResNet50(input_shape=self.input_shape,
                                                    include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    @staticmethod
    def add_custom_layers(model):
        model = Conv2D(32, (3, 3), activation='relu', padding='same')(model)
        model = Dropout(0.3)(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Conv2D(32, (3, 3), activation='relu', padding='same')(model)
        model = BatchNormalization()(model)
        model = Dropout(0.3)(model)
        # switch to vectors for classification
        model = Flatten()(model)
        model = Dense(100)(model)
        model = Dropout(0.3)(model)
        return model

    def add_output_layer(self, model, inputs):
        outputs = Dense(3, activation='softmax')(model)
        return Model(inputs=inputs, outputs=outputs, name=self.name)

    def build_resnet50(self):
        data_augmentation = self.build_data_augmentation()
        preprocess_input, base_model = self.load_resnet50()
        inputs = Input(shape=self.input_shape)
        mod = data_augmentation(inputs)
        mod = preprocess_input(mod)
        mod = base_model(mod, self.trainable_base)
        mod = self.add_custom_layers(mod)
        model = self.add_output_layer(mod, inputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall')])

        return model
