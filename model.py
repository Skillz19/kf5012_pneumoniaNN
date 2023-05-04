import tensorflow as tf
# noinspection PyUnresolvedReferences
from tensorflow.keras import Input
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Sequential, Model
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, \
    RandomFlip, RandomRotation, RandomCrop, ZeroPadding2D
from f1_score import F1Score
from enum_models import Models


class Network:
    def __init__(self, input_shape, include_top, weights_src, learning_rate, base_type,
                 rotation=0.2, flip='horizontal', crop=124, pad=2,  trainable_base=False,
                 model_name='harbottle_pneumonia'):
        self.include_top = include_top
        self.weights_src = weights_src
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.trainable_base = trainable_base
        self.rotation = rotation
        self.flip = flip
        self.crop = crop
        self.pad = pad
        self.name = model_name
        self.base_type = base_type

    def build_data_augmentation(self):
        return Sequential([
            RandomFlip(self.flip),
            RandomRotation(self.rotation),
            RandomCrop(self.crop, self.crop),
            ZeroPadding2D(self.pad)
        ])

    def load_model(self):
        if self.base_type == Models.ENUM_RES50:
            preprocess_input, base_model = self.load_resnet50()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_RES50V2:
            preprocess_input, base_model = self.load_resnet50v2()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_INCEPTIONV3:
            preprocess_input, base_model = self.load_inceptionv3()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_EFFICIENTNET:
            preprocess_input, base_model = self.load_efficientnet()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_EFFICIENTNETV2:
            preprocess_input, base_model = self.load_efficientnetv2()
            return preprocess_input, base_model
        else:
            print('Invalid base model type')
            exit(-1)

    def load_efficientnetv2(self):
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        # load the model
        base_model = tf.keras.applications.EfficientNetB0(input_shape=self.input_shape,
                                                          include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_efficientnet(self):
        preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
        # load the model
        base_model = tf.keras.applications.EfficientNetB0(input_shape=self.input_shape,
                                                          include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_inceptionv3(self):
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        # load the model
        base_model = tf.keras.applications.InceptionV3(input_shape=self.input_shape,
                                                       include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_resnet50v2(self):
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        # load the model
        base_model = tf.keras.applications.ResNet50V2(input_shape=self.input_shape,
                                                      include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

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

    def build_model(self):
        data_augmentation = self.build_data_augmentation()
        preprocess_input, base_model = self.load_model()
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
                               tf.keras.metrics.Recall(name='recall'),
                               F1Score()])
        model.summary()
        return model
