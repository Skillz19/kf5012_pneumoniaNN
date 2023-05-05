import tensorflow as tf
# noinspection PyUnresolvedReferences
from tensorflow.keras import Input
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Sequential, Model
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, \
    RandomFlip, RandomRotation
from f1_score import F1Score
from enum_models import Models


class Network:
    def __init__(self, input_shape, include_top, weights_src, learning_rate, base_type,
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
        self.base_type = base_type

    def build_data_augmentation(self):
        return Sequential([
            RandomFlip(self.flip),
            RandomRotation(self.rotation)
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
        elif self.base_type == Models.ENUM_VGG16:
            preprocess_input, base_model = self.load_vgg16()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_VGG19:
            preprocess_input, base_model = self.load_vgg19()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_MOBILENET:
            preprocess_input, base_model = self.load_mobilenet()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_MOBILENETV2:
            preprocess_input, base_model = self.load_mobilenetv2()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_DENSENET121:
            preprocess_input, base_model = self.load_densenet121()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_DENSENET169:
            preprocess_input, base_model = self.load_densenet169()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_DENSENET201:
            preprocess_input, base_model = self.load_densenet201()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_NASNETLARGE:
            preprocess_input, base_model = self.load_nasnetlarge()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_NASNETMOBILE:
            preprocess_input, base_model = self.load_nasnetmobile()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_XCEPTION:
            preprocess_input, base_model = self.load_xception()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_INCEPTIONRESNETV2:
            preprocess_input, base_model = self.load_inceptionresnetv2()
            return preprocess_input, base_model
        elif self.base_type == Models.ENUM_CONVNET:
            preprocess_input, base_model = self.load_convnet()
            return preprocess_input, base_model
        else:
            print('Invalid base model type')
            exit(-1)

    def load_convnet(self):
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        # load the model
        base_model = tf.keras.applications.VGG16(input_shape=self.input_shape,
                                                 include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_nasnetmobile(self):
        preprocess_input = tf.keras.applications.nasnet.preprocess_input
        # load the model
        base_model = tf.keras.applications.NASNetMobile(input_shape=self.input_shape,
                                                        include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_xception(self):
        preprocess_input = tf.keras.applications.xception.preprocess_input
        # load the model
        base_model = tf.keras.applications.Xception(input_shape=self.input_shape,
                                                    include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_inceptionresnetv2(self):
        preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
        # load the model
        base_model = tf.keras.applications.InceptionResNetV2(input_shape=self.input_shape,
                                                             include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_vgg19(self):
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
        # load the model
        base_model = tf.keras.applications.VGG19(input_shape=self.input_shape,
                                                 include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_mobilenet(self):
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
        # load the model
        base_model = tf.keras.applications.MobileNet(input_shape=self.input_shape,
                                                     include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_mobilenetv2(self):
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        # load the model
        base_model = tf.keras.applications.MobileNetV2(input_shape=self.input_shape,
                                                       include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_densenet121(self):
        preprocess_input = tf.keras.applications.densenet.preprocess_input
        # load the model
        base_model = tf.keras.applications.DenseNet121(input_shape=self.input_shape,
                                                       include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_densenet169(self):
        preprocess_input = tf.keras.applications.densenet.preprocess_input
        # load the model
        base_model = tf.keras.applications.DenseNet169(input_shape=self.input_shape,
                                                       include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_densenet201(self):
        preprocess_input = tf.keras.applications.densenet.preprocess_input
        # load the model
        base_model = tf.keras.applications.DenseNet201(input_shape=self.input_shape,
                                                       include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_nasnetlarge(self):
        preprocess_input = tf.keras.applications.nasnet.preprocess_input
        # load the model
        base_model = tf.keras.applications.NASNetLarge(input_shape=self.input_shape,
                                                       include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

    def load_vgg16(self):
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        # load the model
        base_model = tf.keras.applications.VGG16(input_shape=self.input_shape,
                                                 include_top=self.include_top, weights=self.weights_src)
        base_model.trainable = self.trainable_base
        return preprocess_input, base_model

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

    @staticmethod
    def add_custom_layers2(model):
        model = Conv2D(64, (3, 3), activation='relu', padding='valid', strides=2, dilation_rate=(1, 1))(model)
        model = BatchNormalization()(model)
        model = Conv2D(32, (3, 3), activation='relu', padding='valid', strides=2, dilation_rate=(1, 1))(model)
        model = BatchNormalization()(model)
        model = Conv2D(256, (3, 3), activation='relu', padding='valid', strides=2, dilation_rate=(1, 1))(model)
        model = BatchNormalization()(model)
        model = Conv2D(512, (3, 3), activation='relu', padding='valid', strides=2, dilation_rate=(1, 1))(model)
        model = BatchNormalization()(model)
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
