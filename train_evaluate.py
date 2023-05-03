import matplotlib.pyplot as plt
import os
import tensorflow as tf


class TrainTest:
    def __init__(self, epochs, train_ds, test_ds, val_ds, plot_learn_curve=False,
                 plot_filename=None, print_images=False, use_early_stop=True, early_stop_patience=3,
                 early_stop_monitor='val_loss'):
        self.epochs = epochs
        self.val_ds = val_ds
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.plot_learn_curve = plot_learn_curve
        self.plot_filename = plot_filename
        self.print_images = print_images
        self.use_early_stop = use_early_stop
        self.early_stop_patience = early_stop_patience
        self.early_stop_monitor = early_stop_monitor

    # print some images from the training set
    def print_loaded_image(self):
        # Get the first batch of images and labels from your train dataset
        image_batch, label_batch = next(iter(self.train_ds))

        # Define a list of class names
        class_names = ['class_0', 'class_1', 'class_2']

        # Display the images with their corresponding labels
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 10))

        for i, ax in enumerate(axes.flat):
            # Display the image
            ax.imshow(image_batch[i].numpy().astype("uint8"))

            # Get the label of the image
            label_index = tf.argmax(label_batch[i])
            label = class_names[label_index]

            # Add the label to the image
            ax.set_title(label)

            # Hide the axis
            ax.axis('off')

        plt.show()

    def train(self, model):
        model.summary()
        if self.print_images:
            self.print_loaded_image()
        hist = model.fit(self.train_ds, epochs=self.epochs, validation_data=self.val_ds,
                         callbacks=[] if not self.use_early_stop
                         else [tf.keras.callbacks.EarlyStopping(monitor=self.early_stop_monitor,
                                                                patience=self.early_stop_patience,
                                                                verbose=1)])
        loss, accuracy, precision, recall, f1_score = model.evaluate(self.test_ds)
        msg = f' F1_Score/Accuracy/precision/recall/ scores on the test dataset: '\
              f'{f1_score:.3f}/{accuracy:.3f}/{precision:.3f}/{recall:.3f} loss: {loss:.3f}'
        print(msg)
        self.plot_learning_curves(hist)
        return hist, msg

    def plot_learning_curves(self, hist):
        if self.plot_learn_curve:
            acc = hist.history['accuracy']
            val_acc = hist.history['val_accuracy']

            precision = hist.history['precision']
            val_precision = hist.history['val_precision']

            recall = hist.history['recall']
            val_recall = hist.history['val_recall']

            loss = hist.history['loss']
            val_loss = hist.history['val_loss']

            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')

            plt.plot(precision, label='Training Precision')
            plt.plot(val_precision, label='Validation Precision')

            plt.plot(recall, label='Training Recall')
            plt.plot(val_recall, label='Validation Recall')

            plt.legend(loc='lower right')
            plt.ylabel('Accuracy')
            plt.ylim([min(plt.ylim()), 1])
            plt.title('Training and Validation Accuracy, Precision and Recall')

            plt.subplot(2, 1, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.ylabel('Cross Entropy')
            plt.ylim([0, 1.0])
            plt.title('Training and Validation Loss')
            plt.xlabel('epoch')
            fig = plt.gcf()
            plt.show()
            path = os.path.dirname(self.plot_filename)
            if not os.path.exists(path):
                os.makedirs(path)
            if self.plot_filename is not None:
                fig.savefig(self.plot_filename)
