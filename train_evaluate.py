import matplotlib.pyplot as plt


class TrainTest:
    def __init__(self,epochs, train_ds, test_ds, val_ds, plot_learn_curve = False):
        self.epochs = epochs
        self.val_ds = val_ds
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.plot_learn_curve = plot_learn_curve

    def train(self,model):
        model.summary()
        hist = model.fit(self.train_ds, epochs=self.epochs, validation_data=self.val_ds)
        loss, accuracy, precision, recall = model.evaluate(self.test_ds)
        print(f' Accuracy/precision/recall scores on the test dataset: '
              f'{accuracy:.3f}/{precision:.3f}/{recall:.3f} loss: {loss:.3f}')
        self.plot_learning_curves(hist)
        return hist

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
            plt.show()
