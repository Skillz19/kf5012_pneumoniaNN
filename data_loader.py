import tensorflow as tf


class DataLoader:
    def __init__(self, height, width, batch_size, parent_dir, use_prefetch=True, color_mode='rgb'):
        self.image_height = height
        self.image_width = width
        self.batch_size = batch_size
        self.color_mode = color_mode
        self.class_names = []
        self.parent_dir = parent_dir
        self.train_dir = parent_dir + '/train'
        self.val_dir = parent_dir + '/val'
        self.test_dir = parent_dir + '/test'
        self.use_prefetch = use_prefetch

    def load_data_from_folder(self):

        def load_data_from(directory):
            ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory,
                shuffle=True,
                batch_size=self.batch_size,
                image_size=(self.image_height, self.image_width),
                color_mode=self.color_mode
            )
            # Convert labels to one-hot vectors
            self.class_names = ds.class_names
            ds = ds.map(lambda x, y: (x, tf.one_hot(y, depth=len(ds.class_names))))
            return ds

        return [load_data_from(folder) for folder in [self.train_dir, self.val_dir, self.test_dir]]

    def print_dataset_details(self, dataset):
        print(f'Dataset type is {type(dataset)}')
        print(f' in the training set {len(self.class_names)} class names to classify the images for: '
              f'{self.class_names}')
        for image_batch, labels_batch in dataset:
            print(f'image batch shape: {image_batch.shape}')
            print(f'image label shape: {labels_batch.shape}')
            break

    @staticmethod
    def use_buffered_prefetching(train, val, test):
        def use_buffered_prefetching_for(dataset):
            return dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return [use_buffered_prefetching_for(dataset) for dataset in [train, val, test]]

    def load_data(self):
        train_ds, val_ds, test_ds = self.load_data_from_folder()
        if self.use_prefetch:
            return self.use_buffered_prefetching(train_ds, val_ds, test_ds)
        else:
            return train_ds, val_ds, test_ds
