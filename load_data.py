import tensorflow as tf


class LoadInputData:
    def __init__(self, height, width, batch_size, parent_dir, val_split, use_prefetch=True, color_mode='rgb'):
        self.image_height = height
        self.image_width = width
        self.batch_size = batch_size
        self.color_mode = color_mode
        self.parent_dir = parent_dir
        self.train_dir = parent_dir + '/train'
        self.test_dir = parent_dir + '/test'
        self.use_prefetch = use_prefetch
        self.class_names = None
        self.val_split = val_split

    def __load_inputs(self):
        train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(self.train_dir,
                                                                               validation_split=self.val_split,
                                                                               subset="both", seed=123,
                                                                               image_size=(
                                                                                   self.image_height, self.image_width),
                                                                               batch_size=self.batch_size,
                                                                               label_mode='categorical')
        test_ds = self.load_test_data()

        self.class_names = train_ds.class_names
        return train_ds, val_ds, test_ds

    def load_test_data(self):
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(self.test_dir, seed=123,
                                                                      image_size=(self.image_height, self.image_width),
                                                                      batch_size=self.batch_size,
                                                                      label_mode='categorical')
        return test_ds

    def print_dataset_details(self, dataset):
        print(self.class_names)
        # print(dataset)
        for image_batch, labels_batch in dataset:
            print(image_batch.shape)
            print(labels_batch.shape)
            break

    def load_data(self):
        train_ds, val_ds, test_ds = self.__load_inputs()
        return self.use_buffered_prefetching(train_ds, val_ds, test_ds)

    def use_buffered_prefetching(self, train, val, test):
        if not self.use_prefetch:
            return train, val, test

        def use_buffered_prefetching_for(dataset):
            return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        return [use_buffered_prefetching_for(dataset) for dataset in [train, val, test]]
