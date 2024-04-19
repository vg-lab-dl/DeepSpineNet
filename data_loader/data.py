#  This file is part of DeepSpineNet
#  Copyright (C) 2021 VG-Lab (Visualization & Graphics Lab), Universidad Rey Juan Carlos
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import tensorflow as tf
from configuration.parser import YAMLConfig
from data_loader.preprocessing import PaddedStackPatcher


class TFRecordDataset:
    def __init__(self, configuration: YAMLConfig):
        self.configuration = configuration
        self.train_record_paths = configuration.get_entry(['Data', 'train_paths'])
        self.validation_record_paths = configuration.get_entry(['Data', 'validation_paths'])
        self.buffer_size = configuration.get_entry(['Data', 'buffer_size'])
        self.batch_size = configuration.get_entry(['Training', 'batch_size'])
        self.scaling = configuration.get_entry(['Data', 'scaling'], False) or 'FullRange'
        self.seg_threshold = configuration.get_entry(['Data', 'seg_threshold'], False)

        # Precomputed dataset-level stats
        self.min_value = 66
        self.max_value = 4095
        self.mean = 122.74
        self.std = 103.45

    def __parse__(self, serialized_example):
        feature = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                   'label': tf.io.FixedLenFeature([], tf.string),
                   'height': tf.io.FixedLenFeature([], tf.int64),
                   'width': tf.io.FixedLenFeature([], tf.int64),
                   'depth': tf.io.FixedLenFeature([], tf.int64)
                   }

        example = tf.io.parse_single_example(serialized_example, feature)
        image = tf.cast(tf.io.decode_raw(example['image_raw'], tf.uint16), tf.float32)

        if self.scaling == 'FullRange':
            image = image / 4095
        if self.scaling == 'DatasetNorm':
            image = (image - self.min_value) / (self.max_value - self.min_value)
        elif self.scaling == 'DatasetSTD':
            image = (image - self.mean) / self.std

        label = tf.io.decode_raw(example['label'], tf.uint8)
        height = int(example['width'])
        width = int(example['height'])
        depth = int(example['depth'])

        image = tf.reshape(image, shape=tf.stack([height, width, depth, 1]))
        label = tf.reshape(label, shape=tf.stack([height, width, depth, 1]))

        image = tf.transpose(image, [1, 2, 0, 3])
        label = tf.transpose(label, [1, 2, 0, 3])
        return image, label, depth, width, height

    def filter(self, images, labels, dweight=None):
        return (tf.math.count_nonzero(labels != 0) / tf.cast(tf.size(labels), tf.int64)) > self.seg_threshold

    def filter_non_seg(self, image, labels, height, width, depth):
        return tf.math.count_nonzero(labels == 1) > 0 and tf.math.count_nonzero(labels == 2) > 0

    def __get_dataset__(self, paths, train=False):
        sp = PaddedStackPatcher(self.configuration)
        dataset = tf.data.TFRecordDataset(paths)
        dataset = dataset.map(lambda x: self.__parse__(x), deterministic=True)
        dataset = dataset.filter(self.filter_non_seg)
        dataset = dataset.flat_map(
            lambda image, labels, height, width, depth: sp.get_patch_tf(image, labels, height, width, depth))
        if train and self.seg_threshold is not None:
            dataset = dataset.filter(self.filter)
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size * 5)
        return dataset

    def get_train_dataset(self):
        return self.__get_dataset__(self.train_record_paths, True)

    def get_val_dataset(self):
        return self.__get_dataset__(self.validation_record_paths)


class TFRecordValDataset:
    def __init__(self, configuration: YAMLConfig):
        self.record_paths = configuration.get_entry(['Test', 'data_path'])

    def read_tfrecord(self):
        reconstructed_images = []

        for path in self.record_paths:
            record_iterator = tf.compat.v1.io.tf_record_iterator(path=path)

            for string_record in record_iterator:
                example = tf.train.Example()
                example.ParseFromString(string_record)

                height = int(example.features.feature['height'].int64_list.value[0])
                width = int(example.features.feature['width'].int64_list.value[0])
                depth = int(example.features.feature['depth'].int64_list.value[0])
                img_string = (example.features.feature['image_raw'].bytes_list.value[0])
                label_string = (example.features.feature['label'].bytes_list.value[0])

                img_1d = np.frombuffer(img_string, dtype=np.uint16)
                reconstructed_img = img_1d.reshape((width, height, depth))

                label_1d = np.frombuffer(label_string, dtype=np.uint8)
                reconstructed_label = label_1d.reshape((width, height, depth))

                reconstructed_example = {'raw': reconstructed_img, 'label': reconstructed_label}
                reconstructed_images.append(reconstructed_example)

        return reconstructed_images
