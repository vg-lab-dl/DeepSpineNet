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
from scipy.ndimage import distance_transform_edt
from tensorflow.python.data import Dataset

from configuration.parser import YAMLConfig


class PaddedStackPatcher:
    def __init__(self, configuration: YAMLConfig):
        self.configuration = configuration
        self.input_size = configuration.get_entry(['Network', 'input_size'])
        self.output_size = configuration.get_entry(['Network', 'output_size'])
        self.input_d = configuration.get_entry(['Network', 'input_depth'])
        self.output_d = configuration.get_entry(['Network', 'output_depth'])
        self.spacing = [0.279911, 0.0751562, 0.0751562]
        self.d_weight = configuration.get_entry(['Data', 'd_weight'])

        self.scaling = configuration.get_entry(['Data', 'scaling'], False) or 'DatasetNorm'

        self.data_aug_rot = configuration.get_entry(['Data', 'data_augmentation', 'rotate'], False)
        self.data_aug_flip = configuration.get_entry(['Data', 'data_augmentation', 'flip'], False)
        self.data_aug_trans = configuration.get_entry(['Data', 'data_augmentation', 'translate'], False)
        self.data_aug = self.data_aug_rot or self.data_aug_flip or self.data_aug_trans

        if self.data_aug:
            self.data_aug_prob = configuration.get_entry(['Data', 'data_augmentation', 'probability'])

        if self.data_aug_trans:
            self.data_aug_trans_mag = configuration.get_entry(['Data', 'data_augmentation', 'translation_magnitude'])

        self.padding_type = configuration.get_entry(['Data', 'padding_type'], False) or "CONSTANT"
        self.padding_type_depth = configuration.get_entry(['Data', 'padding_type_depth'], False) or "CONSTANT"

        if self.d_weight:
            self.weight_method = configuration.get_entry(['Data', 'weight_method', 'type'])
            self.at_stack_level = configuration.get_entry(['Data', 'weight_method', 'at_stack_level'])
            self.use_depth_context = configuration.get_entry(['Data', 'weight_method', 'use_depth_context'])

            if self.weight_method == "exp":
                self.weight_factor = configuration.get_entry(['Data', 'weight_method', 'factor'])

            elif self.weight_method == "window":
                self.weight_cutoff_dist = configuration.get_entry(['Data', 'weight_method', 'cutoff_dist'])
                self.weight_close_weight = configuration.get_entry(['Data', 'weight_method', 'close_weight'])
                self.weight_far_weight = configuration.get_entry(['Data', 'weight_method', 'far_weight'])
            else:
                raise ValueError('Selected weight method "{}" its not supported.'.format(self.weight_method))

    def get_weight_map(self, label_stack):
        label_inv = label_stack == 0
        if self.use_depth_context:
            d = distance_transform_edt(label_inv, sampling=self.spacing).astype(np.float32)
        else:
            d = []
            for i in range(label_inv.shape[2]):
                label_slice = label_inv[:, :, i]
                if np.any(label_slice == False):
                    d.append(distance_transform_edt(label_slice, sampling=self.spacing[:-1]).astype(np.float32))
                else:
                    d.append(np.ones_like(label_slice, dtype=np.float32) * np.inf)
            d = np.stack(d, axis=2)

        if self.weight_method == "exp":
            weight_map = np.exp(self.weight_factor * d)
        elif self.weight_method == "window":
            d[d > self.weight_cutoff_dist] = self.weight_cutoff_dist
            weight_map = np.power((1 - np.power((d / self.weight_cutoff_dist), self.weight_close_weight)),
                                  self.weight_far_weight)
        else:
            raise ValueError('Selected weight method "{}" its not supported.'.format(self.weight_method))

        return weight_map

    def get_patch_tf(self, image_stack, label_stack, height, width, channels):
        rng = tf.random.get_global_generator()
        input_height = self.input_size
        input_width = self.input_size
        input_channels = self.input_d
        output_height = self.output_size
        output_width = self.output_size
        output_channels = self.output_d

        if self.scaling == 'StackNorm':
            min_value = tf.math.reduce_min(image_stack)
            max_value = tf.math.reduce_max(image_stack)
            image_stack = (image_stack - min_value) / (max_value - min_value)
        elif self.scaling == 'StackSTD':
            image_stack = (image_stack - tf.math.reduce_mean(image_stack)) / tf.math.reduce_std(image_stack)

        padding_l = int((input_width - output_width) / 2)
        padding_t = int((input_height - output_height) / 2)
        padding_r = int((input_width - output_width) / 2)
        padding_b = int((input_height - output_height) / 2)
        padding_s = int(tf.math.ceil((input_channels - output_channels) / 2))
        padding_e = int(tf.math.ceil((input_channels - output_channels) / 2))

        padded_width = width + padding_l + padding_r
        padded_height = height + padding_t + padding_b
        padded_channels = channels + padding_s + padding_e

        padded_img = tf.pad(image_stack[:, :, :, 0],
                            [[padding_t, padding_b], [padding_l, padding_r], [0, 0]], self.padding_type)
        padded_labels = tf.pad(label_stack[:, :, :, 0],
                               [[padding_t, padding_b], [padding_l, padding_r], [0, 0]], self.padding_type)

        padded_img = tf.pad(padded_img, [[0, 0], [0, 0], [padding_s, padding_e]], self.padding_type_depth)
        padded_labels = tf.pad(padded_labels, [[0, 0], [0, 0], [padding_s, padding_e]], self.padding_type_depth)

        w_patches = tf.cast(tf.math.ceil(padded_width / self.output_size), dtype=tf.int32)
        start_w_index = tf.cast(tf.linspace(padding_l, padded_width - padding_r - self.output_size, w_patches),
                                tf.int32)

        h_patches = tf.cast(tf.math.ceil(padded_height / self.output_size), dtype=tf.int32)
        start_h_index = tf.cast(tf.linspace(padding_t, padded_height - padding_b - self.output_size, h_patches),
                                tf.int32)

        c_patches = tf.cast(tf.math.ceil(padded_channels / output_channels), dtype=tf.int32)
        start_c_index = tf.cast(tf.linspace(padding_s, padded_channels - padding_e - output_channels, c_patches),
                                tf.int32)

        rows_idx = tf.tile(tf.repeat(start_h_index, w_patches), [c_patches])
        cols_idx = tf.tile(start_w_index, [h_patches * c_patches])
        channels_idx = tf.repeat(start_c_index, w_patches * h_patches)

        num_patches = w_patches * h_patches * c_patches
        batch_image = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=1)
        batch_label = tf.TensorArray(dtype=tf.uint8, dynamic_size=True, size=1)
        batch_d_weight = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=1)
        patch_idx = 0
        padded_d_weight = None

        if self.d_weight and self.at_stack_level:
            padded_d_weight = tf.numpy_function(self.get_weight_map, inp=[padded_labels],
                                                Tout=tf.float32)

        for i in range(num_patches):
            soi = [rows_idx[i], cols_idx[i], channels_idx[i]]
            eoi = [rows_idx[i] + self.output_size, cols_idx[i] + self.output_size, channels_idx[i] + output_channels]
            sii = [rows_idx[i] - padding_t, cols_idx[i] - padding_l, channels_idx[i] - padding_s]
            eii = [sii[0] + input_height, sii[1] + input_height, sii[2] + input_channels]

            label_patch = padded_labels[soi[0]:eoi[0], soi[1]:eoi[1], soi[2]:eoi[2]]

            if tf.math.reduce_any(label_patch == 1) or tf.math.reduce_any(label_patch == 2):
                image_patch = padded_img[sii[0]:eii[0], sii[1]:eii[1], sii[2]:eii[2]]

                if self.scaling == 'PatchNorm':
                    min_value = tf.math.reduce_min(image_patch)
                    max_value = tf.math.reduce_max(image_patch)
                    image_patch = (image_patch - min_value) / (max_value - min_value)
                elif self.scaling == 'PatchSTD':
                    image_patch = (image_patch - tf.math.reduce_mean(image_patch)) / tf.math.reduce_std(image_patch)

                d_weight_patch = []
                if self.d_weight:
                    if self.at_stack_level:
                        d_weight_patch = padded_d_weight[soi[0]:eoi[0], soi[1]:eoi[1], soi[2]:eoi[2]]
                    else:
                        d_weight_patch = tf.numpy_function(self.get_weight_map, inp=[label_patch],
                                                           Tout=tf.float32)

                data_aug = self.data_aug and rng.uniform([1])[0] <= self.data_aug_prob
                if data_aug:
                    if self.data_aug_trans:
                        max_width = int(input_width * self.data_aug_trans_mag)
                        max_height = int(input_height * self.data_aug_trans_mag)
                        t_col = rng.uniform([1], -max_width, max_width, tf.dtypes.int32)[0]
                        t_rows = rng.uniform([1], -max_height, max_height, tf.dtypes.int32)[0]
                        padded_img_height = height + padding_t + padding_b
                        padded_img_width = width + padding_l + padding_r

                        sii[0] = tf.math.minimum(tf.math.maximum(0, sii[0] + t_rows), padded_img_height - input_height)
                        eii[0] = tf.math.minimum(tf.math.maximum(input_height, eii[0] + t_rows), padded_img_height)
                        sii[1] = tf.math.minimum(tf.math.maximum(0, sii[1] + t_col), padded_img_width - input_width)
                        eii[1] = tf.math.minimum(tf.math.maximum(input_width, eii[1] + t_col), padded_img_width)

                        soi[0] = tf.math.minimum(tf.math.maximum(0, soi[0] + t_rows), padded_img_height - output_height)
                        eoi[0] = tf.math.minimum(tf.math.maximum(output_height, eoi[0] + t_rows), padded_img_height)
                        soi[1] = tf.math.minimum(tf.math.maximum(0, soi[1] + t_col), padded_img_width - output_width)
                        eoi[1] = tf.math.minimum(tf.math.maximum(output_width, eoi[1] + t_col), padded_img_width)

                        label_patch = padded_labels[soi[0]:eoi[0], soi[1]:eoi[1], soi[2]:eoi[2]]
                        image_patch = padded_img[sii[0]:eii[0], sii[1]:eii[1], sii[2]:eii[2]]
                        if not tf.math.reduce_any(label_patch == 1) and not tf.math.reduce_any(label_patch == 2):
                            continue

                    if self.data_aug_rot:
                        times = rng.uniform([1], 1, 3, tf.dtypes.int32)[0]
                        image_patch = tf.image.rot90(image_patch, times)
                        label_patch = tf.image.rot90(label_patch, times)
                        if self.d_weight:
                            d_weight_patch = tf.image.rot90(d_weight_patch, times)

                    if self.data_aug_flip:
                        flip_type = rng.uniform([1])[0]
                        if flip_type < 0.5:
                            image_patch = tf.image.flip_up_down(image_patch)
                            label_patch = tf.image.flip_up_down(label_patch)
                            if self.d_weight:
                                d_weight_patch = tf.image.flip_up_down(d_weight_patch)
                        else:
                            image_patch = tf.image.flip_left_right(image_patch)
                            label_patch = tf.image.flip_left_right(label_patch)
                            if self.d_weight:
                                d_weight_patch = tf.image.flip_left_right(d_weight_patch)

                batch_image = batch_image.write(patch_idx, image_patch[:, :, :, None])
                batch_label = batch_label.write(patch_idx, label_patch[:, :, :, None])
                if self.d_weight:
                    batch_d_weight = batch_d_weight.write(patch_idx, d_weight_patch)

                patch_idx += 1
            else:
                continue

        if self.d_weight:
            return Dataset.from_tensor_slices((batch_image.stack(), batch_label.stack(), batch_d_weight.stack()))
        else:
            return Dataset.from_tensor_slices((batch_image.stack(), batch_label.stack()))
