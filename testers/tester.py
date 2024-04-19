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

import csv
import math
from pathlib import Path

from skimage import io
from tensorflow.python.keras import Model
from tqdm import tqdm

from configuration.parser import YAMLConfig
from data_loader.data import TFRecordValDataset
import os
import numpy as np

from metrics.metrics import Metrics

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Tester:
    def __init__(self, configuration: YAMLConfig, model: Model, dataset: TFRecordValDataset = None):
        self.num_classes = configuration.get_entry(['Data', 'num_classes'])
        self.model_save_path = configuration.get_entry(['Output', 'model_save_path'])
        self.input_size = configuration.get_entry(['Network', 'input_size'])
        self.output_size = configuration.get_entry(['Network', 'output_size'])
        self.input_d = configuration.get_entry(['Network', 'input_depth'])
        self.output_d = configuration.get_entry(['Network', 'output_depth'])
        self.batch_size = configuration.get_entry(['Test', 'batch_size'])
        self.output_image_path = Path(configuration.get_entry(['Test', 'images_output_path']))

        self.output_image_path.mkdir(parents=True, exist_ok=True)
        self.images = None
        if dataset is not None:
            self.images = dataset.read_tfrecord()
        self.model = model
        self.metrics = Metrics(configuration)

        self.scaling = configuration.get_entry(['Data', 'scaling'], False)

        self.padding_type = configuration.get_entry(['Data', 'padding_type'], False) or "CONSTANT"
        self.padding_type_depth = configuration.get_entry(['Data', 'padding_type_depth'], False) or "CONSTANT"

        self.padding_type = self.padding_type.lower()
        self.padding_type_depth = self.padding_type_depth.lower()

        # Precomputed dataset-level stats
        self.min_value = 66
        self.max_value = 4095
        self.mean = 122.74
        self.std = 103.45

    def test(self, w_map):
        output_csv_path = self.output_image_path / 'results.csv'
        output_file = open(str(output_csv_path), mode='a', newline='')
        csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not output_csv_path.exists():
            csv_writer.writerow(
                ['image_id', 'F1_c0', 'F1_c1', 'F1_c2', 'F1_mean', 'Recall_c0', 'Recall_c1', 'Recall_c2',
                 'Precision_c0', 'Precisionl_c1', 'Precision_c2', 'IoU_c0', 'IoU_c1', 'IoU_c2', 'IoU_mean', 'w_map'])

        num_images = len(self.images)
        total_errors = np.zeros((self.num_classes, 4), dtype=np.int64)

        for img_index, val_data in enumerate(self.images):
            image_stack = val_data['raw']

            if self.scaling is None or self.scaling == 'FullRange':
                image_stack = image_stack / 4095
            elif self.scaling == 'DatasetNorm':
                image_stack = (image_stack - self.min_value) / (self.max_value - self.min_value)
            elif self.scaling == 'DatasetSTD':
                image_stack = (image_stack - self.mean) / self.std
            elif self.scaling == 'StackNorm':
                min_value = image_stack.min()
                max_value = image_stack.max()
                image_stack = (image_stack - min_value) / (max_value - min_value)
            elif self.scaling == 'StackSTD':
                image_stack = (image_stack - image_stack.mean()) / image_stack.std()

            image_stack = np.transpose(image_stack, axes=[1, 2, 0])
            label_raw = val_data['label']

            if not np.any(label_raw == 1) or not np.any(label_raw == 2):
                continue

            input_height = self.input_size
            input_width = self.input_size
            input_depth = self.input_d
            output_height = self.output_size
            output_width = self.output_size
            output_channels = self.output_d

            height, width, channels = image_stack.shape
            padding_l = int((input_width - output_width) / 2)
            padding_t = int((input_height - output_height) / 2)
            padding_r = int((input_width - output_width) / 2)
            padding_b = int((input_height - output_height) / 2)
            padding_s = int(math.ceil((input_depth - output_channels) / 2))
            padding_e = int(math.ceil((input_depth - output_channels) / 2))

            padded_width = width + padding_l + padding_r
            padded_height = height + padding_t + padding_b
            padded_channels = channels + padding_s + padding_e
            padded_img = np.pad(image_stack, [(padding_t, padding_b), (padding_l, padding_r), (0, 0)],
                                self.padding_type)
            padded_img = np.pad(padded_img, [(0, 0), (0, 0), (padding_s, padding_e)], self.padding_type_depth)

            start_w_index = np.array(list(range(padding_l, padded_width - padding_r - self.output_size,
                                                self.output_size)) + [padded_width - padding_r - self.output_size])
            start_h_index = np.array(list(range(padding_t, padded_height - padding_b - self.output_size,
                                                self.output_size)) + [padded_height - padding_b - self.output_size])
            start_c_index = np.array(list(range(padding_s, padded_channels - padding_e - self.output_d, self.output_d))
                                     + [padded_channels - padding_e - self.output_d])

            rows_idx = np.tile(np.repeat(start_h_index, start_w_index.shape[0]), start_c_index.shape[0])
            cols_idx = np.tile(start_w_index, start_h_index.shape[0] * start_c_index.shape[0])
            channels_idx = np.repeat(start_c_index, start_w_index.shape[0] * start_h_index.shape[0])

            start_output_idx = np.array(list(zip(rows_idx, cols_idx, channels_idx)))
            end_output_idx = start_output_idx + [self.output_size, self.output_size, self.output_d]
            start_input_idx = start_output_idx - [padding_t, padding_l, padding_s]
            end_input_idx = start_input_idx + [self.input_size, self.input_size, self.input_d]

            prediction_img = np.zeros_like(padded_img)
            prediction_prob_map = np.zeros(list(padded_img.shape) + [3])

            iterator = tqdm(zip(start_output_idx, end_output_idx, start_input_idx, end_input_idx),
                            total=start_output_idx.shape[0])
            iterator.set_description('Image {} of {}'.format(img_index + 1, num_images))
            for i, (soi, eoi, sii, eii) in enumerate(iterator):
                image_patch = padded_img[sii[0]:eii[0], sii[1]:eii[1], sii[2]:eii[2]]

                if self.scaling == 'PatchNorm':
                    min_value = image_patch.min()
                    max_value = image_patch.max()
                    image_patch = (image_patch - min_value) / (max_value - min_value)
                elif self.scaling == 'PatchSTD':
                    image_patch = (image_patch - image_patch.mean()) / image_patch.std()

                pred_patch, prob_patch = self.model.test_iteration(image_patch[None, :, :, :, None])
                prediction_img[soi[0]:eoi[0], soi[1]:eoi[1], soi[2]:eoi[2]] = pred_patch[0, :, :, :, 0]
                prediction_prob_map[soi[0]:eoi[0], soi[1]:eoi[1], soi[2]:eoi[2], :] = prob_patch[0]


            prediction_img = prediction_img[padding_t: padding_t + height, padding_l: padding_l + width,
                             padding_s:padding_s + channels]
            prediction_img = np.rollaxis(prediction_img, -1, 0)

            prediction_prob_map = prediction_prob_map[padding_t: padding_t + height, padding_l: padding_l + width,
                                  padding_s:padding_s + channels, :]
            prediction_prob_map = np.rollaxis(prediction_prob_map, 2, 0)

            io.imsave(str(self.output_image_path / 'RAW_{}.tif'.format(img_index)), val_data['raw'].astype(np.uint16),
                      check_contrast=False)
            io.imsave(str(self.output_image_path / 'LABEL_{}.tif'.format(img_index)), label_raw.astype(np.uint16),
                      check_contrast=False)
            io.imsave(str(self.output_image_path / 'PRED_{}.tif'.format(img_index)), prediction_img.astype(np.uint16),
                      check_contrast=False)
            np.save(str(self.output_image_path / 'PROB_{}.tif'.format(img_index)), prediction_prob_map.astype(np.half))

            (metrics, mean_accuracy, mean_f1, mean_iou), errors = \
                self.metrics.metrics_from_prediction(label_raw, prediction_img, w_map)

            csv_writer.writerow([img_index + 1,
                                 metrics[0]['f1'], metrics[1]['f1'], metrics[2]['f1'], mean_f1,
                                 metrics[0]['recall'], metrics[1]['recall'], metrics[2]['recall'],
                                 metrics[0]['precision'], metrics[1]['precision'], metrics[2]['precision'],
                                 metrics[0]['iou'], metrics[1]['iou'], metrics[2]['iou'], mean_iou, w_map])

            total_errors += errors

        metrics, mean_accuracy, mean_f1, mean_iou = self.metrics.metrics_from_base_metrics(total_errors)
        csv_writer.writerow(["Total",
                             metrics[0]['f1'], metrics[1]['f1'], metrics[2]['f1'], mean_f1,
                             metrics[0]['recall'], metrics[1]['recall'], metrics[2]['recall'],
                             metrics[0]['precision'], metrics[1]['precision'], metrics[2]['precision'],
                             metrics[0]['iou'], metrics[1]['iou'], metrics[2]['iou'], mean_iou, w_map])
        output_file.close()

    def infer(self, image_stack):
        image_stack = image_stack / 4095
        image_stack = np.transpose(image_stack, axes=[1, 2, 0])

        input_height = self.input_size
        input_width = self.input_size
        input_depth = self.input_d
        output_height = self.output_size
        output_width = self.output_size
        output_channels = self.output_d

        height, width, channels = image_stack.shape
        padding_l = int((input_width - output_width) / 2)
        padding_t = int((input_height - output_height) / 2)
        padding_r = int((input_width - output_width) / 2)
        padding_b = int((input_height - output_height) / 2)
        padding_s = int(math.ceil((input_depth - output_channels) / 2))
        padding_e = int(math.ceil((input_depth - output_channels) / 2))

        padded_width = width + padding_l + padding_r
        padded_height = height + padding_t + padding_b
        padded_channels = channels + padding_s + padding_e
        padded_img = np.pad(image_stack, [(padding_t, padding_b), (padding_l, padding_r), (padding_s, padding_e)],
                            mode='constant')

        start_w_index = np.array(list(range(padding_l, padded_width - padding_r - self.output_size,
                                            self.output_size)) + [padded_width - padding_r - self.output_size])
        start_h_index = np.array(list(range(padding_t, padded_height - padding_b - self.output_size,
                                            self.output_size)) + [padded_height - padding_b - self.output_size])
        start_c_index = np.array(list(range(padding_s, padded_channels - padding_e - self.output_d, self.output_d))
                                 + [padded_channels - padding_e - self.output_d])

        rows_idx = np.tile(np.repeat(start_h_index, start_w_index.shape[0]), start_c_index.shape[0])
        cols_idx = np.tile(start_w_index, start_h_index.shape[0] * start_c_index.shape[0])
        channels_idx = np.repeat(start_c_index, start_w_index.shape[0] * start_h_index.shape[0])

        start_output_idx = np.array(list(zip(rows_idx, cols_idx, channels_idx)))
        end_output_idx = start_output_idx + [self.output_size, self.output_size, self.output_d]
        start_input_idx = start_output_idx - [padding_t, padding_l, padding_s]
        end_input_idx = start_input_idx + [self.input_size, self.input_size, self.input_d]

        prediction_img = np.zeros_like(padded_img)

        iterator = tqdm(zip(start_output_idx, end_output_idx, start_input_idx, end_input_idx),
                        total=start_output_idx.shape[0])
        for i, (soi, eoi, sii, eii) in enumerate(iterator):
            image_patch = padded_img[sii[0]:eii[0], sii[1]:eii[1], sii[2]:eii[2]]
            prediction_img[soi[0]:eoi[0], soi[1]:eoi[1], soi[2]:eoi[2]] = self.model.test_iteration(
                image_patch[None, :, :, :, None])[0, :, :, :, 0]

        prediction_img = prediction_img[padding_t: padding_t + height, padding_l: padding_l + width,
                         padding_s:padding_s + channels]
        prediction_img = np.rollaxis(prediction_img, -1, 0)

        return prediction_img
