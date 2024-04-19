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

import logging
from pathlib import Path
from shutil import copyfile

from tensorflow.keras import mixed_precision

from configuration.parser import YAMLConfig
from data_loader.data import TFRecordDataset, TFRecordValDataset
from log.tensorboard import Log
from models.UNet import UNet3D, UNet3DDeep, VNet
from testers.tester import Tester
from trainers.trainer import Trainer
import tensorflow as tf
import numpy as np
import random
import argparse


def get_model(configuration):
    network_type = configuration.get_entry(['Network', 'type'], False) or '3DUNetDeep'

    if network_type == 'VNet':
        return VNet
    elif network_type == '3DUNetDeep':
        return UNet3DDeep
    elif network_type == '3DUNet':
        return UNet3D
    else:
        raise ValueError('Network type "{}" not supported, use one of: [VNet, 3DUNetDeep, 3DUNet]'.format(network_type))


def train(config_file_path):
    configuration = YAMLConfig(config_file_path)

    seed = configuration.get_entry(['Training', 'seed'])
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_global_generator = tf.random.Generator.from_seed(seed)

    use_mixed_precision = configuration.get_entry(['Network', 'mixed_precision'])
    if use_mixed_precision:
        mixed_precision.experimental.set_policy('mixed_float16')

    model_save_path = Path(configuration.get_entry(['Output', 'model_save_path']))
    log_path = Path(configuration.get_entry(['Output', 'log_path']))

    model_save_path.mkdir(exist_ok=True, parents=True)
    log_path.mkdir(exist_ok=True, parents=True)

    copyfile(config_file_path, log_path / 'run_config1.yaml')

    model = get_model(configuration)(configuration)

    load_model = configuration.get_entry(['Network', 'load_model'])
    if load_model:
        model_load_path = configuration.get_entry(['Output', 'model_load_path'])
        model.load_weights(model_load_path)
        print('Loaded saved model from {}'.format(model_load_path))

    log = Log(log_path)
    dataset = TFRecordDataset(configuration)
    trainer = Trainer(configuration, model, dataset, log)
    trainer.train()


def test(config_file_path):
    configuration = YAMLConfig(config_file_path)

    seed = configuration.get_entry(['Training', 'seed'])
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_global_generator = tf.random.Generator.from_seed(seed)

    use_mixed_precision = configuration.get_entry(['Network', 'mixed_precision'])
    if use_mixed_precision:
        mixed_precision.experimental.set_policy('mixed_float16')

    model = get_model(configuration)(configuration)

    model_load_path = configuration.get_entry(['Output', 'model_load_path'])
    model.load_weights(model_load_path)
    print('Loaded saved model from {}'.format(model_load_path))

    dataset = TFRecordValDataset(configuration)
    tester = Tester(configuration, model, dataset)
    results_path = Path(configuration.get_entry(['Test', 'images_output_path'])) / 'results.csv'
    if results_path.exists():
        results_path.unlink()
    tester.test(w_map=True)
    tester.test(w_map=False)


def infer(config_file_path, image):
    configuration = YAMLConfig(config_file_path)

    seed = configuration.get_entry(['Training', 'seed'])
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_global_generator = tf.random.Generator.from_seed(seed)

    use_mixed_precision = configuration.get_entry(['Network', 'mixed_precision'])
    if use_mixed_precision:
        mixed_precision.experimental.set_policy('mixed_float16')

    model = get_model(configuration)(configuration)

    model_load_path = configuration.get_entry(['Output', 'model_load_path'])
    model.load_weights(model_load_path)
    print('Loaded saved model from {}'.format(model_load_path))

    tester = Tester(configuration, model)
    tester.infer(image)


def get_weights(config_file_path):
    configuration = YAMLConfig(config_file_path)

    dataset = TFRecordDataset(configuration)
    label_count = np.array([0, 0, 0], dtype=np.int64)

    for mu in [1, 2, 3, 4, 5]:
        for i, train_data in enumerate(dataset.get_train_dataset()):
            labels = train_data[1].numpy()
            label_count[0] += (labels == 0).sum()
            label_count[1] += (labels == 1).sum()
            label_count[2] += (labels == 2).sum()

        log = np.log(mu * label_count.sum() / label_count)
        log[log < 1.0] = 1
        print('Mu: {} - values: {}'.format(mu, log / log.sum()))


if __name__ == "__main__":
    phase_choices = ['train', 'test', 'weights']
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', '--phase', required=True, choices=phase_choices, type=str, help='Configuration file path')
    parser.add_argument('-cf', '--config_file', required=True, type=str, help='Configuration file path')

    args = parser.parse_args()

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger('info')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    if args.phase == phase_choices[0]:
        train(args.config_file)
    elif args.phase == phase_choices[1]:
        test(args.config_file)
    elif args.phase == phase_choices[2]:
        get_weights(args.config_file)
