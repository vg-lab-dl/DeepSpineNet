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

import gc
import logging
import sys

# import cv2
# import cv2
import cv2
from skimage import io
from tensorflow.keras import mixed_precision
from tensorflow.python.keras import Model
from tensorflow_addons.optimizers import AdamW
from tqdm import tqdm

from configuration.parser import YAMLConfig
from data_loader.data import TFRecordDataset
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD
import os
import tensorflow as tf
import numpy as np

from data_loader.preprocessing import PaddedStackPatcher
from log.tensorboard import Log
from metrics.metrics import Metrics

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Trainer:
    EPOCH_PHASE = 0
    ITERATION_PHASE = 1

    def __init__(self, configuration: YAMLConfig, model: Model, dataset: TFRecordDataset, log: Log):
        self.configuration = configuration
        self.optimizer_name = configuration.get_entry(['Training', 'optimizer'])
        self.validate_iterations = configuration.get_entry(['Training', 'validate_iterations'])
        self.num_classes = configuration.get_entry(['Data', 'num_classes'])
        self.num_epochs = configuration.get_entry(['Training', 'epochs'])
        self.d_weight = configuration.get_entry(['Data', 'd_weight'])
        self.model_save_path = configuration.get_entry(['Output', 'model_save_path'])
        self.convergence_iterations = configuration.get_entry(['Training', 'convergence_iterations'])
        self.learning_rate_type = configuration.get_entry(['Training', 'learning_rate_type'])
        self.use_mixed_precision = configuration.get_entry(['Network', 'mixed_precision'])

        self.logger = logging.getLogger('info')

        if self.optimizer_name == 'Adam':
            optimizer_class = Adam
        elif self.optimizer_name == 'AdamW':
            optimizer_class = AdamW
        elif self.optimizer_name == 'Adagrad':
            optimizer_class = Adagrad
        elif self.optimizer_name == 'RMSProp':
            optimizer_class = RMSprop
        elif self.optimizer_name == 'SGD':
            optimizer_class = SGD
        else:
            raise ValueError('Optimizer type not found: {}'.format(self.optimizer_name))

        if self.learning_rate_type == 'Exponential':
            self.learning_rate = configuration.get_entry(['Training', 'learning_rate'])
            self.decay_steps = configuration.get_entry(['Training', 'decay_steps'])
            self.decay_rate = configuration.get_entry(['Training', 'decay_rate'])
            self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate,
                                                                                decay_steps=self.decay_steps,
                                                                                decay_rate=self.decay_rate,
                                                                                staircase=False)
        elif self.learning_rate_type == 'Step':
            self.lr_bounds = configuration.get_entry(['Training', 'lr_bounds'])
            self.lr_values = configuration.get_entry(['Training', 'lr_values'])
            self.learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=self.lr_bounds,
                                                                                      values=self.lr_values)

        else:
            self.learning_rate = configuration.get_entry(['Training', 'learning_rate'])

        if self.optimizer_name == 'AdamW':
            self.weight_decay = configuration.get_entry(['Training', 'weight_decay'])
            self.optimizer = optimizer_class(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.optimizer = optimizer_class(learning_rate=self.learning_rate)

        if self.use_mixed_precision:
            self.optimizer = mixed_precision.experimental.LossScaleOptimizer(self.optimizer, 'dynamic')

        self.trainingIterator = dataset.get_train_dataset()
        self.validationIterator = dataset.get_val_dataset()

        self.model = model
        self.model.compile(self.optimizer)

        self.metrics = Metrics(configuration)

        self.log = log
        self.num_train_examples = 0
        self.best_val_f1_scores = [0, 0, 0]
        self.best_val_epoch = 0

    def __reset_metrics(self, phase):
        self.num_val_steps = 0
        self.val_running_loss = 0
        self.val_errors = np.zeros((self.num_classes, 4), dtype=np.int64)

        if phase == Trainer.EPOCH_PHASE:
            self.num_train_steps = 0
            self.train_running_loss = 0
            self.train_errors = np.zeros((self.num_classes, 4), dtype=np.int64)

    def __log_metrics(self, phase, step):
        phase_text = "Iteration" if phase == Trainer.ITERATION_PHASE else "Epoch"

        train_metrics = self.metrics.metrics_from_base_metrics(self.train_errors)
        val_metrics = self.metrics.metrics_from_base_metrics(self.val_errors)

        train_loss = self.train_running_loss / self.num_train_steps
        val_loss = self.val_running_loss / self.num_val_steps
        self.log.write_scalar(Log.TRAIN, '{}_loss'.format(phase_text), train_loss, step)
        self.log.write_scalar(Log.VAL, '{}_loss'.format(phase_text), val_loss, step)

        for log_type, metrics in zip([Log.TRAIN, Log.VAL], [train_metrics, val_metrics]):
            metrics, mean_accuracy, mean_f1, mean_iou = metrics
            for c in range(self.num_classes):
                self.log.write_scalar(log_type, '{}_tp/c{}'.format(phase_text, c), metrics[c]['tp'], step)
                self.log.write_scalar(log_type, '{}_fp/c{}'.format(phase_text, c), metrics[c]['fp'], step)
                self.log.write_scalar(log_type, '{}_fn/c{}'.format(phase_text, c), metrics[c]['fn'], step)
                self.log.write_scalar(log_type, '{}_tn/c{}'.format(phase_text, c), metrics[c]['tn'], step)
                self.log.write_scalar(log_type, '{}_accuracy/c{}'.format(phase_text, c), metrics[c]['accuracy'], step)
                self.log.write_scalar(log_type, '{}_precision/c{}'.format(phase_text, c), metrics[c]['precision'], step)
                self.log.write_scalar(log_type, '{}_recall/c{}'.format(phase_text, c), metrics[c]['recall'], step)
                self.log.write_scalar(log_type, '{}_f1/c{}'.format(phase_text, c), metrics[c]['f1'], step)
                self.log.write_scalar(log_type, '{}_IoU/c{}'.format(phase_text, c), metrics[c]['iou'], step)

            self.log.write_scalar(log_type, '{}_f1/mean'.format(phase_text), mean_f1, step)
            self.log.write_scalar(log_type, '{}_IoU/mean'.format(phase_text), mean_iou, step)

        self.logger.info(
            '{} {:04d} - Train loss {:.08f} - Validation loss: {:.08f}\n'.format(phase_text, step, train_loss,
                                                                                 val_loss))
        sys.stdout.flush()

        return train_metrics, val_metrics

    def train(self):
        val_message = 'Validation - improved {} F1-Score: {:.04f} at epoch: {:02d}'
        for epoch in range(self.num_epochs):
            self.__reset_metrics(Trainer.EPOCH_PHASE)
            for i, patch_data in enumerate(self.trainingIterator):
                self.num_train_steps += 1
                global_step = (epoch * self.num_train_examples) + (self.num_train_steps - 1)

                gradients, loss, predictions = self.model.train_iteration(patch_data, self.optimizer)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                errors = self.metrics.base_metrics_from_prediction(patch_data[1].numpy(), predictions, False)

                self.train_errors += errors
                self.train_running_loss += loss

                self.logger.info('Training - iteration {:04d}: {:.08f}'.format((self.num_train_steps - 1), loss))
                sys.stdout.flush()

                if self.num_train_steps % self.validate_iterations == 0:
                    for val_patch_data in self.validationIterator:
                        loss, predictions = self.model.validation_iteration(val_patch_data)
                        errors = self.metrics.base_metrics_from_prediction(val_patch_data[1].numpy(), predictions)

                        self.val_running_loss += loss
                        self.val_errors += errors
                        self.num_val_steps += 1

                    self.__log_metrics(Trainer.ITERATION_PHASE, global_step)
                    self.__reset_metrics(Trainer.ITERATION_PHASE)

            for val_patch_data in self.validationIterator:
                loss, predictions = self.model.validation_iteration(val_patch_data)
                errors = self.metrics.base_metrics_from_prediction(val_patch_data[1].numpy(), predictions)

                self.val_running_loss += loss
                self.val_errors += errors
                self.num_val_steps += 1

            _, val_metrics = self.__log_metrics(Trainer.EPOCH_PHASE, epoch + 1)

            self.num_train_examples = self.num_train_steps

            if self.best_val_f1_scores[0] < val_metrics[0][1]['f1']:
                self.best_val_f1_scores[0] = val_metrics[0][1]['f1']
                self.logger.info(
                    val_message.format("dendrites", self.best_val_f1_scores[0], epoch + 1))
                self.model.save_weights(self.model_save_path + '/best_dendrite/')

            if self.best_val_f1_scores[1] < val_metrics[0][2]['f1']:
                self.best_val_f1_scores[1] = val_metrics[0][2]['f1']
                self.logger.info(val_message.format("spines", self.best_val_f1_scores[1], epoch + 1))
                self.model.save_weights(self.model_save_path + '/best_spine/')

            if self.best_val_f1_scores[2] < val_metrics[2]:
                self.best_val_f1_scores[2] = val_metrics[2]
                self.best_val_epoch = (self.num_train_steps - 1)
                self.logger.info(val_message.format("overall", self.best_val_f1_scores[2], epoch + 1))
                self.model.save_weights(self.model_save_path + '/best/')

            if epoch - self.best_val_epoch >= self.convergence_iterations:
                self.model.save_weights(self.model_save_path + '/')
                return
