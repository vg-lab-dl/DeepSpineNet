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

import tensorflow as tf
import numpy as np
from scipy.ndimage import distance_transform_edt
from tensorflow.python.framework.ops import EagerTensor

from configuration.parser import YAMLConfig


class Metrics:
    """
    Class for segmentation metrics.

    Use metrics_from_prediction to compute metrics (F1-Score, IoU, ...) and base metrics (TP, FP, FN, TN) from an image
    Use metrics_from_base_metrics to compute metrics (F1-Score, IoU, ...) from base metrics (TP, FP, FN, TN)

    The second output of metrics_from_prediction, the base metrics, shall be added and passed to
    metrics_from_base_metrics in order to get the overall metrics of the classifier:


    metrics = Metrics()

    metrics1, base_metrics1 = metrics.metrics_from_prediction(labels1, predictions1)
    metrics2, base_metrics2 = metrics.metrics_from_prediction(labels2, predictions2)
    metrics3, base_metrics3 = metrics.metrics_from_prediction(labels3, predictions3)

    total_base_metrics = base_metrics1 + base_metrics2 + base_metrics3

    overall_metrics, overall_base_metrics = metrics.metrics_from_base_metrics(total_base_metrics)
    """

    def __init__(self, configuration: YAMLConfig):
        self.num_classes = configuration.get_entry(['Data', 'num_classes'])
        self.spacing = [0.279911, 0.0751562, 0.0751562]
        self.spine_distance_th = configuration.get_entry(['Test', 'spine_distance_threshold'], False) or 1.0
        self.dendrite_distance_th = configuration.get_entry(['Test', 'dendrite_distance_threshold'], False) or 5.0

    def __get_distance_mask(self, label_image):
        """
        Calculates a distance mask based on the content of the image, distance from dendrite and spine
        :param label_image: label image from which to compute the distance mask
        :return: an binary ndarray mask
        """
        mask_dend = label_image != 2
        mask_spine = label_image != 1
        distance_dend = distance_transform_edt(mask_dend, sampling=self.spacing)
        distance_spine = distance_transform_edt(mask_spine, sampling=self.spacing)
        return np.logical_or(distance_dend <= self.spine_distance_th, distance_spine <= self.dendrite_distance_th)

    def __get_basic_metrics_tf(self, labels, predictions, use_distance_map):
        """
        Calculates TP FP FN TN between the GT and the predictions
        Labels must be a number between [0, self.num_classes - 1]
        :param labels: tf EagerTensor with the GT
        :param predictions: tf EagerTensor with the network prediction
        :return: an ndarray with the base metrics of each class as rows where columns are TP FP FN TN
        """
        mask = None
        if use_distance_map:
            mask = self.__get_distance_mask(labels)

        base_metrics = []
        for class_i in range(self.num_classes):
            if use_distance_map:
                tp = tf.math.count_nonzero(
                    tf.logical_and(tf.logical_and(labels == class_i, predictions == class_i), mask)).numpy()
                fp = tf.math.count_nonzero(
                    tf.logical_and(tf.logical_and(labels != class_i, predictions == class_i), mask)).numpy()
                fn = tf.math.count_nonzero(
                    tf.logical_and(tf.logical_and(labels == class_i, predictions != class_i), mask)).numpy()
                tn = tf.math.count_nonzero(
                    tf.logical_and(tf.logical_and(labels != class_i, predictions != class_i), mask)).numpy()
            else:
                tp = tf.math.count_nonzero(tf.logical_and(labels == class_i, predictions == class_i)).numpy()
                fp = tf.math.count_nonzero(tf.logical_and(labels != class_i, predictions == class_i)).numpy()
                fn = tf.math.count_nonzero(tf.logical_and(labels == class_i, predictions != class_i)).numpy()
                tn = tf.math.count_nonzero(tf.logical_and(labels != class_i, predictions != class_i)).numpy()
            base_metrics.append([tp, fp, fn, tn])

        return np.array(base_metrics)

    def __get_basic_metrics_np(self, labels, predictions, use_distance_map):
        """
        Calculates TP FP FN TN between the GT and the predictions
        Labels must be a number between [0, self.num_classes - 1]
        :param labels: ndarray with the GT
        :param predictions: ndarray with the network prediction
        :return: an ndarray with the base metrics of each class as rows and TP FP FN TN as columns
        """
        mask = None
        if use_distance_map:
            mask = self.__get_distance_mask(labels)

        base_metrics = []
        for class_i in range(self.num_classes):
            if use_distance_map:
                tp = np.logical_and(np.logical_and(labels == class_i, predictions == class_i), mask).sum()
                fp = np.logical_and(np.logical_and(labels != class_i, predictions == class_i), mask).sum()
                fn = np.logical_and(np.logical_and(labels == class_i, predictions != class_i), mask).sum()
                tn = np.logical_and(np.logical_and(labels != class_i, predictions != class_i), mask).sum()
            else:
                tp = np.logical_and(labels == class_i, predictions == class_i).sum()
                fp = np.logical_and(labels != class_i, predictions == class_i).sum()
                fn = np.logical_and(labels == class_i, predictions != class_i).sum()
                tn = np.logical_and(labels != class_i, predictions != class_i).sum()
            base_metrics.append([tp, fp, fn, tn])

        return np.array(base_metrics)

    def metrics_from_base_metrics(self, base_metrics):
        """
        Calculates the metrics of accuracy, precision, recall, F1-Score, IoU, TP, FP, FN, TN
        :param base_metrics: an ndarray with the errors of each class as rows where columns are TP FP FN TN
        :return: a tuple of metrics: (per class metrics, mean accuracy, mean F1-Score, mean IoU)
        """
        metrics = []
        for tp, fp, fn, tn in base_metrics:
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (2 * precision * recall) / (precision + recall)
            iou = tp / (tp + fp + fn)
            metrics.append(dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1, iou=iou,
                                tp=tp, fp=fp, fn=fn, tn=tn))

        mean_accuracy = (metrics[1]['accuracy'] + metrics[2]['accuracy']) * 0.5
        mean_f1 = (metrics[1]['f1'] + metrics[2]['f1']) * 0.5
        mean_iou = (metrics[1]['iou'] + metrics[2]['iou']) * 0.5

        return metrics, mean_accuracy, mean_f1, mean_iou

    def base_metrics_from_prediction(self, labels, predictions, use_distance_map=False):
        """
        Calculates the metrics of TP, FP, FN, TN
        :param labels: tensor with the GT
        :param predictions: tensor with the network prediction
        :param use_distance_map: compute metrics discarding part of the image based on a distance map
        :return: a tuple of metrics: ((per class metrics, mean accuracy, mean F1-Score, mean IoU), base_metrics)
        """
        if type(labels) == type(predictions):
            if isinstance(labels, np.ndarray):
                base_metrics = self.__get_basic_metrics_np(labels, predictions, use_distance_map)
            elif isinstance(labels, EagerTensor):
                base_metrics = self.__get_basic_metrics_tf(labels, predictions, use_distance_map)
            else:
                raise ValueError('Not supported data type found: {}'.format(type(labels)))
        else:
            raise ValueError('Labels and predictions must be '
                             'of the same type, found: {} and {}'.format(type(labels), type(predictions)))

        return base_metrics

    def metrics_from_prediction(self, labels, predictions, use_distance_map=False):
        """
        Calculates the metrics of accuracy, precision, recall, F1-Score, IoU, TP, FP, FN, TN
        :param labels: tensor with the GT
        :param predictions: tensor with the network prediction
        :param use_distance_map: compute metrics discarding part of the image based on a distance map
        :return: a tuple of metrics: ((per class metrics, mean accuracy, mean F1-Score, mean IoU), base_metrics)
        """
        base_metrics = self.base_metrics_from_prediction(labels, predictions, use_distance_map)
        return self.metrics_from_base_metrics(base_metrics), base_metrics
