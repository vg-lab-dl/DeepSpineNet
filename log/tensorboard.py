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


class Log:
    TRAIN = 0
    VAL = 1
    TEST = 2

    def __init__(self, log_path):
        self.train_summary_writer = tf.summary.create_file_writer(str(log_path / 'train'))
        self.val_summary_writer = tf.summary.create_file_writer(str(log_path / 'val'))

    def write_scalar(self, writer, key, value, step):
        if writer == Log.TRAIN:
            writer = self.train_summary_writer
        elif writer == Log.VAL:
            writer = self.val_summary_writer

        with writer.as_default():
            tf.summary.scalar(key, value, step=step)

        writer.flush()

    def write_image_stack(self, writer, stack, step):
        if writer == Log.TRAIN:
            prefix = 'Train'
            writer = self.train_summary_writer
        elif writer == Log.VAL:
            prefix = 'Validation'
            writer = self.val_summary_writer

        with writer.as_default():
            tf.summary.image('{} example'.format(prefix), stack, step=step)

    def close_writers(self):
        self.train_summary_writer.close()
        self.val_summary_writer.close()
