# ==============================================================================
# MIT License
#
# Copyright 2021 Institute for Automotive Engineering of RWTH Aachen University.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


import argparse
import os.path
import shlex

import tensorflow as tf

from data_loader import DataLoader
from utils.args_loader import load_model_config
from utils.callbacks import TensorBoard
from utils.util import *

# arguments for debugging

debug_args = (
    '--data_path="/app/_data/pandaset_converted_256x256" '
    + '--checkpoints_dir="/app/_data/output" '
    + '--tensorboard_dir="/app/.tensorboard" '
    + "--epochs=5 --model=squeezesegv2"
)


def train(arg):
    config, model = load_model_config(args.model)

    train = (
        DataLoader("train", arg.data_path, config)
        .write_tfrecord_dataset()
        .read_tfrecord_dataset()
    )
    val = (
        DataLoader("val", arg.data_path, config)
        .write_tfrecord_dataset()
        .read_tfrecord_dataset()
    )

    tensorboard_callback = TensorBoard(
        arg.tensorboard_dir, val, profile_batch=(95, 100)
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(arg.checkpoints_dir, "checkpoint")
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.LEARNING_RATE,
        decay_steps=config.LR_DECAY_STEPS,
        decay_rate=config.LR_DECAY_FACTOR,
        staircase=True,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule, clipnorm=config.MAX_GRAD_NORM
    )

    model.compile(optimizer=optimizer)

    model.fit(
        train,
        validation_data=val,
        epochs=arg.epochs,
        callbacks=[tensorboard_callback, checkpoint_callback],
    )

    model.save(filepath=os.path.join(arg.checkpoints_dir, "model"))


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    parser = argparse.ArgumentParser(description="Parse Flags for the training script!")
    parser.add_argument(
        "-d", "--data_path", type=str, help="Absolute path to the dataset"
    )
    parser.add_argument(
        "-n", "--net", type=str, default="squeezeSeg", help="Network architecture"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=50, help="Maximal number of training epochs"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        help="Directory where to write the checkpoints",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        help="TensorBoard directory",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model name either `squeezesegv2`, `darknet53`, `darknet21`",
    )
    args = parser.parse_args()

    if args.model is None:
        if debug_args is not None:
            args = parser.parse_args(shlex.split(debug_args))

    train(args)
