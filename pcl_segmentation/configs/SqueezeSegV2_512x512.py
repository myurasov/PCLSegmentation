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

"""Model configuration for SqueezeSeg"""
import numpy as np
from easydict import EasyDict


def SqueezeSegV2Config():
    mc = EasyDict()

    mc.CLASSES = [
        "Road",
        "Sidewalk",
        "Building",
        "Pole",
        "Vegetation",
        "Person",
        "TwoWheeler",
        "Car",
        "Truck",
        "Bus",
        "None",
    ]
    mc.NUM_CLASS = len(mc.CLASSES)
    mc.CLS_2_ID = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
    mc.CLS_LOSS_WEIGHT = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    mc.CLS_COLOR_MAP = (
        np.array(
            [
                [128, 64, 128],  # Road
                [244, 35, 232],  # Sidewalk
                [70, 70, 70],  # Building
                [153, 153, 153],  # Pole
                [107, 142, 35],  # Vegetation
                [220, 20, 60],  # Person
                [255, 0, 0],  # Two Wheeler
                [0, 0, 142],  # Car
                [0, 0, 70],  # Truck
                [0, 60, 100],  # Bus
                [0, 0, 0],  # None
            ]
        )
        / 255.0
    )

    # Input Shape
    mc.BATCH_SIZE = 8
    mc.AZIMUTH_LEVEL = 512
    mc.ZENITH_LEVEL = 512
    mc.NUM_FEATURES = 6

    # Loss
    mc.FOCAL_GAMMA = 2.0
    mc.CLS_LOSS_COEF = 15.0
    mc.DENOM_EPSILON = 1e-12  # small value used in denominator to prevent division by 0

    # Gradient Decent
    mc.LEARNING_RATE = 0.05
    mc.LR_DECAY_STEPS = 500
    mc.LR_DECAY_FACTOR = 0.9
    mc.MAX_GRAD_NORM = 100.0

    # Network
    mc.L2_WEIGHT_DECAY = 0.05
    mc.DROP_RATE = 0.1
    mc.BN_MOMENTUM = 0.9
    mc.REDUCTION = 16

    # Dataset
    mc.DATA_AUGMENTATION = True
    mc.RANDOM_FLIPPING = True
    mc.RANDOM_SHIFT = True

    # x, y, z, intensity, distance
    mc.INPUT_MEAN = np.array(
        [
            [
                [
                    0.012068249284807306,
                    0.1625781492851936,
                    0.27266734680382476,
                    3.0477590267747265,
                    4.487176079602066,
                ]
            ]
        ]
    )
    mc.INPUT_STD = np.array(
        [
            [
                [
                    10.617603530622631,
                    11.7928034735013,
                    1.3919522982269328,
                    10.174204497653436,
                    15.287492446206222,
                ]
            ]
        ]
    )

    return mc
