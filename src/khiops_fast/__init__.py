######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Interface scikit-learn de Khiops, copie de ``khiops.sklearn`` (khiops 11.0.1.0).

Les estimateurs seront modifiés pour prédire et transformer avec khiops-kni au lieu du runner Khiops.
"""
from khiops_fast.estimators import (
    KhiopsFastClassifier,
    KhiopsFastEncoder,
    KhiopsFastRegressor,
)
from khiops_fast.helpers import train_test_split_dataset

__all__ = ["KhiopsFastClassifier", "KhiopsFastEncoder", "KhiopsFastRegressor", "train_test_split_dataset"]
__version__ = "0.1.0"
