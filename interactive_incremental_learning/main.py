#!/usr/bin/env python

################################################################################
# Copyright (c) 2025. Markus Knauer, Joao Silverio                             #
# Licensed under the MIT License. See LICENSE file for details.                 #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 04-03-2025                                                             #
# Author: Markus Knauer                                                        #
# E-mail: markus.knauer@dlr.de                                                 #
# Website: https://github.com/DLR-RM/interactive-incremental-learning          #
################################################################################

"""
Script to run all the experiments

If you are using this code please cite us:
M. Knauer, A. Albu-Schäffer, F. Stulp and J. Silvério, "Interactive Incremental Learning of Generalizable
Skills With Local Trajectory Modulation," in IEEE Robotics and Automation Letters (RA-L), vol. 10, no. 4,
pp. 3398-3405, April 2025, doi: 10.1109/LRA.2025.3542209

See CITATION.bib for the bib file!
"""

__author__ = "Markus Knauer, Joao Silverio"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import logging

from interactive_incremental_learning import ConfigParams, initialize_tpkmp
from interactive_incremental_learning.experiments.adding_frames import AddFramesExperiment
from interactive_incremental_learning.experiments.adding_via_points import AddViaPointsExperiment
from interactive_incremental_learning.experiments.calculate_variable_stiffness import CalculateVariableStiffnessExperiment
from interactive_incremental_learning.experiments.generalization import GeneralizationExperiment

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Runs the interactive_incremental_learning approach")
    parser.add_argument(
        "--experiment",
        help="For Generalization 0, Adding Via Points 1, Adding Frames 2, Calculate Variable Stiffness 3",
        default="0123",
    )
    parser.add_argument("--plot", help="This activates the plotting", default=False, action="store_true")
    args = parser.parse_args()
    valid_experiments = {"0", "1", "2", "3"}
    if not any(exp in args.experiment for exp in valid_experiments):
        raise ValueError(f"You must use 0, 1, 2 and/or 3 as experiment, not {args.experiment}")

    ################################################### Initialize Model
    params = ConfigParams()
    tp_kmp_model, tp_kmp_params, frames = initialize_tpkmp(params=params)

    ################################################### RUN Generalization EXPERIMENT
    if "0" in args.experiment:
        pred_frames_obj, tp_kmp_model = GeneralizationExperiment().run(
            tp_kmp_params=tp_kmp_params,
            tp_kmp_model=tp_kmp_model,
            frames_used_for_demonstrations=frames,
            params=params,
            plot=args.plot,
        )

    ################################################### RUN Adding Via-Point EXPERIMENT
    if "1" in args.experiment:
        pred_frames_obj, tp_kmp_model = AddViaPointsExperiment().run(
            tp_kmp_model=tp_kmp_model,
            tp_kmp_params=tp_kmp_params,
            frames_used_for_demonstrations=frames,
            plot=args.plot,
            params=params,
        )

    ################################################### RUN Adding Frame EXPERIMENT
    if "2" in args.experiment:
        pred_frames_obj, tp_kmp_model = AddFramesExperiment().run(
            tp_kmp_model=tp_kmp_model,
            tp_kmp_params=tp_kmp_params,
            frames_used_for_demonstrations=frames,
            params=params,
            plot=args.plot,
        )

    if "3" in args.experiment:
        ################################################### RUN Calculate variable Stiffness EXPERIMENT
        CalculateVariableStiffnessExperiment().run(
            tp_kmp_model=tp_kmp_model,
            tp_kmp_params=tp_kmp_params,
            frames_used_for_demonstrations=frames,
            params=params,
            plot=args.plot,
        )

    logging.info("done")
