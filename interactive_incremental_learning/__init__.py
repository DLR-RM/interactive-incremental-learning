import pathlib

from interactive_incremental_learning.common.data_handler import DataHandler
from interactive_incremental_learning.common.data_object import (
    ORIENTATION_QUAT_W_FIRST,
    ORIENTATION_QUAT_W_LAST,
    ORIENTATION_SO3_MANIFOLD,
    ORIENTATION_VECTOR_6D,
    POSITION,
    DataObject,
    DataObjectType,
    InputType,
    RotationOutputType,
)
from interactive_incremental_learning.common.tp_kmp import TPKMP, TPKMPConfigParams
from interactive_incremental_learning.config import ConfigParams

__all__ = [
    "ORIENTATION_QUAT_W_FIRST",
    "ORIENTATION_QUAT_W_LAST",
    "ORIENTATION_SO3_MANIFOLD",
    "ORIENTATION_VECTOR_6D",
    "POSITION",
    "TPKMP",
    "ConfigParams",
    "DataHandler",
    "DataObject",
    "DataObjectType",
    "InputType",
    "RotationOutputType",
    "TPKMPConfigParams",
    "initialize_tpkmp",
]

import os
import pickle

import numpy as np


def initialize_tpkmp(
    params: ConfigParams, tp_kmp_params: TPKMPConfigParams | None = None
) -> tuple[TPKMP, TPKMPConfigParams, list[DataObject]]:
    """
    Initializes the TP-KMP model by loading the demonstration data and fitting the model to it.
    :param params: configuration parameters
    :param tp_kmp_params: optional TP-KMP parameters (uses defaults if None)
    :return: initialized TP-KMP model, parameters and frames
    """
    ######## Load demonstrations ########
    path_for_all_data_storage = pathlib.Path(__file__).parent.parent / "data" / "tpkmp_all_data_storage.pickle"

    if tp_kmp_params is None:
        tp_kmp_params = TPKMPConfigParams()

    if os.path.isfile(path_for_all_data_storage):
        with open(path_for_all_data_storage, "rb") as f:
            _all_data = pickle.load(f)
    else:
        raise FileNotFoundError(f"Data file not found: {path_for_all_data_storage}")

    _data = np.array(_all_data["demonstrations"])
    _frames = np.array(_all_data["frames"])
    frames_obj_list = []

    ###################################################  PREPARE DATA and MODEL
    # Initialize Datahandler
    data_handler = []
    for _idx, (demo_, frame_) in enumerate(zip(_data, _frames)):
        # We create a multi_data_handler for every demonstration with the corresponding points of interest (frames)
        frames = DataObject(data=frame_, data_structure=params.frames_data_structure, data_type=DataObjectType.ARRAY_2D)
        frames_obj_list.append(frames)
        multi_data_handler = DataHandler(
            data=demo_,
            data_structure=params.robot_data_structure,
            data_type=DataObjectType.ARRAY_2D,
            list_of_frames=frames,
        )
        data_handler.append(multi_data_handler)

    for multi_data_handler in data_handler:
        multi_data_handler.prepare_for_ml()

    # Initialize TP-KMP
    tp_kmp_model = TPKMP(
        params=tp_kmp_params,
        data_handler=data_handler,
        input_data_structure=params.training_input_data_structure,
        output_data_structure=params.training_output_data_structure,
    )

    # Fit the model
    tp_kmp_model.train()
    return tp_kmp_model, tp_kmp_params, frames_obj_list
