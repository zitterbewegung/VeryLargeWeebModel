"""Dataset loaders for VeryLargeWeebModel training.

Available datasets:
- GazeboOccWorldDataset: Synthetic Tokyo PLATEAU data from Gazebo simulation
- UAVScenesDataset: Real drone dataset with LiDAR and camera
- NuScenesOccWorldDataset: NuScenes with occupancy ground truth
- NuScenes6DoFDataset: NuScenes for 6DoF world model training
- TartanAirDataset: Synthetic indoor/outdoor with perfect ground truth
- MidAirDataset: Synthetic drone dataset with multiple climates
"""

from .gazebo_occworld_dataset import GazeboOccWorldDataset, GazeboDataConfig, collate_fn as gazebo_collate_fn
from .uavscenes_dataset import UAVScenesDataset, UAVScenesConfig, collate_fn as uavscenes_collate_fn
from .nuscenes_occworld_dataset import NuScenesOccWorldDataset, NuScenesOccConfig, collate_fn as nuscenes_occ_collate_fn
from .nuscenes_6dof_dataset import NuScenes6DoFDataset, NuScenes6DoFConfig, collate_fn as nuscenes_6dof_collate_fn
from .tartanair_dataset import TartanAirDataset, TartanAirConfig
from .midair_dataset import MidAirDataset, MidAirConfig

__all__ = [
    # Datasets
    'GazeboOccWorldDataset',
    'UAVScenesDataset',
    'NuScenesOccWorldDataset',
    'NuScenes6DoFDataset',
    'TartanAirDataset',
    'MidAirDataset',
    # Configs
    'GazeboDataConfig',
    'UAVScenesConfig',
    'NuScenesOccConfig',
    'NuScenes6DoFConfig',
    'TartanAirConfig',
    'MidAirConfig',
    # Collate functions
    'gazebo_collate_fn',
    'uavscenes_collate_fn',
    'nuscenes_occ_collate_fn',
    'nuscenes_6dof_collate_fn',
]
