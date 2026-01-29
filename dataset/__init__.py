"""Dataset loaders for VeryLargeWeebModel training.

Available datasets:
- GazeboOccWorldDataset: Synthetic Tokyo PLATEAU data from Gazebo simulation
- UAVScenesDataset: Real drone dataset with LiDAR and camera
- NuScenesOccWorldDataset: NuScenes with occupancy ground truth
- NuScenes6DoFDataset: NuScenes for 6DoF world model training
- TartanAirDataset: Synthetic indoor/outdoor with perfect ground truth (optional)
- MidAirDataset: Synthetic drone dataset with multiple climates (optional)
"""

# Core datasets (always available)
from .gazebo_occworld_dataset import GazeboOccWorldDataset, DatasetConfig as GazeboDataConfig, collate_fn as gazebo_collate_fn
from .uavscenes_dataset import UAVScenesDataset, UAVScenesConfig, collate_fn as uavscenes_collate_fn
from .nuscenes_occworld_dataset import NuScenesOccWorldDataset, NuScenesConfig as NuScenesOccConfig, collate_fn as nuscenes_occ_collate_fn
from .nuscenes_6dof_dataset import NuScenes6DoFDataset, NuScenes6DoFConfig, collate_fn as nuscenes_6dof_collate_fn

# Optional datasets (may have additional dependencies)
try:
    from .tartanair_dataset import TartanAirDataset, TartanAirConfig
    _HAS_TARTANAIR = True
except ImportError:
    TartanAirDataset = None
    TartanAirConfig = None
    _HAS_TARTANAIR = False

try:
    from .midair_dataset import MidAirDataset, MidAirConfig
    _HAS_MIDAIR = True
except ImportError:
    MidAirDataset = None
    MidAirConfig = None
    _HAS_MIDAIR = False

__all__ = [
    # Datasets
    'GazeboOccWorldDataset',
    'UAVScenesDataset',
    'NuScenesOccWorldDataset',
    'NuScenes6DoFDataset',
    # Configs
    'GazeboDataConfig',
    'UAVScenesConfig',
    'NuScenesOccConfig',
    'NuScenes6DoFConfig',
    # Collate functions
    'gazebo_collate_fn',
    'uavscenes_collate_fn',
    'nuscenes_occ_collate_fn',
    'nuscenes_6dof_collate_fn',
]

# Add optional datasets to __all__ if available
if _HAS_TARTANAIR:
    __all__.extend(['TartanAirDataset', 'TartanAirConfig'])

if _HAS_MIDAIR:
    __all__.extend(['MidAirDataset', 'MidAirConfig'])
