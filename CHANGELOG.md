# Changelog

## [Unreleased] - 2026-02-06

### Bug Fixes

#### Training Pipeline (`train.py`)
- **Critical**: Move NaN/Inf loss check BEFORE `loss.backward()` and `optimizer.step()` to prevent corrupting model weights with NaN gradients
- **Critical**: Move `optimizer.zero_grad()` before batch validation to prevent stale gradients accumulating from skipped batches
- Add batch key validation in `train_epoch()` and `validate()` — skips batches missing required keys instead of crashing
- Add division-by-zero guard: `total_loss / max(num_batches, 1)`

#### Data Loading
- **`gazebo_occworld_dataset.py`**: Change occupancy tensor from `.long()` to `.float()` for type consistency
- **`gazebo_occworld_dataset.py`**: Add empty point cloud guard before subsampling
- **`midair_dataset.py`**: Add `__del__` method to close cached HDF5 file handles (prevents resource leaks)
- **`uavscenes_dataset.py`**: Fix ego-frame transform (`@ R` to `@ R.T`) for correct world-to-ego rotation
- **`uavscenes_dataset.py`**: Use `.copy()` for velocity computation to avoid in-place mutation of source data
- **`uavscenes_dataset.py`**: Use `np.round` instead of truncation for grid size calculation
- **`uavscenes_dataset.py`**: Clip voxel coordinates before int conversion to prevent int32 overflow on out-of-range points

#### CLI (`scripts/vlwm_cli.py`)
- **Critical**: Fix `cmd_train()` to pass correct arguments matching `train.py`'s argparse interface (was passing invalid `--cfg-options` and `--launcher`)
- Fix Python 3.10+ syntax (`list[str] | None` to plain parameters) for Python 3.8/3.9 compatibility
- Add subprocess return code check for pip install

#### Utilities
- **`voxel_config.py`**: Add zero-division validation for voxel_size
- **`system_packages.py`**: Fix Python 3.10+ type hints for backward compatibility
- **`system_packages.py`**: Add regex validation for package names to prevent command injection

### Security
- Add `shlex.quote()` for SSH commands in deploy functionality
- Add regex-based package name validation in system package installer

### New Features
- **Unified CLI** (`scripts/vlwm_cli.py`): Single entry point with subcommands — `setup`, `download`, `train`, `deploy`, `sanity`, `info`
- **GPU utilities** (`scripts/utils/gpu.py`): Auto-detection, batch size selection, precision selection
- **Environment detection** (`scripts/utils/environment.py`): Auto-detect Vast.ai, Lambda, RunPod cloud providers
- **Download utilities** (`scripts/utils/download.py`): Fast downloads with aria2c/axel/curl fallback

### Tests
- Add `tests/test_comprehensive.py` with 73 new tests covering:
  - All model components: TemporalLSTM, TemporalTransformer, UncertaintyHead, RelocalizationHead, PlaceRecognitionHead, FuturePoseRNN
  - Training functions: batch validation, NaN loss skipping (verifies `optimizer.step()` not called), empty dataloader handling
  - Data validation: `validate_data`, `validate_pretrained_models`
  - VoxelConfig edge cases (zero/negative voxel size)
  - UAVScenes: grid size, voxelization clipping, ego transforms, velocity no-mutation
  - Dataset loading: occupancy dtype, pose shape, lidar subsampling, collate_fn
  - CLI subcommands and parser completeness
  - Loss function components (focal loss, mean-matching, debug counter)
  - SimpleOccupancyModel (encoder shape, forward pass, output range)
  - MidAir HDF5 cache cleanup
  - GPU detection edge cases, environment detection
  - Mini training integration tests (loss decreases over epochs)
- Add `tests/test_cli.py` with CLI utility tests
- Add `tests/test_shell_scripts.py` with shell script validation
- Total test count: 323 passing, 2 expected skips

### Documentation
- Rewrite `README.md` as concise project overview with AerialWorld identity
- Consolidate 5 duplicate training guides into single `docs/TRAINING.md`
- Rename `training.md` to `TRAINING_LOG.md`
- Remove empty `prompts.md` and duplicate `HOW_TO_TRAIN.md`
- Update `docs/README.md` index

### Paper
- Add 11 new citations to related work (UrbanWorld, OccSora, LidarDM, etc.)
