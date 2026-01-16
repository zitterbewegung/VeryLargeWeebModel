#!/usr/bin/env python3
"""
Simple PLATEAU to Gazebo converter - no mesh simplification.

This script converts PLATEAU OBJ files to Gazebo SDF format without
mesh simplification, avoiding dependency issues with fast_simplification.

Usage:
    python scripts/convert_plateau_simple.py

    # Or with options:
    python scripts/convert_plateau_simple.py --max-models 100 --input data/plateau/meshes/obj
"""
import os
import sys
import argparse
from pathlib import Path

try:
    import trimesh
    import numpy as np
except ImportError:
    print("Error: trimesh not installed. Run: pip install trimesh numpy")
    sys.exit(1)


def create_sdf_model(name: str, mesh_path: str, output_dir: str) -> str:
    """Create a Gazebo SDF model from an OBJ mesh (no simplification)."""

    model_dir = os.path.join(output_dir, name)
    meshes_dir = os.path.join(model_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    try:
        # Load mesh without simplification
        mesh = trimesh.load(mesh_path, force='mesh')

        # Export as DAE (Collada) for Gazebo
        output_mesh = os.path.join(meshes_dir, f"{name}.dae")
        mesh.export(output_mesh)
        mesh_file = f"meshes/{name}.dae"

    except Exception as e:
        print(f"  Error: {e}")
        return None

    # Create model.config
    config_content = f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author>
    <name>PLATEAU/OccWorld</name>
  </author>
  <description>Tokyo PLATEAU building model for Gazebo simulation.</description>
</model>
"""
    with open(os.path.join(model_dir, "model.config"), "w") as f:
        f.write(config_content)

    # Create model.sdf
    sdf_content = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <mesh>
            <uri>model://{name}/{mesh_file}</uri>
          </mesh>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>model://{name}/{mesh_file}</uri>
          </mesh>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.7 0.7 0.7 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
    with open(os.path.join(model_dir, "model.sdf"), "w") as f:
        f.write(sdf_content)

    return model_dir


def create_world_file(models_dir: str, output_file: str, world_name: str = "tokyo_plateau"):
    """Create a Gazebo world file including all converted models."""

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    model_dirs = [d for d in os.listdir(models_dir)
                  if os.path.isdir(os.path.join(models_dir, d))]

    if not model_dirs:
        print(f"Warning: No models found in {models_dir}")
        return

    # Generate model includes with grid positioning
    includes = []
    grid_size = int(np.ceil(np.sqrt(len(model_dirs))))
    spacing = 50  # meters between models

    for i, model_name in enumerate(model_dirs):
        x = (i % grid_size) * spacing
        y = (i // grid_size) * spacing
        includes.append(f"""
    <include>
      <uri>model://{model_name}</uri>
      <name>{model_name}</name>
      <pose>{x} {y} 0 0 0 0</pose>
    </include>""")

    world_content = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{world_name}">
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>

    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>35.6762</latitude_deg>
      <longitude_deg>139.6503</longitude_deg>
      <elevation>40</elevation>
    </spherical_coordinates>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 100 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane><normal>0 0 1</normal><size>2000 2000</size></plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane><normal>0 0 1</normal><size>2000 2000</size></plane>
          </geometry>
        </visual>
      </link>
    </model>
    {"".join(includes)}
  </world>
</sdf>
"""

    with open(output_file, "w") as f:
        f.write(world_content)

    print(f"Created world file: {output_file} ({len(model_dirs)} models)")


def main():
    parser = argparse.ArgumentParser(description="Convert PLATEAU OBJ to Gazebo SDF (simple)")
    parser.add_argument("--input", "-i", default="data/plateau/meshes/obj",
                        help="Input OBJ directory")
    parser.add_argument("--output", "-o", default="data/plateau/gazebo_models",
                        help="Output model directory")
    parser.add_argument("--max-models", "-m", type=int, default=50,
                        help="Max models to convert (0=all)")
    parser.add_argument("--world", "-w", default="worlds/tokyo_plateau.sdf",
                        help="Output world file path")
    parser.add_argument("--no-world", action="store_true",
                        help="Skip world file generation")
    args = parser.parse_args()

    # Find OBJ files
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)

    obj_files = list(input_path.rglob("*.obj"))
    print(f"Found {len(obj_files)} OBJ files")

    if args.max_models > 0:
        obj_files = obj_files[:args.max_models]
        print(f"Processing first {args.max_models} models")

    # Convert models
    os.makedirs(args.output, exist_ok=True)
    created = 0

    for i, obj_file in enumerate(obj_files):
        name = f"plateau_{obj_file.stem.replace(' ', '_').replace('-', '_')}"
        print(f"[{i+1}/{len(obj_files)}] Converting: {name}")

        result = create_sdf_model(name, str(obj_file), args.output)
        if result:
            created += 1

    print(f"\nCreated {created} Gazebo models in {args.output}")

    # Create world file
    if not args.no_world and created > 0:
        create_world_file(args.output, args.world)


if __name__ == "__main__":
    main()
