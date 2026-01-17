# AerialWorld: Occupancy World Models for Urban Aerial Navigation

Research paper describing the VeryLargeWeebModel project.

## Compile

```bash
# Using make
make

# Or directly
pdflatex main.tex
pdflatex main.tex  # Run twice for references
```

## View

```bash
make view
# or
open main.pdf
```

## Clean

```bash
make clean
```

## Abstract

World models have emerged as a powerful paradigm for autonomous driving, enabling vehicles to predict future scene occupancy and plan safe trajectories. However, existing occupancy world models focus exclusively on ground vehicles, leaving aerial navigation unexplored. We present AerialWorld, the first occupancy world model designed for unmanned aerial vehicles (UAVs) navigating urban environments.

## Key Contributions

1. First occupancy world model for aerial urban navigation
2. Data generation pipeline using PLATEAU 3D city data (CC BY 4.0)
3. Multi-agent training (drone + rover perspectives)
4. Aggressive augmentation for static geometry
5. Full open-source release

## Citation

```bibtex
@article{aerialworld2025,
  title={AerialWorld: Occupancy World Models for Urban Aerial Navigation},
  author={Anonymous},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/zitterbewegung/VeryLargeWeebModel}
}
```

## License

Paper content: CC BY 4.0
Code: See repository LICENSE
