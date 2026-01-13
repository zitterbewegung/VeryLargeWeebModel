# Visual Assets for Documentation

This directory contains visual assets for the blog post, slides, and video.

## Required Images

### From OccWorld Paper/GitHub
Download from: https://github.com/wzzheng/OccWorld

| Filename | Source | Description |
|----------|--------|-------------|
| `occworld_architecture.png` | Paper Fig. 2 | Full architecture diagram |
| `occworld_teaser.png` | README | Teaser image showing predictions |
| `occworld_results.png` | Paper Fig. 4 | Qualitative results grid |

### From Tokyo PLATEAU
Download from: https://www.mlit.go.jp/plateau/ or https://www.geospatial.jp/

| Filename | Source | Description |
|----------|--------|-------------|
| `plateau_tokyo_overview.png` | PLATEAU website | Aerial view of Tokyo 3D model |
| `plateau_buildings.png` | PLATEAU VIEW | Close-up of building detail |
| `plateau_shinjuku.png` | Dataset | Shinjuku ward rendering |

### Custom Diagrams (Create These)

| Filename | Description | Tool Suggestion |
|----------|-------------|-----------------|
| `header_tokyo_occworld.png` | Blog header: Tokyo + AI overlay | Figma/Canva |
| `pipeline_overview.png` | PLATEAU → Gazebo → Training flow | draw.io |
| `training_curves.png` | Loss curves from actual training | matplotlib |
| `prediction_comparison.png` | GT vs Prediction side-by-side | matplotlib |
| `cost_breakdown.png` | Lambda Cloud cost infographic | Canva |

### Screenshots (Capture These)

| Filename | Description |
|----------|-------------|
| `lambda_dashboard.png` | Lambda Cloud instance page |
| `tensorboard_graphs.png` | TensorBoard training metrics |
| `nvtop_gpu.png` | GPU utilization during training |
| `gazebo_simulation.png` | Drone in Tokyo Gazebo world |
| `terminal_training.png` | Training logs in terminal |

---

## Image Specifications

### Blog Post
- Header: 1200 x 630 px (social sharing)
- Inline: 800 px width max
- Format: PNG or WebP

### Slides (Marp)
- Aspect: 16:9 (1920 x 1080)
- Background images: Full resolution
- Diagrams: SVG preferred

### Video Thumbnail
- Size: 1280 x 720 px
- Text: Large, readable at small sizes
- Colors: Dark bg (#1a1a2e), cyan (#00d9ff), purple (#9d4edd)

---

## Diagram: Pipeline Overview

```
Use draw.io or Excalidraw to create:

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │   PLATEAU    │     │    GAZEBO    │     │   OCCWORLD   │   │
│   │              │────▶│              │────▶│              │   │
│   │  [Tokyo 3D]  │     │ [Simulation] │     │  [Training]  │   │
│   └──────────────┘     └──────────────┘     └──────────────┘   │
│                                                                  │
│   Download OBJ    →    Spawn Drones    →    Fine-tune Model    │
│   Convert to SDF  →    Record Sensors  →    50 Epochs          │
│   ~2.1 GB         →    Generate Data   →    ~24 Hours          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diagram: OccWorld Architecture

```
Simplified version for slides:

    ┌─────────────────────────────────────────────────────────┐
    │                                                          │
    │   HISTORY                              FUTURE            │
    │   ┌───┬───┬───┬───┐                   ┌───┬───┬───┐     │
    │   │ t │t-1│t-2│t-3│                   │t+1│t+2│...│     │
    │   └─┬─┴─┬─┴─┬─┴─┬─┘                   └─▲─┴─▲─┴─▲─┘     │
    │     │   │   │   │                       │   │   │       │
    │     ▼   ▼   ▼   ▼                       │   │   │       │
    │   ┌─────────────────┐   ┌─────────┐   ┌─────────────┐   │
    │   │     VQVAE       │──▶│  GPT-2  │──▶│    VQVAE    │   │
    │   │    Encoder      │   │Transform│   │   Decoder   │   │
    │   └─────────────────┘   └─────────┘   └─────────────┘   │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
```

---

## Color Palette

| Name | Hex | Usage |
|------|-----|-------|
| Background Dark | #1a1a2e | Slide backgrounds |
| Background Light | #16213e | Card backgrounds |
| Primary Cyan | #00d9ff | Headings, accents |
| Secondary Purple | #9d4edd | Subheadings |
| Text Light | #eaeaea | Body text |
| Text Muted | #a0a0a0 | Captions |
| Success Green | #00ff88 | Positive metrics |
| Warning Orange | #ff9f1c | Alerts |

---

## Tools for Creating Assets

### Diagrams
- [draw.io](https://draw.io) - Free, exports PNG/SVG
- [Excalidraw](https://excalidraw.com) - Hand-drawn style
- [Mermaid](https://mermaid.js.org) - Code-based diagrams

### Charts
- matplotlib + seaborn (Python)
- [Plotly](https://plotly.com) - Interactive
- Google Sheets/Excel

### Graphics
- [Canva](https://canva.com) - Templates
- [Figma](https://figma.com) - Professional
- GIMP/Photoshop - Photo editing

### Screenshots
- macOS: Cmd+Shift+4
- Linux: gnome-screenshot / flameshot
- Annotate with: Skitch, Shottr

---

## Licensing Notes

- **OccWorld figures**: Check paper license, cite properly
- **PLATEAU images**: CC BY 4.0, attribute MLIT
- **Your screenshots**: Your own work
- **Lambda logo**: Check brand guidelines
