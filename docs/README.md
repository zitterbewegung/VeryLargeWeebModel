# Documentation

This directory contains all documentation, guides, and media assets for the OccWorld Tokyo training project.

## Contents

### Technical Guides

| File | Description |
|------|-------------|
| [training_guide.md](training_guide.md) | Complete guide to training OccWorld on Tokyo data |
| [lambda_cloud_deployment.md](lambda_cloud_deployment.md) | Deploying to Lambda Cloud GPU instances |
| [architecture_blueprint.md](architecture_blueprint.md) | System architecture and design |
| [optimization_and_debugging.md](optimization_and_debugging.md) | Performance tuning and troubleshooting |

### Media & Outreach

| File | Description |
|------|-------------|
| [blog_post.md](blog_post.md) | Technical blog post for publication |
| [youtube_video_script.md](youtube_video_script.md) | Full script for tutorial video |
| [slides.md](slides.md) | Marp-compatible slide deck |
| [assets/](assets/) | Images, diagrams, and visual resources |

---

## Quick Links

### Getting Started
1. Read [training_guide.md](training_guide.md) for the full workflow
2. For cloud training, see [lambda_cloud_deployment.md](lambda_cloud_deployment.md)

### Creating Content
1. Use [blog_post.md](blog_post.md) for written tutorials
2. Use [youtube_video_script.md](youtube_video_script.md) for video production
3. Use [slides.md](slides.md) for presentations

---

## Converting Slides

The slides are written in [Marp](https://marp.app/) markdown format.

### Option 1: Marp CLI
```bash
# Install
npm install -g @marp-team/marp-cli

# Convert to HTML
marp slides.md -o slides.html

# Convert to PDF
marp slides.md -o slides.pdf

# Convert to PowerPoint
marp slides.md -o slides.pptx
```

### Option 2: VS Code Extension
1. Install "Marp for VS Code" extension
2. Open slides.md
3. Click preview icon or export

### Option 3: Marp Web
1. Go to https://web.marp.app/
2. Paste markdown content
3. Export as needed

---

## Blog Post Publishing

The blog post is written in standard Markdown and can be published to:

| Platform | Notes |
|----------|-------|
| Medium | Copy/paste, upload images separately |
| Dev.to | Direct markdown support |
| Hashnode | Direct markdown support |
| Personal blog | Hugo/Jekyll/Gatsby compatible |
| GitHub Pages | Render directly |

### Before Publishing
1. Replace `[YOUR_USERNAME]` with actual GitHub username
2. Replace `[YOUR NAME]` with author name
3. Replace `[DATE]` with publication date
4. Add actual images to replace placeholders
5. Update repository links

---

## Video Production

### Script Sections
1. **Hook & Intro** (0:00-1:30) - Attention grabber
2. **What is OccWorld?** (1:30-3:00) - Background
3. **Project Overview** (3:00-5:00) - Pipeline explanation
4. **Lambda Cloud Setup** (5:00-8:00) - Screen recording
5. **Download Models & Data** (8:00-11:00) - Terminal recording
6. **Start Training** (11:00-14:00) - Live demo
7. **Monitor & Results** (14:00-17:00) - TensorBoard walkthrough
8. **Next Steps & Outro** (17:00+) - CTA and resources

### Required Recordings
- [ ] Screen recording: Lambda Cloud signup/launch
- [ ] Terminal recording: SSH, git clone, setup scripts
- [ ] Terminal recording: Training start and logs
- [ ] Screen recording: TensorBoard graphs
- [ ] B-roll: Gazebo simulation with drone
- [ ] B-roll: Tokyo 3D model flythrough

### Tools
- OBS Studio for screen recording
- DaVinci Resolve / Premiere for editing
- Canva for thumbnail

---

## External References

### Papers
- [OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](https://arxiv.org/abs/2311.16038) (ECCV 2024)
- [nuScenes: A multimodal dataset for autonomous driving](https://arxiv.org/abs/1903.11027) (CVPR 2020)

### Repositories
- [OccWorld GitHub](https://github.com/wzzheng/OccWorld)
- [BEVFusion GitHub](https://github.com/mit-han-lab/bevfusion)
- [Occ3D GitHub](https://github.com/Tsinghua-MARS-Lab/Occ3D)

### Data Sources
- [Tokyo PLATEAU](https://www.geospatial.jp/ckan/dataset/plateau-tokyo23ku)
- [nuScenes](https://www.nuscenes.org/)

### Cloud Providers
- [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud)
- [Lambda Docs](https://docs.lambda.ai/)

---

## Contributing

To add or update documentation:

1. Follow the existing markdown style
2. Use relative links for internal references
3. Add images to `assets/` directory
4. Update this README if adding new files
5. Test any code snippets before including

## License

Documentation is provided under the same license as the main project.
