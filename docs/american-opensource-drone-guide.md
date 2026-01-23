# Building an Open-Source Drone with American Components

A guide to building a drone using open-source software and US-manufactured components.

## Overview

This guide covers building a quadcopter using:
- **Open-source flight stack**: ArduPilot or PX4
- **American-made hardware**: Where available
- **Open hardware designs**: Community-developed frames and boards

---

## 1. Flight Controller Software (Open Source)

### ArduPilot
- **Origin**: US-based open-source project
- **License**: GPLv3
- **Website**: https://ardupilot.org
- **Features**: Mature, well-documented, supports many vehicles

### PX4
- **Origin**: Dronecode Foundation (Linux Foundation project)
- **License**: BSD
- **Website**: https://px4.io
- **Features**: Modern architecture, good for development

---

## 2. Flight Controller Hardware (American-Made)

### Option A: mRobotics (Mayan Robotics) - California, USA
- **mRo Pixracer R15** (~$99)
- **mRo Control Zero H7** (~$89)
- **mRo Pixhawk 1** (~$199)
- Website: https://store.mrobotics.io
- Fully designed and manufactured in USA

### Option B: Holybro (Some US-assembled options)
- Check for "Made in USA" variants
- Pixhawk 6C/6X series

### Option C: Build Your Own (Open Hardware)
- **PX4 Autopilot Reference Designs**: Open hardware schematics available
- Use US PCB manufacturers:
  - OSH Park (Oregon)
  - Advanced Circuits (Colorado)
  - Sunstone Circuits (Oregon)

---

## 3. Frame (American Options)

### Commercial Frames
| Manufacturer | Location | Products |
|--------------|----------|----------|
| Armattan Productions | Pennsylvania, USA | Lifetime warranty frames |
| ReadyMadeRC | Ohio, USA | Various frames and kits |
| Falcon Multirotors | USA | Heavy-lift frames |
| Detroit Multirotor | Michigan, USA | Custom carbon fiber |

### Open-Source Frame Designs (Build Yourself)
- **OpenPilot CC3D frames** (open hardware)
- **F450/F550 clones** - CNC cut from US carbon fiber suppliers
- **3D Printed**: Use designs from Thingiverse/Printables with US filament

### US Carbon Fiber Suppliers
- DragonPlate (Pennsylvania)
- Rock West Composites (California)
- Clearwater Composites (Minnesota)

---

## 4. Motors (American-Made)

### KDE Direct - Oregon, USA
- Premium brushless motors specifically for drones
- Models: KDE2315XF, KDE3510XF, KDE4014XF series
- Website: https://www.kdedirect.com
- Made in USA, high quality, expensive but reliable

### T-Motor (Limited US Assembly)
- Some products assembled in USA
- Check individual product listings

### Alternative: Rewound/Custom
- Midwest Precision LLC - custom motor work
- Local machine shops can wind stators

---

## 5. Electronic Speed Controllers (ESCs)

### American Options
| Company | Location | Notes |
|---------|----------|-------|
| KDE Direct | Oregon | Premium, Made in USA |
| Castle Creations | Kansas | RC heritage, some drone ESCs |

### Open-Source ESC Firmware
- **BLHeli_32**: Open-source (use on any ESC hardware)
- **AM32**: Fully open-source ESC firmware
- **VESC**: Open hardware + software (originally for EV, adapted for drones)

### Build Your Own ESC
- VESC open hardware designs available
- Order PCBs from US manufacturers
- Source components from Digi-Key, Mouser (US distributors)

---

## 6. Propellers (American-Made)

| Manufacturer | Location | Products |
|--------------|----------|----------|
| APC Propellers | California | Wide selection, Made in USA |
| KDE Direct | Oregon | Matched to their motors |
| Falcon Propellers | USA | Carbon fiber props |

---

## 7. Radio Control System

### Open-Source Options
- **ExpressLRS (ELRS)**: Open-source RC protocol
  - Build transmitter/receiver from open hardware designs
  - US PCB fabrication + US component sourcing

### American RC Companies
| Company | Location | Notes |
|---------|----------|-------|
| Spektrum (Horizon Hobby) | Illinois | DSMX protocol |
| FrSky US | (Check origin) | Some US distribution |

### DIY Approach
- ESP32-based transmitter (open source)
- LoRa modules from US suppliers
- 3D print enclosure

---

## 8. GPS Module

### US Options
- **mRobotics GPS modules** - California
- **Here GPS** (mRo variant)
- Build custom using:
  - u-blox modules (Swiss, but available from US distributors)
  - US PCB fab + assembly

### Open Source GPS
- Use open hardware designs with US-sourced components

---

## 9. Battery

### American Cell Manufacturers
- **Electrochem Solutions** - Massachusetts (specialty cells)

### American Pack Assemblers
- **Titan Power** - Pennsylvania (custom packs)
- **Battery Junction** - Massachusetts
- **MaxAmps** - Washington (high-quality RC packs)

### Note on LiPo Cells
Most lithium cells are manufactured in Asia. US options:
- Use US-assembled packs with imported cells
- Tesla/Panasonic cells (US Gigafactory)
- Research US cell startups (emerging market)

---

## 10. Complete Bill of Materials (Example 5" Quad)

| Component | American Option | Est. Cost |
|-----------|-----------------|-----------|
| Flight Controller | mRo Control Zero H7 | $89 |
| Frame | Armattan Rooster | $85 |
| Motors (4x) | KDE2315XF-885 | $280 |
| ESCs (4x) | Castle Creations | $120 |
| Propellers | APC 5x4.5 (set) | $12 |
| GPS | mRo GPS u-blox | $50 |
| Radio RX | ExpressLRS DIY | $30 |
| Battery | MaxAmps 1500mAh 4S | $60 |
| Power Distribution | mRo PDB | $25 |
| Wiring/Connectors | US suppliers | $30 |
| **Total** | | **~$780** |

---

## 11. Software Setup

### Ground Control Station
- **QGroundControl**: Open source, cross-platform
- **Mission Planner**: Open source, Windows (ArduPilot)
- **APM Planner 2**: Open source, cross-platform

### Firmware Installation
```bash
# ArduPilot - Install prerequisites
sudo apt-get install python3-pip
pip3 install pymavlink mavproxy

# Clone ArduPilot
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
git submodule update --init --recursive

# Build for your flight controller
./waf configure --board mRoControlZeroH7
./waf copter

# Upload firmware
./waf --upload
```

### Configuration Basics
1. Connect via USB to ground station
2. Calibrate accelerometer and compass
3. Set frame type (Quad X, etc.)
4. Calibrate ESCs
5. Calibrate radio
6. Set failsafe parameters
7. Test in stabilize mode first

---

## 12. Compliance & Legal (USA)

### FAA Requirements
- **Register** your drone if over 250g: https://faadronezone.faa.gov
- **Remote ID**: Required for most operations (as of 2023)
- **Part 107**: Required for commercial operations
- **TRUST**: Free recreational safety test

### Remote ID Compliance
- ArduPilot supports Remote ID broadcast modules
- Open-source Remote ID modules in development
- mRo offers Remote ID modules

### ITAR/EAR Considerations
- Be aware of export restrictions if sharing designs internationally
- Most hobby components are not controlled

---

## 13. Community Resources

### Forums & Documentation
- ArduPilot Discourse: https://discuss.ardupilot.org
- PX4 Discuss: https://discuss.px4.io
- RCGroups: https://www.rcgroups.com

### Open Hardware Repositories
- GitHub: Search "open source drone"
- OpenPilot LibrePilot archives
- Dronecode hardware reference designs

### US Makerspaces with Drone Programs
- Check local hackerspaces for drone building communities
- Many have CNC, 3D printers for custom parts

---

## 14. Next Steps

1. **Start small**: Build a micro quad first to learn
2. **Simulate first**: Use ArduPilot SITL simulation
3. **Join communities**: Get help from experienced builders
4. **Document**: Share your builds back to the community
5. **Iterate**: American manufacturing is growing - more options coming

---

## Appendix: US Electronics Distributors

| Distributor | Location | Notes |
|-------------|----------|-------|
| Digi-Key | Minnesota | Huge selection, fast shipping |
| Mouser | Texas | Wide range, good stock |
| Newark | Texas | Industrial focus |
| SparkFun | Colorado | Hobbyist-friendly, some US-made |
| Adafruit | New York | Great tutorials, some US-made |
| Pololu | Nevada | Robotics focus |

---

*Last updated: January 2026*
*This is a community guide - verify current availability and specifications before purchasing.*
