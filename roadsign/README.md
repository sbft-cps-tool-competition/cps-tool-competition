# RoadSign

Signal-based road generation tool, developed for the SBFT 2023 Cyber-Physical Systems Testing Tool Competition.

RoadSign generates roads in two steps: (1) Random diverse roads are seeded as an initial population, and (2) a genetic algorithm evolves the population in order to maximize the following features: Instability, discontinuity, and growth. For Step 2, a signal is generated for each road, based on periodic samples of its facing angle, and the features are computed over these signals.

## Usage

This tool is integrated with the competition's [code pipeline](https://github.com/sbft-cps-tool-competition/cps-tool-competition). It can be executed as follows:


`python competition.py --visualize-tests --executor beamng --beamng-home $Env:BEAMNG_HOME --beamng-user $Env:BEAMNG_USER --time-budget 3600 --map-size 200 --module-name roadsign.roadsign_generator --class-name RoadSignGenerator`

# License

This software is distributed under GNU GPL license. See the LICENSE.md file.

## Authors

Jon Ayerdi - Mondragon Unibertsitatea, Arrasate, Spain - jayerdi@mondragon.edu

Aitor Arrieta - Mondragon Unibertsitatea, Arrasate, Spain

Miren Illarramendi - Mondragon Unibertsitatea, Arrasate, Spain
