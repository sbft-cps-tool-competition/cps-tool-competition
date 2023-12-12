# OptAngle tool for AV test path generation #

This directory contains the implementation of our **OptAngle** tool for AV test path generation within the context of the [SBFT 2024 CPS Testing Tool Competition](https://github.com/sbft-cps-tool-competition/cps-tool-competition).

**OptAngle** leverages a multi-objctive optimization algorithm over an angle-based representation of a road sequence. It's objetive functions take into consideration (1) the validity of a generated test, (2) the outcome of its execution (pass/fail) and (3) the diversity of the test features.

## Installation

Before running the **OptAngle** tool, please install the required libraries listed in the `optangle/additional-requirements.txt` file (in addition to the generic installation steps required to run the tool competition code).

```
pip install -r optangle/additional-requirements.txt
```

## Running the tool

Once the installation has been completed, you may run the **OptAngle** tool as follows:

```
python competition.py \
        --time-budget 60 \
        --executor beamng \
        --beamng-home <BEAMNG_HOME> --beamng-user <BEAMNG_USER> \
        --map-size 200 \
        --module-path optangle \
        --module-name  src.optangle \
        --class-name OptAngleGenerator
``````

## Author

**[Aren A. Babikian](https://www.arenbabikian.com/)**\
McGill University, Montreal, Canada\
aren.babikian@mail.mcgill.ca

*Please feel free to contact me if you encounter any kind of issues during installation or at runtime.*