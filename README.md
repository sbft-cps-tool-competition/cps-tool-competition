# Cyber-Physical Systems Testing Tool Competition #

The [SBFT Workshop](https://sbft24.github.io/) offers a challenge for software testers who want to work with self-driving cars in the context of the usual [tool competition](https://sbft24.github.io/tools/).

## Important Dates

The deadline to submit your tool is: ~~**November 11th 2023**~~ **December 1st 2023**

The results of the evaluation will be communicated to participants on: ~~**December 21st 2023**~~ **January 11 2024**

The camera-ready paper describing your tool is due to: **January 25th 2024**

## Overview ##

The competitors should propose a test generator that produces virtual roads to test a lane keeping assist system. The aim of the generation is to produce diverse failure-inducing tests, i.e., roads that make the lane keeping assist system drive out of the lane. 

The ranking of the tools is based on coverage which measures the number of failed tests and their diversity. In fact, it is in the interest of any tester to provide as diverse failures as possible. This facilitates better root cause analysis. 

The generated roads are evaluated in the [**BeamNG.tech**](https://www.beamng.tech/) driving simulator.
This simulator is ideal for researchers due to its state-of-the-art soft-body physics simulation, ease of access to sensory data, and a Python API to control the simulation.

In the competition two lane keeping assist systems are used: BeamnNG.AI provided by the BeamnNG.tech simulator and DAVE-2 trained by the competition organizers (with a `speed-limit` of 25 km/h).

[![Video by BeamNg GmbH](https://github.com/BeamNG/BeamNGpy/raw/master/media/steering.gif)](https://github.com/BeamNG/BeamNGpy/raw/master/media/steering.gif)

>Note: BeamNG GmbH, the company developing the simulator, kindly offers it for free for researcher purposes upon registration (see [Installation](documentation/INSTALL.md)).

## Implement Your Test Generator ##

We make available a [code pipeline](code_pipeline) that will integrate your test generator with the simulator by validating, executing and evaluating your test cases. Moreover, we offer some [sample test generators](sample_test_generators/README.md) to show how to use our code pipeline.

## Comparing the Test Generators ##

Deciding which test generator is the best is far from trivial and, currently, remains an open challenge. In the 2024 edition of the competition, we use different diversity metrics to rank the test generators. The first metric we use relies on feature maps and measures much each tool covers the map. Possible features to be used include:

* Direction Coverage (DirCov).
* Standard Deviation of the Steering Angle (StdSA).
* Maximum Curvature (MaxCurv).
* Mean Lateral Position (MLP).
* Standard Deviation of the Speed (StdSpeed).

The second metric relies on clustering techniques to measure the diversity of the trajectories of the ego vehicle. The more clusters a test generator covers, the more diverse its failures.

We expect that the submitted tools are stochastic in nature, so we compute the coverage as the total coverage over several repetitions of the tool.

## Information About the Competition ##

More information can be found on the SBFT tool competition website: [https://sbft24.github.io/tools/](https://sbft24.github.io/tools/). See also the tool report of the previous competition edition: <https://ieeexplore.ieee.org/document/10190377>

## Repository Structure ##

[Code pipeline](code_pipeline): code that integrates your test generator with the simulator.

[Self driving car testing library](self_driving): library that helps the integration of the test input generators, our code pipeline, and the BeamNG simulator.

[Scenario template](levels_template/tig): basic scenario used in this competition.

[Documentation](documentation/README.md): contains the installation guide, detailed rules of the competition, and the frequently asked questions.

* [Installation Guide](documentation/INSTALL.md): information about the prerequisites and how to install the code pipeline.
* [Guidelines](documentation/GUIDELINES.md): goal and rules of the competition.
* [FAQ](documentation/FAQ.md): answers to the most frequent asked questions.

[Sample test generators](sample_test_generators/README.md): sample test generators already integrated with the code pipeline for illustrative purposes.

[Requirements](requirements.txt): contains the list of the required packages.

## License ##

The software we developed is distributed under GNU GPL license. See the [LICENSE.md](LICENSE.md) file.

## Contacts ##

Dr. Matteo Biagiola  - Università della Svizzera italiana, Lugano, Switzerland - matteo.biagiola@usi.ch

Dr. Stefan Klikovits - Johannes Kepler University, Linz, Austria - stefan.klikovits@jku.at
