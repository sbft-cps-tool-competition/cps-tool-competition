# Cyber-Physical Systems Testing Tool Competition #

The [SBFT Workshop](https://sbft25.github.io/) has offered three editions of the self-driving cars testing challenge in the past four years. We provide the link to the respective reports in reversed chronological order:
- 2024: [DOI](https://doi.org/10.1145/3643659.3643932)
- 2023: [DOI](https://doi.org/10.1109/SBFT59156.2023.00010)
- 2022: [DOI](https://doi.org/10.1145/3526072.3527538)
- 2021: [DOI](https://doi.org/10.1109/SBST52555.2021.00011)

Below, is an overview of the 2024 edition of the testing challenge.

## Overview ##

The competitors should propose a test generator that produces virtual roads to test a lane keeping assist system. The aim of the generation is to produce diverse failure-inducing tests, i.e., roads that make the lane keeping assist system drive out of the lane. 

The ranking of the tools is based on coverage which measures the number of failed tests and their diversity. In fact, it is in the interest of any tester to provide as diverse failures as possible. This facilitates better root cause analysis. 

The generated roads are evaluated in the [**BeamNG.tech**](https://www.beamng.tech/) driving simulator.
This simulator is ideal for researchers due to its state-of-the-art soft-body physics simulation, ease of access to sensory data, and a Python API to control the simulation.

In the competition two lane keeping assist systems are used: BeamnNG.AI provided by the BeamnNG.tech simulator and DAVE-2 trained by the competition organizers (with a `speed-limit` of 25 km/h).

[![Video by BeamNg GmbH](https://user-images.githubusercontent.com/93574498/207164554-3f3d9478-3970-4c08-b1e3-2b656313ae33.webm)]([https://github.com/BeamNG/BeamNGpy/raw/master/media/steering.gif](https://user-images.githubusercontent.com/93574498/207164554-3f3d9478-3970-4c08-b1e3-2b656313ae33.webm))

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

## Citing this Project ##

To cite this repository in publications:

```bibtex
// 2024 edition
@inproceedings{DBLP:conf/sbst/BiagiolaK24,
  author       = {Matteo Biagiola and
                  Stefan Klikovits},
  title        = {{SBFT} Tool Competition 2024 - Cyber-Physical Systems Track},
  booktitle    = {Proceedings of the 17th {ACM/IEEE} International Workshop on Search-Based
                  and Fuzz Testing, {SBFT} 2024, Lisbon, Portugal, 14 April 2024},
  pages        = {33--36},
  publisher    = {{ACM}},
  year         = {2024},
  url          = {https://doi.org/10.1145/3643659.3643932},
  doi          = {10.1145/3643659.3643932},
  timestamp    = {Tue, 22 Oct 2024 21:07:12 +0200},
  biburl       = {https://dblp.org/rec/conf/sbst/BiagiolaK24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

// 2023 edition
@inproceedings{DBLP:conf/icse/BiagiolaKPR23,
  author       = {Matteo Biagiola and
                  Stefan Klikovits and
                  Jarkko Peltom{\"{a}}ki and
                  Vincenzo Riccio},
  title        = {{SBFT} Tool Competition 2023 - Cyber-Physical Systems Track},
  booktitle    = {{IEEE/ACM} International Workshop on Search-Based and Fuzz Testing,
                  SBFT@ICSE 2023, Melbourne, Australia, May 14, 2023},
  pages        = {45--48},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.1109/SBFT59156.2023.00010},
  doi          = {10.1109/SBFT59156.2023.00010},
  timestamp    = {Sun, 04 Aug 2024 19:39:38 +0200},
  biburl       = {https://dblp.org/rec/conf/icse/BiagiolaKPR23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

// 2022 edition
@inproceedings{DBLP:conf/sbst/GambiJRZ22,
  author       = {Alessio Gambi and
                  Gunel Jahangirova and
                  Vincenzo Riccio and
                  Fiorella Zampetti},
  title        = {{SBST} Tool Competition 2022},
  booktitle    = {15th {IEEE/ACM} International Workshop on Search-Based Software Testing,
                  SBST@ICSE 2022, Pittsburgh, PA, USA, May 9, 2022},
  pages        = {25--32},
  publisher    = {{IEEE}},
  year         = {2022},
  url          = {https://doi.org/10.1145/3526072.3527538},
  doi          = {10.1145/3526072.3527538},
  timestamp    = {Tue, 21 Mar 2023 21:02:23 +0100},
  biburl       = {https://dblp.org/rec/conf/sbst/GambiJRZ22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

// 2021 edition
@inproceedings{DBLP:conf/sbst/PanichellaGZR21,
  author       = {Sebastiano Panichella and
                  Alessio Gambi and
                  Fiorella Zampetti and
                  Vincenzo Riccio},
  title        = {{SBST} Tool Competition 2021},
  booktitle    = {14th {IEEE/ACM} International Workshop on Search-Based Software Testing,
                  {SBST} 2021, Madrid, Spain, May 31, 2021},
  pages        = {20--27},
  publisher    = {{IEEE}},
  year         = {2021},
  url          = {https://doi.org/10.1109/SBST52555.2021.00011},
  doi          = {10.1109/SBST52555.2021.00011},
  timestamp    = {Tue, 21 Mar 2023 21:02:23 +0100},
  biburl       = {https://dblp.org/rec/conf/sbst/PanichellaGZR21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
