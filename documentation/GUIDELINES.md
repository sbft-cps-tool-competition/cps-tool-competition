# Competition Guidelines #

## Goal ##

The tool should generate test inputs to test a Lane Keeping Assist System (LKAS).
The competitors should generate roads that force the ego-car, i.e., the test subject,
to drive off its lane without creating invalid roads.

## Tests as Driving Tasks ##

In the competition, tests are driving tasks that the ego-car, i.e., a car equipped with the
Lane Keeping Assist System under test, must complete. These driving tasks are defined in terms
of a road that the ego-car must follow. 

### What is a test input for the current competition? ###

For simplicity, we consider driving scenarios on single, flat roads surrounded by green grass.
The ego-car must drive along the roads keeping the right lane. The environmental conditions
and the road layout are fixed and predefined. In particular, the roads consist of two fixed-width
lanes, which are divided by a solid yellow line. Two additional white lines define the exterior
boundaries of the lanes.

### What makes a test input? ###

The competitor tools should generate roads as sequences of points, i.e., _road points_, defined
in a two-dimensional squared map with predefined size (e.g., 200-by-200 meter). 
The sequence of _road points_ defines the _road spine_, i.e., the road's center line.

The **first point** in the sequence of _road points_ defines the starting location of the ego-car
by convention, while the **last point** defines the target location. The road points are automatically
interpolated using cubic splines to obtain the final road geometry.

> **NOTE**: the road's geometry automatically defines the initial placement and rotation of
>the ego-vehicle.

The following image illustrates how a road is defined from the following _road points_ over a map
of size 200-by-200 meters: 


```
[
    (10.0, 20.0), (30.0, 20.0), (40.0, 30.0), 
    (50.0, 40.0), (150.0, 100.0), (30.0, 180.0)
]
```

![Sample Road caption="test"](./figures/sample_road.PNG "Sample Road")

In the figure, the inner square identifies the map's boundary (200x200), the white dots correspond
to the _road points_, the solid yellow line corresponds to the _road spine_ that interpolates them,
and, finally, the gray area is the road.

As the figure illustrates, the road layout consists of one left lane and one right lane
(where the car drives). Each lane is four meters wide, and the lane markings are defined similarly
to the US standards: solid yellow line in the middle and solid white lines on the
side (not drawn in the figure).


### Valid Roads ###

We perform the following validity checks on the roads before using them in the driving tasks:

* Roads must be made of at least 2 _road points_.
* Roads must never intersect or overlap.
* Turns must have a geometry that allows the ego-car to completely fit in the lane while driving on them; so, "too" sharp edges, i.e., turns with small radius, are disallowed.
* Roads must completely fit the given squared map boundaries; implicitly, this limits the roads' maximum length. 
* To avoid overly complex roads and limit the issues with spline interpolation, we also limit the number of _road points_ that can be used to define roads (500/1000 points).

Invalid roads are reported as such (hence not executed), so they do not count as a *failed* test. 

## Competition ##
The contest's organizers provide this
[code pipeline](https://github.com/se2p/tool-competition-av/tree/main/code_pipeline) to check the tests' validity,
execute them, and keep track of the time budget. The submission should integrate with it **without modifiying it**.

At the moment, execution can be mocked or simulated. Mocked execution generates random data and is
meant **only** to support development. Simulation instead requires executing BeamNG.tech simulation
(see the [Installation Guide](INSTALL.md) for details about registering and installing the simulation software).

There's no limit on the number of tests that can be generated and executed. However, there's a limit on
the execution time: the generation can continue until the given time budget is reached. The time budget is measured in
real time (wall time).

To participate, competitors must submit the code of their test generator and instructions about installing
it before the official deadline.

## How to Submit ##

Submitting a tool to this competition requires participants to share their code with us.
So, the easiest way is to fork the master branch of this repo and send us a pull request with your code
in it. Alternatively, you can send the URL of a repo where we can download the code or even a "tar-ball"
with your code in it.

We will not publish or release the competitors' code unless stated otherwise, but we encourage competitors
to let us merge their code into this repository after the competition is over. 

We will come back to you if we need support to install and run your code.

## Results ##

The test generators' evaluation will be conducted using the same simulation and code-pipeline used for
the development. Still, we will not release the test subjects used for the evaluation before the
submission deadline to avoid biasing the solutions towards it.

As explained [here](../README.md), the evaluation is based on a feature coverage with features possibly including:
* Direction Coverage (DirCov). This measures how many directions a road segment covers. We measure the angles between adjacent road points and place these angles into bins each spanning 36 degrees.
* Standard Deviation of the Steering Angle (StdSA). We compute the standard deviation of the car steering angle during a road segment.
* Maximum Curvature (MaxCurv). We quantify the smoothness of a road as the inverse of its turns’ radius.
* Mean Lateral Position (MLP). We measure how close the driving agent drives to the lane margins.
 
The features are computed only for failed tests with respect to the road segment relevant to the failure. This segment is the 60 unit long segment centered around the out-of-bounds location (30 units before, 30 units after).

> **Note**: tests fail for different reasons. For example, a test fail if the ego-car does
>not move, or does not reach the end of the road within a timeout (computed over the length
>of the road), or drives off the lane.

## Sample Test Generators ##
The submission package comes with an implementation of [sample test generators](../sample_test_generators/README.md). This serves the dual purpose of providing an example on how to use our code pipeline and as a baseline for the evaluation.

## Installation ##
Check the [Installation Guide](INSTALL.md)

## Technical considerations ##
The competition code can be run by executing `competition.py` from the main folder of this repo.

Usage (from command line): 

```
Usage: competition.py [OPTIONS]

Options:
  --executor [mock|beamng|dave2]  The name of the executor to use. Currently
                                  we have 'mock', 'beamng' or 'dave2'.
                                  [default: (Mock Executor (meant for
                                  debugging))]
                                  
  --dave2-model PATH              Path of the pre-trained Dave2 driving AI
                                  model (in .h5 format). Mandatory if the
                                  executor is dave2
                                  
  --beamng-home PATH              Customize BeamNG executor by specifying the
                                  home of the simulator.  [default: (None)]
                                  
  --beamng-user PATH              Customize BeamNG executor by specifying the
                                  location of the folder where levels, props,
                                  and other BeamNG-related data will be
                                  copied.** Use this to avoid spaces in
                                  URL/PATHS! **  [default: (Currently Active
                                  User (~/BeamNG.tech/))]
                                  
  --time-budget TEXT              Overall budget for the generation and
                                  execution. Expressed in 'real-time'seconds.
                                  
  --map-size INTEGER              The lenght of the size of the squared map
                                  where the road must fit.Expressed in meters.
                                  [default: (200m, which leads to a 200x200m^2
                                  squared map)]

  --oob-tolerance FLOAT           The tolerance value that defines how much of
                                  the vehicle should be outside the lane to
                                  trigger a failed test. Must be a value
                                  between 0.0 (all oob) and 1.0 (no oob)
                                  [default: (0.95)]
                                  
  --speed-limit INTEGER           The max speed of the ego-vehicleExpressed in
                                  Kilometers per hours  [default: (70 Km/h)]
                                  
  --module-name TEXT              Name of the module where your test generator
                                  is located.  [required]
                                  
  --module-path PATH              Path of the module where your test generator
                                  is located.
                                  
  --class-name TEXT               Name of the class implementing your test
                                  generator.  [required]
                                  
  --visualize-tests               Visualize the last generated test, i.e., the
                                  test sent for the execution. Invalid tests
                                  are also visualized.  [default: (Disabled)]
                                  
  --log-to PATH                   Location of the log file. If not specified
                                  logs appear on the console
                                  
  --debug                         Activate debugging (results in more logging)
                                  [default: (Disabled)]
                                  
  --help                          Show this message and exit.
```

## Examples

The following sections exemplifies how to use the code pipeline with sample generators and custom generators.

### Using the sample test generators

As an example, you can use the `mock` executor to "pretend to" execute the `one-test-generator.py`, or you can use `beamng` executor as shown below. 

In any case, we **strongly** suggest you to activate a virtual environment (see [installation instructions](./INSTALL.md)). 

`cd` to the root of this repository and run the following command after replacing `<BEAMNG_HOME>` and `<BEAMNG_USER>` accordingly:

``` 
py.exe competition.py \
        --time-budget 60 \
        --executor beamng \
        --beamng-home <BEAMNG_HOME> --beamng-user <BEAMNG_USER> \
        --map-size 200 \
        --module-name sample_test_generators.one_test_generator \
        --class-name OneTestGenerator
```

Similarly, you can run the `RandomTestGenerator` from the `random_generator` module or the `JanusGenerator` from the `deepjanus_seed_generator` module by updating the above command with the following options:

```
        --module-name sample_test_generators.random_generator \
        --class-name RandomTestGenerator
```
or 

```
        --module-name sample_test_generators.deepjanus_seed_generator \
        --class-name JanusGenerator
```

### Using the public test generators submitted to previous SBST competitions

To exemplify how one can use a custom generator, we imported existing test generators that have been submitted to past editions of the competition as git submodules (read more about this [here](https://git-scm.com/book/it/v2/Git-Tools-Submodules)).

> Note: we included only **publicly available** generators but forked the original repositories to port the test generators to the updated code_pipeline API

To use the custom generators, first initialize and update their submodules:

```
git submodule init
git submodule update
```

Then you can invoke them by specifying their location (`--module-path`), module's fully qualified name (`--module-name`), and test generator main class's name (`--class-name`).

For instance, you can run [Frenetic](https://ieeexplore.ieee.org/document/9476234) by
Ezequiel Castellano, Ahmet Cetinkaya, Cédric Ho Thanh, Stefan Klikovits, Xiaoyi Zhang, and Paolo Arcaini as follows:

``` 
py.exe competition.py \
        --time-budget 60 \
        --executor beamng \
        --beamng-home <BEAMNG_HOME> --beamng-user <BEAMNG_USER> \
        --map-size 200 \
        --module-path frenetic-sbst2021 \
        --module-name src.generators.random_frenet_generator \
        --class-name Frenetic
```

Likewise, you can run [SWAT](https://ieeexplore.ieee.org/document/9476167) by Dmytro Humeniuk, Giuliano Antoniol, and Foutse Khomh as follows:

``` 
py.exe competition.py \
        --time-budget 60 \
        --executor beamng \
        --beamng-home <BEAMNG_HOME> --beamng-user <BEAMNG_USER> \
        --map-size 200 \
        --module-path swat-sbst2021 \
        --module-name swat_gen.swat_generator \
        --class-name SwatTestGenerator
```

In case of import errors, check that you have installed the packages listed in the respective `requirements.txt` file for Frenetic and SWAT.
