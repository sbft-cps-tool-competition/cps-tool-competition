# Spirale at Cyber-Physical Systems Testing Competition

**_Spirale_** is a software project made for participating in the Cyber-physical systems (CPS) testing competition at the 16th Intl. Workshop on Search-Based and Fuzz Testing [SBST](https://sbst21.github.io/tools/).

Spirale is a test generator that produces virtual roads to test a lane keeping assist system. The aim of the generation is to produce diverse failure-inducing tests, i.e., roads that make the lane keeping assist system drive out of the lane. based on the generation of random roads as input in a self-driving cars simulation environment.

### What is the structure of Spirale?

Spirale uses a genetic approach to generate roads modeled as a combination of arcs of spiral. 
- Initially, it generates a random starting population of roads.
- Iteratively, the inizial population is crossed with itself to generate a new set of heir roads
- and a fitness function is used to select an optimized subset of generated roads, on the basis of the test execution results. 

### What are the goals ?
- Minimize the invalid tests. 
- Maximize the tests that outcome as "Fail". 

### How it works?
The run of Spirale depends on the pipeline of the [CPS too competition](https://github.com/sbft-cps-tool-competition/cps-tool-competition).

Spirale can be launched by the following command line: 

> python competition.py --time-budget 180 --executor mock --map-size 200 --module-path ../Spirale --module-name main --class-name Roadtition

The _competition.py_ module expects to find the start() method that must be implemented in the Roadtition class (placed in the _main_ module).

1) The first method called by start() is initial_population_generator() , in which roads having curvatures derived from spiral arcs are created.
 In the below image there is an example of  valid road belonging to the initial population:

![A single road](https://user-images.githubusercontent.com/108838837/211591654-c62199c8-abfb-4670-a79e-a2e403217710.png)

2) The second method called by start() is hebi_generator(initial_population_of_roads) in which the crossover between two different roads of the initial population takes place.Then a fitness value is associated to every single road and to the enterely heir population as quantifiers. Hebi means snake in japanee like the new road in the image:

![A single crossover](https://user-images.githubusercontent.com/108838837/211593200-c45bdaf3-5112-4f08-98e7-a58d4e1c5206.png)

3) The thrid method called is fitness_generator(heir_population) in which a subset of the heir road are selected by the fitness valued of the roads. In this derived new population there are only some roads that have the minus fitness value associated. Iteratively is done until a condition on the fitness new heir population is True.

4) If the time budget is not over and the fitness_generator(heir_population) cannot improve that specific heir population, a new initial population in made re-calling the initial_start_population_generator() repeat the loop from the first point.
   
5) The output of the execution will be a table reporting summary information about the test executions.

![A result](https://user-images.githubusercontent.com/108838837/211600193-dad3c582-94fa-478a-a4f1-b460c5ddb0ca.png)
