# Instructions for generating tests with __EvoMBTGenerator__

The __EvoMBTGenerator__ uses the [__EvoMBT__](https://github.com/iv4xr-project/iv4xr-mbt) tool, which is available separately as an executable jar, to generate the tests. For this reason, the correct jar file needs to be present in the `evombt_generator` folder.

The __EvoMBT__ jar file can be downloaded from the corresponding [GitHub release specifically prepared for the SBFT Tool Competition](https://github.com/iv4xr-project/iv4xr-mbt/releases/download/1.2.2/EvoMBT-1.2.2-jar-with-dependencies.jar)

Alternatively, the jar can be built from source by cloning the GitHub repository or downloading a zip of the sources from the repository:
https://github.com/iv4xr-project/iv4xr-mbt

To obtain the jar from source, it suffices to run the following Maven command from the project root folder:

> mvn clean package -DskipTests

after the build finishes, the jar file could be found in the folder `target`

Java 11 is required to run the __EvoMBTGenerator__ generator
