
<h1 align="center">
	AmbieGenVAE tool for autonomous robotic systems testing
</h1>

This repository contains the implementation of our tool AmbieGenVAE, adapted for the [SBFT 2024 CPS testing competition](https://github.com/sbft-cps-tool-competition/cps-tool-competition). This tool is based on our previous search-based tool [AmbieGen](https://github.com/swat-lab-optimization/AmbieGen-tool):
```
@article{HUMENIUK2023102990,
title = {AmbieGen: A search-based framework for autonomous systems testing},
journal = {Science of Computer Programming},
volume = {230},
pages = {102990},
year = {2023},
issn = {0167-6423},
doi = {https://doi.org/10.1016/j.scico.2023.102990},
url = {https://www.sciencedirect.com/science/article/pii/S0167642323000722},
author = {Dmytro Humeniuk and Foutse Khomh and Giuliano Antoniol}
}
```

In AmbieGenVAE we focus on improving the computational efficiency of the search algorithm by (1) using an efficient representation i.e. 1D array of real numbers and (2) sampling the search-space in the promising areas.

We achieve this by firstly collecting a dataset of challenging and diverse road topologies and training a Variational Autoencoder (VAE) with it. To produce the dataset, we run the genetic algorithm maximizing the curvature of the road topologies, while keeping it in the allowed bounds. In this implementation, we represent the road topology as a sequence of kappa values, as descibed by [Castellano et al](https://ieeexplore.ieee.org/document/9724804). The VAE is trained to encode an array of kappa values into a 1D array of real numbers from normal distribution with mean 0 and standard deviation of 1. The decoder is trained to reconstruct the original road topology from the encoded representation.

In the second step, we run a single-objective genetic algorithm with the encoded representations of the road topologies. Performing the optimization in the latent space, we only sample new test cases from the promising areas of the search space (as defined by our dataset of 10k samples). Moreover, having a compact representation (1D array of real numbers) alows us to use some advanced search operators (e.g. [simulated binary crossover](https://dl.acm.org/doi/10.1145/1276958.1277190)). As the objective, we maximize the percentrage of the vehicle going out of the bounds in the BeamNg simulator. To assure the diversity of the generated road topologies, we calculate the cosine similarity between the indivuals after each generation and remove the individuals with the similarity higher than a certain threshold. 


## Usage
Our tool was implemented with python=3.9.18, however, it should work with any python>=3.7. 

1. Before using the tool, install the packages from the provided requirements files:
```python 
pip install -r requirements.txt
```
2. Install the additional packages needed for our implementation:
```python
pip install -r additional-requirements.txt
```
3. Install the cuda-enabled version of [PyTorch](https://pytorch.org/get-started/locally/). To know the version of cuda on your machine, run:
```
nmvidia-smi
```
and check the version of cuda in the first line of the output. 
We used PyTorch 2.1.0 with cuda 11.7. We utilized the following installation command:
```python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
If you have a different version of cuda, change the command according to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).
If you don't have a GPU, install the cpu version of PyTorch:
```python
pip install torch torchvision torchaudio
```
However, we tested our soultion with GPU and this option is recommended.

4. Start test case generation:  
For BeamNg agent: 
```python
python competition.py --time-budget 10800 --executor beamng --map-size 200 --module-name ambiegenvae_generator --class-name AmbiegenVAEGenerator --beamng-home "" --beamng-user "" --oob-tolerance 0.85
```
For dave2 agent:
```python
python competition.py --time-budget 10800 --executor dave2 --map-size 200 --module-name ambiegenvae_generator --class-name AmbiegenVAEGenerator --beamng-home "" --beamng-user "" --oob-tolerance 0.85 --dave2-model "dave2\\beamng-dave2.h5"
```


## Authors
### [Dmytro Humeniuk](https://dgumenyuk.github.io/) and [Foutse Khomh](http://khomh.net/)
Polytechnique Montréal, Canada, 2023  
Contact e-mail: dmytro.humeniuk@polymtl.ca
