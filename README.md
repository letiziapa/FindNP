# MPhil DIS Final Project

## Description
This repository contains the code implemented in the project **Searches for New Physics at the Large Hadron Collider**. 
The code base is contained in the `src` directory. The Neural Networks are trained on Monte Carlo simulations of the Standard Model and Physics Beyond the Standard model (not uploaded because of size limits).
The functions used throughout the analysis are saved under `utils`.
The weights of each trained Neural Network are saved under `weights` ('Parton_level_< >' for training at the parton level and 'Full_model_weights_< >' for the complete analysis)

The `traditional_analysis.py` file provides the results and plots used in the binned analysis.
The `PartonLevel_< >.py` files refer to the second section of the project (analysis of the neural networks at the parton level). The outputs of each model can be compared with the analytical descriptions of the decision boundary. These are unique to each coefficient (in their linear and quadratic form).
The `FullAnalysis.py` concludes the study by combining the Neural Networks trained on the full-feature datasets and the Nested Sampling algorithm used at the beginning. The NS parameter inference might require some more time to execute (on average less than 5 minutes) due to the complexity of the likelihood.

The report and executive summary can be found under `report`.

## Usage
The repository can be cloned into a local directory by typing the following on the command line:
```bash
git clone https://github.com/letiziapa/FindNP.git
```

The required packages are specified in the `environment.yml` file, and a `conda` environment can be created by running on the command line:
```bash
conda env create --name findNP --file environment.yml
```
The python files can be ran from the command line while in the `src` directory:
```bash
python src/<filename>.py
```
While the Jupyter notebooks can be ran from an IDE.

The documentation for this project can be built using Doxygen. First, navigate to the `docs` directory:
```bash
cd docs
```
Then, run Doxygen:
```bash
doxygen Doxyfile
```
This will generate the documentation in HTML format in the docs/html directory. To view the documentation, open the index.html file from the command line:
```bash
open html/index.html
```

The code was developed on Python 3.10.13 in Visual Studio Code using macOS on an Apple M2 processor.


