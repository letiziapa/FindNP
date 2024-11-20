# MPhil DIS Final Project

## Description
This repository contains the code implemented in the project **Searches for New Physics at the Large Hadron Collider**. 
The code base is contained in the `src` directory. Here, there are several subdirectories with the functions and data used throughout the project:
- `data` contains the toy dataset of observed events (`data.json`) and the Monte Carlo generated samples of the Standard Model and Effective Field Theory coefficients, at the parton level (`PartonLevel`) and with the full set of features (`MonteCarlo`).
- Under `notebooks` one can find the Jupyter notebooks to train the Neural Networks and calculate the losses of each model. It is not recommended to execute these notebooks since the computational time taken by each model is of approximately 20 minutes. These were only included in the repository to show the training process.
- The `plots` subfolder can be used to save the plots obtained from the analysis. Most of the plots presented in the report are not present in the folder because they can be easily reproduced from the `.py` files, except for the loss curves from the Neural Network training.
- The functions used throughout the analysis are saved under `utils`. Only the `funcs.py` file was personally developed, the other files were provided at the beginning of the project as they implement the particle physics functions required for the analytical evaluation.
- The weights of each trained Neural Network are saved under `weights` ('Parton_level_< >' for training at the parton level and 'Full_model_weights_< >' for the complete analysis)

The `traditional_analysis.py` file provides the results and plots used in the first part of the project (unbinned analysis).

The `PartonLevel_< >.py` files refer to the second section of the project (analysis of the neural networks at the parton level). The outputs of each model can be compared with the analytical descriptions of the decision boundary. These are unique to each coefficient (in their linear and quadratic form).
The `FullAnalysis.py` concludes the study by combining the Neural Networks trained on the full-feature datasets and the Nested Sampling algorithm used at the beginning. The NS parameter inference might require some more time to execute (on average less than 5 minutes) due to the complexity of the likelihood.

The report and executive summary can be found under `report`.

## Usage
The repository can be cloned into a local directory by typing the following on the command line:
```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/projects/lp645.git
```

The required packages are specified in the `environment.yml` file, and a `conda` environment can be created by running on the command line:
```bash
conda env create --name findNP --file environment.yml
```
The python files can be ran from the command line while in the `src` directory:
```bash
python src/<filename>.py
```
While the Jupyter notebooks can be ran from an IDE such as VisualStudio.

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

## Author
The source files, documentation and report were developed and written by Letizia Palmas (lp645)

#### Note on use of auto-generating tools
The CoPilot autocompletion feature was occasionally used for practical tasks where similar lines were repeated with previous checks of the correctness of the statements, for example to define the models for each coefficient:
```
model_c8dt = MLP(input_size=14).to(device)
model_c8qt = MLP(input_size=14).to(device)
model_c8qt2 = MLP(input_size=14).to(device)
model_c8dt2 = MLP(input_size=14).to(device)
``` 

ChatGPT was used in the report to query specific Latex features such as writing the algorithms in a suitable structure.