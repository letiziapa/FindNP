"""@mainpage Project documentation
This documentation provides the functions required to infer the presence of new interactions 
between fundamental particles in a toy dataset. These new interactions are parametrised by 2 Wilson coefficients,
whose values can be inferred by evaluating the maximum likelihood points of the distributions.

@section overview Overview
We present a comprehensive overview of the modules and functions used throughout the project, 
from traditional evaluation methods to advanced machine learning training and parameter inference.

@section Funcs Module: Funcs.py
This module contains the main functions used in the project. These can be implemented throughout
different parts of the analysis, including:
- Traditional statistical analysis and assessment of the level of confidence with the results
- Machine Learning training at the parton level
- Combined Neural Network training with parameter inference through Nested Sampling

@subsection Cross-sections
The remaining files contain functions required to compute and validate the particle cross-sections at the parton level, 
which represents a simplified scenario for which we know the analytical form of the likelihood.
These are crucial to assess the performance of the models,
serving as a reference for the expected model behaviour.
These were not developed by the author, but were provided at the beginning of the project.


For more details on each module and its functions, refer to the respective documentation sections.
"""