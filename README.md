# Secure Survival Analysis

This repository provides an implementation of several survival analysis tools using multiparty computation, based on the [MPyC](https://github.com/lschoe/mpyc) library. This includes the proportional hazards model and the concordance index, along with all supporting operations such as secure optimization, logarithms and secure versions of the BFGS and L-BFGS quasi-newton optimization methods.

<div align="center">

[![api](https://img.shields.io/badge/api-Python-blue?style=for-the-badge)](https://github.com/noahmr/secure_survival_analysis#usage)
[![license](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
</div>

## <div align="center">Features</div>

- Proportional hazards model, based on Efron's method
- Harrell's Concordance index
- Secure fixed-point exponentiation
- Secure fixed-point logarithm
- Secure group-by and group aggregation (e.g. sum, count)
- BFGS and L-BFGS Quasi-Newton optimization methods
- Gradient descent optimization method
- Extensive documentation available on all classes, methods and functions


## <div align="center">Install</div>

<details open>
<summary>Dependencies</summary>
 
- [MPyC](https://github.com/lschoe/mpyc)
- [gmpy2](https://pypi.org/project/gmpy2/)
- [Numpy](https://pypi.org/project/numpy/)

  
</details>

<details open>
<summary>Optional dependencies</summary>
 
- [SciPy](https://pypi.org/project/scipy/) (for statistical tests)
- [Pandas](https://pypi.org/project/pandas/) (for demos)

</details>


## <div align="center">About</div>

This implementation was developed as part of the master's thesis:

<em>Van der Meer, Noah. "Privacy-Preserving Survival Analysis" Master Thesis, Eindhoven University of Technology (2025).</em>



## <div align="center">License</div>

Copyright (c) 2025, Noah van der Meer

This software is licenced under the MIT license, which can be found in [LICENSE](LICENSE). By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.
