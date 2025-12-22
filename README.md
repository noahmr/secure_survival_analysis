# Secure Survival Analysis

This package provides an implementation of several survival analysis tools using multiparty computation, based on the [MPyC](https://github.com/lschoe/mpyc) library. This includes the proportional hazards model and the concordance index, along with supporting operations such as secure exponentiation, logarithms, and secure versions of the BFGS and L-BFGS quasi-newton optimization methods.

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

The package can be installed through pip using the following command:
```bash
pip3 install .
```

This should automatically install the mandatory dependencies.

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


## <div align="center">Demos</div>

There are several example scripts located within the [demos](demos) directory. This directory also includes an example dataset of 10.000 records, which was synthetically generated for testing purposes.

<details open>
<summary>Proportional hazards model</summary>
 
To fit the proportional hazards model on the first 100 records with 3 parties:

```bash
python3 test_model.py -M3 100
```
  
</details>

<details>
<summary>Concordance index</summary>
 
To compute the concordance index on the first 100 records with 3 parties:

```bash
python3 test_concordance.py -M3 100
```
  
</details>

<details>
<summary>Secure fixed-point exponentiation</summary>

Generate 100 random fixed-point exponents, and evaluate the exponential function on these with 3 parties:

```bash
python3 test_np_pow.py -M3 100 32
```
This uses a fixed-point bit-length of 32 bits.
  
</details>

<details>
<summary>Secure fixed-point logarithm</summary>

Generate 100 random fixed-point inputs, and compute the logarithms of these with 3 parties:

```bash
python3 test_np_logarithm.py -M3 100 32
```
This uses a fixed-point bit-length of 32 bits.
  
</details>


## <div align="center">About</div>

This implementation was developed as part of the master's thesis:

<em>Van der Meer, Noah. "Privacy-Preserving Survival Analysis" Master Thesis, Eindhoven University of Technology (2025).</em>

which was done under the supervision of Berry Schoenmakers.

## <div align="center">License</div>

Copyright (c) 2025, Noah van der Meer

This software is licenced under the MIT license, which can be found in [LICENSE](LICENSE). By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.
