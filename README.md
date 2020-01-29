# d-FuzzStream

Python implementation of the fuzzy data stream clustering algorithm from the article [d-FuzzStream: A Dispersion-Based Fuzzy Data Stream Clustering](https://doi.org/10.1109/FUZZ-IEEE.2018.8491534).

## Content of repository
* [Source code](d_fuzzstream/__init__.py)
* [Review of the article (russian)](report/main.pdf)
* [Testing code (demo generator)](make_gif.py)

## Demonstration
Colored dots represent data stream examples from the dataset [Bench1_11k](https://github.com/vpozdnyakov/DS_Datasets/tree/master/Synthetic/Non-Stationary/Bench1_11k). Blue circles represent FMiCs, and their radiusis are calculated with respect to the Algorithm 1 from the article — the minimum distance to the nearest FMiC’s prototype if N=1 and the fuzzy dispersion otherwise.
![online clustering](gif/ds_demo.gif)