# Improved Partitioned Bloom Filters

## Overview

This project contains three Python files that implement Turbo Partitioned Learned Bloom Filters (TurboPLBF) and Differentiable Partitioned Learned Bloom Filters (DiffPLBF). The files included are:

- `turbo_structs.py`: Contains utility functions and data structures used by both TurboPLBF and DiffPLBF.
- `turbo_plbf.py`: Implements the TurboPLBF construction and evaluation.
- `diff_plbf.py`: Implements the DiffPLBF construction and evaluation.

## Files and Functions

### turbo_structs.py

This file provides several utility functions and classes:

- `create_ideal_g_h(N)`: Generates ideal probability mass functions `g` and `h` for a given `N`.
- `simple_g_h(N)`: Generates simple linear probability mass functions `g` and `h` for a given `N`.
- `DataLoader`: A class that loads and prepares probability mass functions `g` and `h` based on different distributions.
  - `load()`: Returns the generated `g`, `h`, and their cumulative sums `pg`, `ph`.
- `Node`: A class representing a node in a doubly linked list.
- `LinkedList`: A class implementing a doubly linked list with various operations like `insert`, `remove`, and `fetch` functions.
- `ImplicitMatrix`: A class representing a matrix whose values are derived from a function. Includes methods to fetch elements and delete rows/columns.

### turbo_plbf.py

This file implements the TurboPLBF algorithm:

- `optFPR(ts)`: Calculates the optimal False Positive Rates (FPR) for given thresholds `ts`.
- `normBloom()`: Returns the space used by a normal Bloom Filter.
- `spaceUsed(ts)`: Returns the space used by the TurboPLBF for given thresholds `ts`.
- `dkl(x, y)`: Computes the Kullback-Leibler divergence between two indices.
- `reduce(mat_a)`: Reduces the implicit matrix `mat_a` to find relevant partitions.
- `max_compute(mat_a)`: Computes the maximum values in the implicit matrix `mat_a`.
- Main section: Initializes parameters, loads data, and constructs the TurboPLBF. Prints the construction time, partition score, and space used.

### diff_plbf.py

This file implements the DiffPLBF algorithm:

- `optFPR(ts)`: Calculates the optimal False Positive Rates (FPR) for given thresholds `ts`.
- `normBloom()`: Returns the space used by a normal Bloom Filter.
- `spaceUsed(ts)`: Returns the space used by the DiffPLBF for given thresholds `ts`.
- `dkl(l, r)`: Computes the Kullback-Leibler divergence between two indices.
- `F_func(x)`: Evaluates the function `F` at point `x`.
- `G(x)`: Evaluates the constraint function `G` at point `x`.
- `get_rand_pt(k)`: Generates a random point of dimension `k`.
- `grad(func, pt)`: Computes the gradient of `func` at point `pt`.
- Main section: Initializes parameters, loads data, and constructs the DiffPLBF. Prints the construction time, partition score, and space used.

## How to Run

1. Ensure you have Python installed on your system. These programs require Python 3 and the following libraries:
   - `numpy`

2. To run the TurboPLBF program:
   ```sh
   python turbo_plbf.py
   ```

3. To run the DiffPLBF program:
   ```sh
   python diff_plbf.py
   ```

4. Both scripts will output the construction time, partition score, and space used by the respective PLBF implementations.

## Example

Running `python turbo_plbf.py` will output:
```
Construction Time: XXXXms
Score of partition: X.XXXX
Space Used (Bloom Filter): X.XXXX Mb
Space Used (TurboPLBF): X.XXXX Mb
```

Running `python diff_plbf.py` will output:
```
Construction Time: XXXXms
Score of partition: X.XXXX
Space Used (Bloom Filter): X.XXXX Mb
Space Used (DiffPLBF): X.XXXX Mb
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
