# Usage

## Rubidium neutral atoms

The implemented effective two-level model is

\[H = \frac{\Omega}{2}\sum_i X_i - \sum_i \Delta_i n_i + \sum_{i<j}\frac{C_6}{r_{ij}^6}n_i n_j.\]

All numerical parameters must use a mutually consistent unit system. quantumHams intentionally does not silently convert angular frequency, frequency, distance, or interaction units.

## Ytterbium trapped ions

The initial trapped-ion model is an effective long-range Ising/Mølmer–Sørensen Hamiltonian:

\[H = \sum_{i<j} J_{ij}X_iX_j + \sum_i B_i Z_i.\]

Use an explicit coupling matrix or construct a distance-dependent power-law approximation.

## Simulation

`HamiltonianModel.evolve()` accepts a bit string, basis-state integer, or normalized state vector and an ordered sequence of non-negative sample times.

## Native CUDA-Q Dynamics evolution

```python
import numpy as np
from quantumhams import NeutralAtomRegister, RubidiumRydbergHamiltonian

model = RubidiumRydbergHamiltonian(
    atom_register=NeutralAtomRegister.linear(2, 5.5),
    rabi_frequency=2 * np.pi,
    detuning=0.0,
    c6=2 * np.pi * 862_690.0,
)

# Requires CUDA-Q and a supported CUDA runtime. No SciPy delegation occurs.
result = model.evolve("00", np.linspace(0.0, 1.0, 101), backend="cudaq")
print(result.backend)       # cudaq
print(result.states.shape)  # (101, 4)

# Default: try CUDA-Q, then safely fall back to SciPy if runtime execution fails.
portable_result = model.evolve("00", np.linspace(0.0, 1.0, 101))
print(portable_result.backend)  # cudaq or scipy
```
