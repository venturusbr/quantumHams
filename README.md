# quantumHams

[![CI](https://github.com/venturus/quantumHams/actions/workflows/ci.yml/badge.svg)](https://github.com/venturus/quantumHams/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/quantumhams.svg)](https://pypi.org/project/quantumhams/)
[![Python](https://img.shields.io/pypi/pyversions/quantumhams.svg)](https://pypi.org/project/quantumhams/)

**quantumHams** is an extensible Python library for constructing and simulating Hamiltonians from different quantum-computing technologies. The first models cover:

- rubidium Rydberg neutral-atom arrays;
- ytterbium trapped-ion effective Ising/Mølmer–Sørensen systems.

The core implementation runs with NumPy/SciPy on CPU. When CUDA-Q is installed, quantumHams performs native GPU-accelerated time evolution through the CUDA-Q Dynamics target; otherwise it automatically falls back to SciPy.

## Installation

The Python import is `quantumhams`.

```bash
pip install quantumhams
```

Optional CUDA-Q integration on a supported Linux x86-64 environment:

```bash
pip install "quantumhams[cudaq]"
```

For development:

```bash
git clone https://github.com/venturus/quantumHams.git
cd quantumHams
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
pre-commit install
pytest
```

## Neutral atoms

```python
import numpy as np
from quantumhams import NeutralAtomRegister, RubidiumRydbergHamiltonian

register = NeutralAtomRegister.square(rows=2, columns=2, spacing=5.5)
model = RubidiumRydbergHamiltonian(
    atom_register=register,
    rabi_frequency=2 * np.pi * 1.0,
    detuning=2 * np.pi * 0.5,
    c6=2 * np.pi * 862_690.0,
)

matrix = model.matrix(sparse=True)
print(matrix.shape)
```

## Trapped ions

```python
import numpy as np
from quantumhams import YtterbiumIonChain, YtterbiumMølmerSørensenHamiltonian

chain = YtterbiumIonChain.linear(n_ions=3, spacing=5.0)
model = YtterbiumMølmerSørensenHamiltonian.power_law(
    chain=chain,
    coupling=2 * np.pi * 0.25,
    exponent=1.2,
    transverse_field=2 * np.pi * 0.1,
)

result = model.evolve(initial_state="000", times=np.linspace(0.0, 1.0, 11))
print(result.probabilities[-1])
```

## Backend selection

```python
from quantumhams.backends import backend_status, get_backend

print(backend_status())
backend = get_backend("auto")  # CUDA-Q when available, otherwise SciPy
```

With `backend="auto"` (the default), quantumHams first attempts native CUDA-Q Dynamics evolution and falls back to SciPy with a warning if CUDA-Q or a compatible GPU runtime is unavailable. Use `backend="cudaq"` to require CUDA-Q or `backend="scipy"` to force CPU execution.

## Extending quantumHams

Add a model by implementing `HamiltonianModel` and returning a `PauliHamiltonian` from `pauli_hamiltonian()`. This separates physics models from execution backends, making new technologies and accelerators independently pluggable.

See `docs/extending.md` and `CONTRIBUTING.md`.

## Governance and attribution

Author: **Daniel de Haro Moraes**  
Company: **Venturus**

Licensed under Apache License 2.0.
