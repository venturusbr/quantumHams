# quantumHams

[![CI](https://github.com/venturusbr/quantumHams/actions/workflows/ci.yml/badge.svg)](https://github.com/venturusbr/quantumHams/actions/workflows/ci.yml)
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
git clone https://github.com/venturusbr/quantumHams.git
cd quantumHams
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
pre-commit install
pytest
```

## Examples

### Neutral-atom Bell state

```python
import numpy as np

from quantumhams import NeutralAtomRegister, RubidiumRydbergHamiltonian

omega = 2.0 * np.pi
register = NeutralAtomRegister.linear(n_atoms=2, spacing=1.0)
model = RubidiumRydbergHamiltonian(
    atom_register=register,
    rabi_frequency=omega,
    detuning=0.0,
    c6=100.0 * omega,
)

times = np.linspace(0.0, np.pi / (np.sqrt(2.0) * omega), 101)
result = model.evolve(initial_state="00", times=times)

# Under Rydberg blockade, the final state approaches
# (|01> + |10>) / sqrt(2).
print(result.states[-1])
print(result.probabilities[-1])
```

### Trapped-ion Bell state

```python
import numpy as np

from quantumhams import YtterbiumIonChain, YtterbiumMolmerSorensenHamiltonian

coupling = 2.0 * np.pi
chain = YtterbiumIonChain.linear(n_ions=2, spacing=1.0)
model = YtterbiumMolmerSorensenHamiltonian.power_law(
    chain=chain,
    coupling=coupling,
    exponent=0.0,
    transverse_field=0.0,
)

times = np.linspace(0.0, np.pi / (4.0 * coupling), 101)
result = model.evolve(initial_state="00", times=times)

# The final state approaches (|00> - i|11>) / sqrt(2).
print(result.states[-1])
print(result.probabilities[-1])
```

The complete example supports both technologies and explicit CPU, GPU, or
automatic backend selection:

```bash
python examples/bell_state.py --technology neutral-atom --compute cpu
python examples/bell_state.py --technology trapped-ion --compute auto
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
