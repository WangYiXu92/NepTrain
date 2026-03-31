# NepTrain Perturb Module

Structure perturbation module for generating defective crystal structures.

## Supported Defect Types

- **Interstitial** - BCC/FCC/HCP/Diamond/Tetragonal
- **Twinning** - BCC/FCC/HCP/Diamond/Tetragonal  
- **Vacancy**
- **Substitution**
- **Displacement**

## Quick Start

### Interstitial Generation

```python
from ase.build import bulk
from NepTrain.core.perturb import InterstitialGenerator

# Create base structure
fe = bulk('Fe', 'bcc', a=4.0)

# Generate interstitial structure
gen = InterstitialGenerator(fe, 'C')
fe_with_c = gen.generate(n_interstitials=2, site_type='tetrahedral')

# Stream generation for large systems
for structure in gen.generate_stream(n_structures=10, n_interstitials=2):
    # Process each structure
    pass
```

### Twinning Generation

```python
from NepTrain.core.perturb import TwinningGenerator

al = bulk('Al', 'fcc', a=4.0)
gen = TwinningGenerator(al)
al_twin = gen.generate()
```

## Performance Tips

- Use `generate_stream()` for large systems to avoid memory overflow
- Use `generate_to_file()` for direct file output
- Enable caching for accelerated repeated calculations

## API Documentation

See [API Reference](docs/api.md) for complete details.

## Examples

Run the example scripts in the `examples/` directory:
- `01_basic_interstitial.py` - Basic interstitial generation
- `02_basic_twinning.py` - Basic twinning generation  
- `03_batch_generation.py` - Batch generation
- `04_streaming_generation.py` - Streaming for large systems
- `05_combined_defects.py` - Multiple defects

```python
# Run examples
python examples/01_basic_interstitial.py
python examples/02_basic_twinning.py
python examples/03_batch_generation.py
python examples/04_streaming_generation.py
python examples/05_combined_defects.py
```

## Crystal Structure Support

| Structure | Interstitial Sites | Twinning Law | Examples |
|-----------|-------------------|--------------|----------|
| BCC | Tetrahedral, Octahedral | {112}<111> | Fe, W, Ta |
| FCC | Tetrahedral, Octahedral | {111}<112> | Al, Cu, Ni |
| HCP | Tetrahedral, Octahedral | {10-12}<10-11> | Mg, Zn, Ti |
| Diamond | Tetrahedral, Hexagonal | {111}<112> | Si, Ge, C |
| Tetragonal | Tetrahedral, Octahedral | {101}<101> | Martensite |

## Installation

```bash
pip install -e .
```

## Requirements

- ASE >= 3.25.0
- NumPy >= 1.20.0
- Python >= 3.8

## License

MIT License
