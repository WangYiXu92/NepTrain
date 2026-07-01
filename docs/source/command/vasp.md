# vasp
**Description:**  
Performs single-point energy calculations using VASP.
## Input Parameters
**Usage:**  
```bash
NepTrain vasp <model_path> [options]
```
:::{tip}
The Vasp calculation allows the use of the `INCAR` file as a template, which enables users to conveniently use the parameters one desire. When using the `INCAR` file template, it is usually necessary to pay attention to the consistency of the k-point density. Therefore, it is recommended to specify the `KSPACING` parameter in the INCAR file. For instance, add `KSPACING=0.2` into the `INCAR` file.
::: 

**Options:**  
- `<model_path>`  
  Structure path or file (supports `xyz` and `vasp` formats).
- `-dir, --directory`  
  Directory for VASP calculations. Default: `./cache/vasp`.
- `-o, --out`  
  Output file path for calculated structure. Default: `./vasp_scf.xyz`.
- `-a, --append`  
  Append to output file. Default: `False`.
- `-g, --gamma`  
  Use Gamma-centered k-point scheme. Default: `False`.
- `-n, -np`  
  Number of CPU cores. Default: `1`.
- `--incar`  
  Path to INCAR file. Default: `./INCAR`.
- `--mag`
  Enable magnetic moments in the calculation. Default: `False`.
  **Note**: By default, `I_CONSTRAINED_M=1` (constrained direction) is set to preserve the initial magnetic configuration.
  If the input structure has no magnetic moments, default values will be loaded from the `[magmom]` section in `.NepTrain` or `config.ini` (e.g., `Fe = 2.2`).
  Magnetic outputs are exported with vector extended-XYZ arrays `spin:R:3`,
  `moment:R:3`, and `torque:R:3`. `spin` records the constrained/input spin
  vector, `moment` records DFT per-atom magnetic moments when available, and
  `torque` is written as DFT torques when available or zero vectors otherwise.
- `--mag-relax`
  Allow magnetic moment direction/size relaxation (disables `I_CONSTRAINED_M=1`). Use this if you want the magnetic moments to relax to the ground state.
- `-kspacing`  
  Set k-spacing value.
- `-ka`  
  Set k-points as 1 or 3 numbers (comma-separated). Default: `[1, 1, 1]`.

## Output
The Vasp calculation will be invoked to obtain the single-point energy. The results are output by default in the `./cache/vasp` folder, and the terminal output will be directed to the vasp.out file in the corresponding folder.

## Default INCAR
```text
SYSTEM = NepTrain-default-incar
ALGO = Normal
EDIFF = 1e-06
EDIFFG = -0.01
ENCUT = 500
GGA = PE
IBRION = -1
ISMEAR = 0
ISPIN = 1
ISTART = 0
LCHARG = False
LREAL = Auto
LWAVE = False
NELM = 100
NPAR = 4
NSW = 0
PREC = Normal
SIGMA = 0.05
```

## Example
In the "structure" folder under the current directory, there are files named structure1.xyz, structure2.xyz, ..., structuren.xyz. To perform single-point energy calculations of these structures, you need to run
```shell
NepTrain vasp structure -g
```
Here, `structure` specifies the folder, which will calculate all  `.xyz` or `.vasp` files in the folder. The `-g` flag indicates adding additional parameters, which in this case means using a gamma-centered k-point grid.

:::{note}
If `./INCAR` is an invalid file path, the default `INCAR` will be used.
:::
