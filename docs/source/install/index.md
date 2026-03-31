# Installation Guide




This guide explains how to install **NepTrain** using `pip`.

## Prerequisites

Before installing NepTrain, make sure you have the following:

- **Python**: NepTrain requires Python 3.8 or higher.  

## Installation Steps

1. Open a terminal or command prompt.
2. Run the following command to install NepTrain via `pip`:
   ```bash
   pip install neptrain
   ```
---

## Configuration File

The program's configuration file is located at `~/.NepTrain`. If the file does not exist, you can execute `NepTrain -h`, and the program will create it. 
:::{important}
You must modify the `potcar_path` to match the path of your pseudopotential files.  
If the executable programs are included in your system's environment variables, you do not need to modify their paths.
::: 

The default configuration is as follows:

```text
[environ]
# Pseudopotential file path
potcar_path = ~/POT_GGA_PAW_PBE_54
# VASP execution path
vasp_path = vasp_std

# mpirun execution path
mpirun_path = mpirun
# NEP execution path
nep_path = nep

# GPUMD execution path
gpumd_path = gpumd

[potcar]
# You can set the pseudopotential file version yourself
# If not set, the pseudopotential file version recommended by the VASP official website will be used
# e.g., Ag=Ag_sv
H = H
```


 
 