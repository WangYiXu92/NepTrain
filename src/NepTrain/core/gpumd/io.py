#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 15:42
# @Author  : 兵
# @email    : 1747193328@qq.com

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ase.io import write as ase_write

from NepTrain import utils, Config
from NepTrain.exceptions import FileOperationError, GPUMDOutputError, GPUMDExecutionError
from .plot import *

logger = logging.getLogger(__name__)


class RunInput:
    """GPUMD run.in file handler.
    
    This class handles the GPUMD input file (run.in) format and provides
    methods to read, modify, and write GPUMD input files.
    
    Compatible with GPUMD API: https://github.com/brucefan1983/GPUMD
    """
    
    def __init__(self, nep_txt_path: str):
        """
        Initialize RunInput.
        
        Args:
            nep_txt_path: Path to nep.txt file
        """
        self.nep_txt_path = Path(nep_txt_path)
        self.command = Config.get('environ', 'gpumd_path')
        self.time_step = 1.0  # Default 1 fs
        self.dump_thermo = False
        self.dump_exyz = False
        self.total_time = 0
        self.run_in: list[list[str | list[str]]] = []

    def set_time_temp(
        self, 
        times: float | list[float] | None = None,
        temperature: float | list[float] | None = None
    ) -> None:
        """
        Set simulation time and temperature.
        
        Args:
            times: Simulation time in picoseconds (None to keep current)
            temperature: Temperature in Kelvin (None to keep current)
        """
        if times is not None:
            if isinstance(times, (int, float)):
                times = [times]
            self.total_time = 0
            
        for run_index in range(len(self.run_in)):
            run = self.run_in[run_index]
            if run[0] == "ensemble":
                if temperature is not None:
                    if isinstance(temperature, (int, float)):
                        temperature = [temperature]
                    # Handle multiple temperatures
                    if len(temperature) > 1 and run_index < len(temperature):
                        run[1][1] = str(temperature[run_index])
                        run[1][2] = str(temperature[run_index])
                    else:
                        run[1][1] = str(temperature[0] if isinstance(temperature, list) else temperature)
                        run[1][2] = str(temperature[0] if isinstance(temperature, list) else temperature)
                        
            elif run[0] == "run":
                if times is not None:
                    if isinstance(times, (int, float)):
                        times = [times]
                    # Handle multiple time steps
                    if len(times) > 1 and run_index < len(times):
                        run_time_ps = times[run_index]
                    else:
                        run_time_ps = times[0] if isinstance(times, list) else times
                    
                    # Convert ps to steps (time_step in fs)
                    nsteps = int(int(run_time_ps) * 1000 * 1 / self.time_step)
                    run[1][0] = str(nsteps)
                    self.total_time += nsteps

    def read_run(self, file_name: str) -> None:
        """
        Read GPUMD run.in file.
        
        Args:
            file_name: Path to run.in file
            
        Raises:
            FileOperationError: If file cannot be read
        """
        file_path = Path(file_name)
        if not file_path.exists():
            raise FileOperationError(
                f"run.in file not found: {file_path}",
                path=str(file_path)
            )
            
        self.run_in.clear()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Use regex to find all key-value pairs
                groups = re.findall(r"^([A-Za-z_]+)\s+(.*)", f.read(), re.MULTILINE)

                for group in groups:
                    key = group[0].strip()
                    value = group[1].strip()
                    
                    if not key:
                        continue
                        
                    if key == "time_step":
                        self.time_step = float(value)
                    elif key == "dump_thermo":
                        self.dump_thermo = True
                    elif key == "dump_exyz":
                        self.dump_exyz = True
                    elif key == "run":
                        try:
                            self.total_time += int(value)
                        except ValueError:
                            logger.warning(f"Invalid run value: {value}")
                            
                    self.run_in.append([key, [i for i in value.split(" ") if i.strip()]])
                    
        except IOError as e:
            raise FileOperationError(
                f"Failed to read run.in file: {file_path}",
                path=str(file_path)
            ) from e

    def write_run(self, file_name: str) -> None:
        """
        Write GPUMD run.in file.
        
        Args:
            file_name: Path to output run.in file
        """
        file_path = Path(file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for run_item in self.run_in:
                f.write(f"{run_item[0]}    {' '.join(run_item[1])}\n")

    def calculate(
        self, 
        atoms: Any, 
        directory: str,
        show_progress: bool = True
    ) -> None:
        """
        Run GPUMD calculation.
        
        Args:
            atoms: ASE Atoms object
            directory: Working directory for calculation
            show_progress: Whether to show progress (default: True)
            
        Raises:
            FileOperationError: If required files are missing
            GPUMDExecutionError: If GPUMD calculation fails
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Write run.in
        run_in_path = dir_path / "run.in"
        self.write_run(str(run_in_path))
        
        # Write model.xyz
        model_xyz_path = dir_path / "model.xyz"
        ase_write(str(model_xyz_path), atoms, format="extxyz")
        
        # Check and copy nep.txt
        if self.nep_txt_path is not None and self.nep_txt_path.exists():
            nep_txt_dest = dir_path / "nep.txt"
            if utils.is_diff_path(str(self.nep_txt_path), str(nep_txt_dest)):
                shutil.copy(self.nep_txt_path, str(nep_txt_dest))
        else:
            raise FileOperationError(
                f"nep.txt path is invalid: {self.nep_txt_path}",
                path=str(self.nep_txt_path)
            )

        # Run GPUMD
        gpumd_out = dir_path / "gpumd.out"
        gpumd_err = dir_path / "gpumd.err"

        try:
            with open(gpumd_out, "w") as f_std, open(gpumd_err, "w", buffering=1) as f_err:
                # Use list for command to avoid shell injection
                if isinstance(self.command, str):
                    result = subprocess.run(
                        [self.command],
                        shell=True,
                        stdout=f_std,
                        stderr=f_err,
                        cwd=str(dir_path),
                        check=False
                    )
                    return_code = result.returncode
                else:
                    # Command is a list
                    result = subprocess.run(
                        self.command,
                        stdout=f_std,
                        stderr=f_err,
                        cwd=str(dir_path),
                        check=False
                    )
                    return_code = result.returncode

            # Check return code
            if return_code != 0:
                raise GPUMDExecutionError(
                    f"GPUMD calculation failed with return code {return_code}",
                    return_code=return_code,
                    error_file=str(gpumd_err)
                )
                
            # Verify output files exist
            if self.dump_thermo:
                thermo_path = dir_path / "thermo.out"
                if not thermo_path.exists():
                    logger.warning(f"thermo.out not generated: {thermo_path}")

            if self.dump_exyz:
                dump_path = dir_path / "dump.xyz"
                if not dump_path.exists():
                    raise GPUMDOutputError(
                        "dump.xyz not generated by GPUMD",
                        expected_file=str(dump_path)
                    )
                    
        except subprocess.CalledProcessError as e:
            raise GPUMDExecutionError(
                f"GPUMD calculation subprocess failed",
                return_code=e.returncode,
                error_file=str(gpumd_err)
            ) from e


if __name__ == '__main__':
    # read_thermo("1.out",80)
    run = RunInput()
    run.read_run("./run.in")
    run.set_time_temp(1, 300)
