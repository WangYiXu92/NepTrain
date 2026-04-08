#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 15:01
# @Author  : 兵
# @email    : 1747193328@qq.com


from NepTrain import utils
from .io import RunInput, PredictionRunInput
from .plot import plot_nep_result
from ..utils import check_env

def run_nep(argparse):
    check_env()
    if argparse.prediction:
        run = PredictionRunInput(argparse.nep_txt_path,argparse.train_path,argparse.nep_in_path,argparse.test_path)
    else:

        run = RunInput(argparse.train_path,argparse.nep_in_path,argparse.test_path)


    run.set_restart(argparse.restart_file,argparse.continue_step)
    run.calculate(argparse.directory)
    plot_nep_result(argparse.directory)
    utils.print_success("NEP training task completed!" )


def plot_nep_result_cli(argparse):
    """CLI entry point for plotting NEP training results."""
    check_env()
    utils.print_msg(f"Plotting NEP training results from {argparse.directory}")
    plot_nep_result(argparse.directory)
    utils.print_success("NEP result plot saved!")