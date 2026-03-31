#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 14:33
# @Author  : 兵
# @email    : 1747193328@qq.com
import argparse
import sys
sys.path.append('../../')
from NepTrain.core import *
from NepTrain import __version__
import warnings
from dpdispatcher.dlog import dlog_stdout, dlog
dlog.removeHandler(dlog_stdout)
# 禁用所有 UserWarning
warnings.simplefilter('ignore', UserWarning)
def check_kpoints_number(value):
    """检查值是否为单个数字或三个数字的字符串"""

    if isinstance(value, str):
        values = value.split(',')

        if len(values) == 3 and all(v.isdigit() for v in values):
            return list(map(int, values))
        elif len(values) == 1 and value.isdigit():
            return [int(value),int(value),int(value)]
        else:
            raise argparse.ArgumentTypeError("The ka parameter must be a single number or three numbers separated by `,`.")
    elif isinstance(value, int):
        return value
    else:
        raise argparse.ArgumentTypeError("The ka parameter must be a single number or three numbers separated by `,`.")

def build_init(subparsers):
    parser_init = subparsers.add_parser(
        "init",
        help="Initialize some file templates",
    )
    parser_init.add_argument("type",
                             type=str,
                            choices=["bohrium","slurm","pbs","shell"],default="slurm",
                             help="How to call a task")

    parser_init.add_argument("-f", "--force", action='store_true',
                             default=False,
                             help="Force overwriting of generated templates"
                             )

    parser_init.set_defaults(func=init_template)



def build_perturb(subparsers):
    def perturb_cli_wrapper(args):
        # Wrapper to drive the generator
        # Map CLI args to run_perturb signature
        kwargs = vars(args).copy()
        
        # Remap arguments that differ between CLI and run_perturb
        if 'model_path' in kwargs:
            kwargs['input_file'] = kwargs.pop('model_path')
        if 'out_file_path' in kwargs:
            kwargs['output_file'] = kwargs.pop('out_file_path')
        if 'min_distance' in kwargs:
            kwargs['atom_pert_distance'] = kwargs.pop('min_distance')
            
        # Remove internal argparse arguments
        if 'func' in kwargs:
            del kwargs['func']

        for _ in run_perturb(**kwargs):
            pass

    parser_perturb = subparsers.add_parser(
        "perturb",
        help="Generate perturbed structures.",
    )

    parser_perturb.set_defaults(func=perturb_cli_wrapper)

    parser_perturb.add_argument("model_path",
                             type=str,

                             help="The structure path or structure file required for calculation only supports files in xyz and vasp formats.")
    parser_perturb.add_argument("--num","-n",
                             type=int,
                                default=20,
                             help="The number of perturbations for each structure, if a folder is input, the final number generated should be the number of structures multiplied by num.default 20.")

    parser_perturb.add_argument("--cell", "-c",
                                dest="cell_pert_fraction",
                                type=float,
                                default=0.03,
                                help="The deformation ratio,default 0.03.")

    parser_perturb.add_argument("--distance", "-d",
                                type=float,
                                dest="min_distance",
                                default=0.1,
                                help="Min atom distance, unit Å, default 0.1.")

    parser_perturb.add_argument("--out", "-o",
                             dest="out_file_path",
                             type=str,
                             help="Output file for perturbed structures, default ./perturb.xyz.",
                             default="./perturb.xyz"
                             )
    parser_perturb.add_argument("--append", "-a",
                             dest="append", action='store_true', default=False,
                             help="Write to out_file_path in append mode, default False.",

                             )
    
    parser_perturb.add_argument("--sampler",
                                dest="sampler",
                                type=str,
                                default="random",
                                choices=["random", "sobol"],
                                help="Sampling method for perturbation parameters (default: random).")

    parser_perturb.add_argument("--no-scramble",
                                dest="scramble",
                                action='store_false',
                                default=True,
                                help="Disable scrambling for Sobol sequence (default: scrambling enabled).")

    parser_perturb.add_argument("--state-file",
                                dest="state_file",
                                type=str,
                                default=None,
                                help="JSON file to save/load Sobol sequence state.")

    parser_perturb.add_argument("--resume",
                                dest="resume",
                                action='store_true',
                                default=False,
                                help="Resume Sobol sequence from state file (default: False).")

    parser_perturb.add_argument("--seed",
                                dest="seed",
                                type=int,
                                default=None,
                                help="Random seed for Sobol sequence (default: None).")

    parser_perturb.add_argument("--mag-mode",
                                dest="mag_mode",
                                type=str,
                                default=None,
                                choices=["collinear", "random_collinear", "non_collinear"],
                                help="Magnetic perturbation mode. 'collinear': fixed moments; 'random_collinear': random spin flips; 'non_collinear': random vectors.")

    parser_perturb.add_argument("--mag-flip-prob",
                                dest="mag_flip_prob",
                                type=float,
                                default=0.5,
                                help="Probability of flipping spin in random_collinear mode (default 0.5).")

    parser_perturb.add_argument("--mag-noise",
                                dest="mag_noise",
                                type=float,
                                default=0.0,
                                help="Gaussian noise added to magnetic moment magnitudes (default 0.0).")

    parser_perturb.add_argument("--rotate-formula",
                                dest="rotate_formula",
                                type=str,
                                default=None,
                                help="Chemical formula of fragments to rotate randomly (e.g. 'H2O').")
    
    # Vacancy arguments
    parser_perturb.add_argument("--vac-elements",
                                dest="vac_elements",
                                type=str,
                                default=None,
                                help="Elements to consider for vacancies (comma-separated, e.g. 'Fe,O').")
    
    parser_perturb.add_argument("--vac-num",
                                dest="vac_num",
                                type=int,
                                default=0,
                                help="Number of vacancies to generate (default 0).")

    # Shuffle arguments
    parser_perturb.add_argument("--shuffle-elements",
                                dest="shuffle_elements",
                                type=str,
                                default=None,
                                help="Elements or indices to shuffle (e.g. 'Fe,O' or '0:10').")

    parser_perturb.add_argument("--shuffle-method",
                                dest="shuffle_method",
                                type=str,
                                default="fisher_yates",
                                choices=["fisher_yates", "random_swap"],
                                help="Shuffling method (default: fisher_yates).")

    parser_perturb.add_argument("--skip-normal",
                                dest="skip_normal",
                                action='store_true',
                                default=False,
                                help="Skip normal perturbation (cell deformation and rattle) steps. Default False.")

    # Rigid Body arguments
    parser_perturb.add_argument("--rigid",
                                dest="rigid",
                                action='store_true',
                                default=False,
                                help="Enable rigid body perturbation mode. Molecules/clusters are moved/rotated as rigid bodies.")

    parser_perturb.add_argument("--rigid-method",
                                dest="rigid_method",
                                type=str,
                                default='auto',
                                choices=['auto', 'manual', 'file', 'composition'],
                                help="Method to define rigid bodies: 'auto' (connectivity), 'manual' (CLI list), 'file' (read 'rigid_id' from extxyz), 'composition' (formula). Default 'auto'.")

    parser_perturb.add_argument("--rigid-composition",
                                dest="rigid_composition",
                                type=str,
                                default=None,
                                help="Comma-separated list of allowed formulas for rigid bodies (e.g., 'H2O,CO2'). Used if method='composition'.")

    parser_perturb.add_argument("--rigid-list",
                                dest="rigid_list",
                                type=str,
                                default=None,
                                help="Manual list of rigid bodies indices (e.g. '0-3,4-7'). Used if method='manual'.")

    parser_perturb.add_argument("--rigid-mode",
                                dest="rigid_mode",
                                type=str,
                                default='inter',
                                choices=['inter', 'intra'],
                                help="Rigid perturbation mode: 'inter' (move bodies, fixed internal), 'intra' (fixed bodies, perturb internal). Default 'inter'.")

    parser_perturb.add_argument("--debug-plot",
                                dest="debug_plot",
                                action='store_true',
                                default=False,
                                help="Generate debug comparison plots (first 5 frames).")

    # Surface arguments
    parser_perturb.add_argument("--surface",
                                dest="surface",
                                action='store_true',
                                default=False,
                                help="Generate surface slabs (default False).")

    parser_perturb.add_argument("--indices",
                                dest="surface_indices",
                                type=str,
                                default="1,1,1",
                                help="Miller indices for surface generation (e.g. '1,1,1').")

    parser_perturb.add_argument("--vacuum",
                                dest="surface_vacuum",
                                type=float,
                                default=10.0,
                                help="Vacuum size in Angstroms for surface generation (default 10.0).")

    parser_perturb.add_argument("--layers",
                                dest="surface_layers",
                                type=int,
                                default=3,
                                help="Number of atomic layers for surface generation (default 3).")

    # Grain Boundary arguments
    parser_perturb.add_argument("--gb",
                                dest="gb",
                                action='store_true',
                                default=False,
                                help="Generate grain boundary (default False).")

    parser_perturb.add_argument("--gb-axis",
                                dest="gb_axis",
                                type=str,
                                default="0,0,1",
                                help="Rotation axis for grain boundary (e.g. '0,0,1').")

    parser_perturb.add_argument("--gb-angle",
                                dest="gb_angle",
                                type=float,
                                default=30.0,
                                help="Rotation angle in degrees for grain boundary (default 30.0).")

    parser_perturb.add_argument("--gb-dist",
                                dest="gb_dist",
                                type=float,
                                default=2.0,
                                help="Minimum distance for overlap removal in GB (default 2.0).")

    parser_perturb.add_argument("--no-gb-overlap",
                                dest="gb_delete_overlap",
                                action='store_false',
                                default=True,
                                help="Disable overlap removal for grain boundary (default: removal enabled).")

    # Dislocation arguments
    parser_perturb.add_argument("--dislocation",
                                dest="dislocation",
                                action='store_true',
                                default=False,
                                help="Generate dislocation (default False).")
    
    parser_perturb.add_argument("--dislocation-type",
                                dest="dislocation_type",
                                type=str,
                                default="edge",
                                choices=["edge", "screw", "random"],
                                help="Dislocation type: 'edge', 'screw' or 'random' (default 'edge').")

    parser_perturb.add_argument("--dislocation-axis",
                                dest="dislocation_axis",
                                type=str,
                                default='0,0,1',
                                help="Dislocation line axis vector (e.g. '0,0,1').")
    
    parser_perturb.add_argument("--dislocation-burgers",
                                dest="dislocation_burgers",
                                type=str,
                                default='1,0,0',
                                help="Burgers vector for dislocation (e.g. '1,0,0').")

    # Twinning arguments
    parser_perturb.add_argument("--twinning",
                                dest="twinning",
                                action='store_true',
                                default=False,
                                help="Generate twin boundary (default False).")
    
    parser_perturb.add_argument("--twinning-indices",
                                dest="twinning_indices",
                                type=str,
                                default="1,1,1",
                                help="Miller indices for twin plane (default '1,1,1').")

    parser_perturb.add_argument("--twinning-z",
                                dest="twinning_z",
                                type=float,
                                default=0.5,
                                help="Fractional height for twin boundary (default 0.5).")

    parser_perturb.add_argument("--twinning-min-dist",
                                dest="twinning_min_dist",
                                type=float,
                                default=1.5,
                                help="Minimum distance for overlap removal in twinning (default 1.5).")

    # Stacking Fault arguments
    parser_perturb.add_argument("--stacking-fault",
                                dest="stacking_fault",
                                action='store_true',
                                default=False,
                                help="Generate stacking fault (default False).")

    parser_perturb.add_argument("--sf-indices",
                                dest="sf_normal",
                                type=str,
                                default="1,1,1",
                                help="Miller indices for stacking fault plane (default '1,1,1').")

    parser_perturb.add_argument("--sf-shift",
                                dest="sf_shift",
                                type=str,
                                default="random",
                                help="Shift vector for stacking fault or 'random' (default 'random').")

    parser_perturb.add_argument("--sf-height",
                                dest="sf_height",
                                type=str,
                                default="random",
                                help="Fractional height for stacking fault plane or 'random' (default 'random').")

    parser_perturb.add_argument("--sf-min-dist",
                                dest="sf_min_dist",
                                type=float,
                                default=1.5,
                                help="Minimum distance for overlap removal in stacking fault (default 1.5).")

    # Amorphous arguments
    parser_perturb.add_argument("--amorphous",
                                dest="amorphous",
                                action='store_true',
                                default=False,
                                help="Generate amorphous structure (default False).")
    
    parser_perturb.add_argument("--amorphous-min-dist",
                                dest="amorphous_min_dist",
                                type=float,
                                default=1.5,
                                help="Minimum distance for amorphous generation (default 1.5).")

    parser_perturb.add_argument("--amorphous-rattle",
                                dest="amorphous_rattle",
                                type=float,
                                default=0.5,
                                help="Rattle strength for amorphous generation (default 0.5).")

    parser_perturb.add_argument("--amorphous-steps",
                                dest="amorphous_steps",
                                type=int,
                                default=100,
                                help="Relaxation steps for amorphous generation (default 100).")




def build_vasp(subparsers):
    parser_vasp = subparsers.add_parser(
        "vasp",
        help="Calculate single-point energy using VASP.",
    )
    parser_vasp.set_defaults(func=run_vasp)

    parser_vasp.add_argument("model_path",
                             type=str,

                             help="The required structure path or structure file only supports files in xyz and vasp formats.")
    parser_vasp.add_argument("--directory", "-dir",

                             type=str,
                             help="Set the VASP calculation path. default ./cache/vasp.",
                             default="./cache/vasp"
                             )

    parser_vasp.add_argument("--out", "-o",
                             dest="out_file_path",
                             type=str,
                             help="Structure output file after calculation. default ./vasp_scf.xyz",
                             default="./vasp_scf.xyz"
                             )

    parser_vasp.add_argument("--append", "-a",
                             dest="append", action='store_true', default=False,
                             help="Write to out_file_path in append mode, default False.",

                             )
    parser_vasp.add_argument("--gamma", "-g",
                             dest="use_gamma", action='store_true', default=False,
                             help="Default to using Monkhorst-Pack k-points, add -g to use Gamma-centered k-point scheme.",

                             )
    parser_vasp.add_argument("-n", "-np",
                             dest="n_cpu",
                             default=1,
                             type=int,
                             help="Set the number of CPU cores, default 1.")

    parser_vasp.add_argument("--incar",

                             help="Input path for INCAR file, default is ./INCAR.",default="./INCAR")
    
    parser_vasp.add_argument("--mag",
                             dest="use_mag",
                             action='store_true',
                             default=False,
                             help="Enable magnetic moments in the calculation (default: False)."
                             )
    
    parser_vasp.add_argument("--mag-relax",
                             dest="mag_relax",
                             action='store_true',
                             default=False,
                             help="Allow magnetic moment direction/size relaxation (disables I_CONSTRAINED_M=1). Default: False (constrained)."
                             )



    k_group = parser_vasp.add_mutually_exclusive_group(required=False)
    k_group.add_argument("--kspacing", "-kspacing",

                         type=float,
                         help="Set kspacing, which can also be defined in the INCAR template.")
    k_group.add_argument("--ka", "-ka",
                         default=[1, 1, 1],
                         type=check_kpoints_number,
                         help="ka takes 1 or 3 numbers (comma-separated), sets k-points to (k[0]/a, k[1]/b, k[2]/c). default 1.")
def build_dft(subparsers):
    parser_dft = subparsers.add_parser(
        "dft",
        help="Calculate single-point energy using DFT software.",
    )
    parser_dft.set_defaults(func=run_dft)

    parser_dft.add_argument("model_path",
                             type=str,

                             help="The required structure path or structure file only supports files in xyz and vasp formats.")
    parser_dft.add_argument("--directory", "-dir",

                             type=str,
                             help="Set the VASP calculation path. default ./cache/software.",
                             default=None
                             )

    parser_dft.add_argument("--out", "-o",
                             dest="out_file_path",
                             type=str,
                             help="Structure output file after calculation. default ./software_scf.xyz",
                             default=None
                             )

    parser_dft.add_argument("--append", "-a",
                             dest="append", action='store_true', default=False,
                             help="Write to out_file_path in append mode, default False.",

                             )
    parser_dft.add_argument("--gamma", "-g",
                             dest="use_gamma", action='store_true', default=False,
                             help="Default to using Monkhorst-Pack k-points, add -g to use Gamma-centered k-point scheme.",

                             )
    parser_dft.add_argument("-n", "-np",
                             dest="n_cpu",
                             default=1,
                             type=int,
                             help="Set the number of CPU cores, default 1.")

    parser_dft.add_argument("--in", "--incar",
                                dest="incar",
                             help="Input path for INCAR/INPUT file, default is ./INCAR or ./INPUT.",default=None)

    parser_dft.add_argument("--mag",
                             dest="use_mag",
                             action='store_true',
                             default=False,
                             help="Enable magnetic moments in the calculation (default: False)."
                             )

    parser_dft.add_argument("--mag-relax",
                             dest="mag_relax",
                             action='store_true',
                             default=False,
                             help="Allow magnetic moment direction/size relaxation (disables I_CONSTRAINED_M=1). Default: False (constrained)."
                             )

    k_group = parser_dft.add_mutually_exclusive_group(required=False)
    k_group.add_argument("--kspacing", "-kspacing",

                         type=float,
                         help="Set kspacing, which can also be defined in the INCAR template.")
    k_group.add_argument("--ka", "-ka",
                         default=[1, 1, 1],
                         type=check_kpoints_number,
                         help="ka takes 1 or 3 numbers (comma-separated), sets k-points to (k[0]/a, k[1]/b, k[2]/c). default 1.")

    software_group = parser_dft.add_mutually_exclusive_group(required=False)
    software_group.add_argument("--vasp" ,
                                dest="software",

                                action='store_const', const='vasp',
                         help="use vasp.(default)")
    software_group.add_argument("--abacus",
                                dest="software",
                                action='store_const', const='abacus',
                                help="use abacus")



def build_nep(subparsers):
    parser_nep = subparsers.add_parser(
        "nep",
        help="Train potential functions using NEP.",
    )
    parser_nep.set_defaults(func=run_nep)


    parser_nep.add_argument("--directory", "-dir",
                             type=str,
                             help="Set the path for NEP calculations. default ./cache/nep",
                             default="./cache/nep"
                             )

    parser_nep.add_argument("--in", "-in",
                            dest="nep_in_path",
                             type=str,
                             help="Set the path for the nep.in file; if not present, generate it based on train.xyz. default ./nep.in",
                             default="./nep.in"
                             )

    parser_nep.add_argument("--train", "-train",
                             dest="train_path",

                             type=str,
                             help="Set the path for the train.xyz file, default  ./train.xyz.",
                             default="./train.xyz"
                             )

    parser_nep.add_argument("--test", "-test",
                             dest="test_path",
                             type=str,
                             help="Set the path for the test.xyz file, default is ./test.xyz.",
                             default="./test.xyz"
                             )

    parser_nep.add_argument("--nep", "-nep",
                            dest="nep_txt_path",
                             type=str,
                             help="restart and prediction require the use of a potential function, default is ./nep.txt.",
                             default="./nep.txt"
                             )

    parser_nep.add_argument("--prediction", "-pred","--pred",

                             action="store_true",
                             help="Set the forecast mode, default False",
                             default=False
                             )

    parser_nep.add_argument("--restart_file", "-restart","--restart",

                            type=str,

                            help="To restart running, simply provide a valid path; default is None.",
                             default=None
                             )

    parser_nep.add_argument("--continue_step", "-cs",
                            type=int,
                            help="If a restart_file is provided, this parameter will take effect, continuing for continue_step steps, with a default value of 10000.",
                             default=10000
                             )



def build_gpumd(subparsers):
    parser_gpumd = subparsers.add_parser(
        "gpumd",
        help="run molecular dynamics using GPUMD.",
    )
    parser_gpumd.set_defaults(func=run_gpumd)

    parser_gpumd.add_argument("model_path",
                             type=str,

                             help="The required structure path or structure file only supports files in xyz and vasp formats.")
    parser_gpumd.add_argument("--directory", "-dir",

                             type=str,
                             help="Set the GPUMD calculation path, default is ./cache/gpumd.",
                             default="./cache/gpumd"
                             )
    parser_gpumd.add_argument("--in","-in",dest="run_in_path", type=str,
                              help="The filename for the command _template file, default is ./run.in.", default="./run.in")

    parser_gpumd.add_argument("--nep", "-nep",
                            dest="nep_txt_path",
                             type=str,
                             help="Potential function path, default is ./nep.txt.",
                             default="./nep.txt"
                             )
    parser_gpumd.add_argument("--time", "-t", type=int, help="Molecular dynamics time, unit ps, default 10 ps.", default=10)
    parser_gpumd.add_argument("--temperature", "-T", type=int, help="Molecular dynamics temperature in Kelvin,multiple integers can be input. default is 300 K", nargs="*", default=[300])

    parser_gpumd.add_argument("--out", "-o",
                               dest="out_file_path",

                               type=str,
                               default="./trajectory.xyz",
                               help="Output path for structures."
                               )

def build_train(subparsers):
    parser_train = subparsers.add_parser(
        "train",
        help="Automatic training.",
    )
    parser_train.set_defaults(func=train_nep)

    parser_train.add_argument("config_path",
                             type=str,

                             help="The required configuration file path (YAML format).")


def build_select(subparsers):
    parser_select = subparsers.add_parser(
        "select",
        help="Select samples.",
    )
    parser_select.set_defaults(func=run_select,decomposition='pca')

    parser_select.add_argument("trajectory_paths",
                              nargs="+",
                             help="The trajectory files needed for sampling (xyz format).")


    parser_select.add_argument("--base", "-base",
                               type=str,
                               default="train.xyz",
                               help="Provide a path to base.xyz, and sample the trajectory based on base.xyz, default is train.xyz."
                               )
    parser_select.add_argument("--nep", "-nep",
                               type=str,
                               default=None,
                               help="Provide a path to a nep.txt file, or 'nep89' to force universal model. If not provided, tries ./nep.txt, then universal NEP89, then falls back to SOAP."
                               )
    parser_select.add_argument("--max_selected", "-max", type=int,
                               help="Maximum number of structures to select, default is 20.",
                               default=20)
    parser_select.add_argument("--min_distance","-d", type=float,
                               help="Minimum bond length for farthest-point sampling, default is 0.01.",
                               default=0.01)
    parser_select.add_argument("--filter", "-f", type=float,
                               const=0.6,nargs='?',
                               help="Whether to filter based on covalent radius, the default is False. If True, the default coefficient is 0.6, and a coefficient can be passed in",
                               default=False)

    dc_group = parser_select.add_mutually_exclusive_group(required=False)
    dc_group.add_argument('-pca',"--pca", action='store_const', const='pca', dest='decomposition',
                       help='Use PCA for decomposition')
    dc_group.add_argument('-umap',"--umap", action='store_const', const='umap', dest='decomposition',
                       help='Use UMAP for decomposition')

    parser_select.add_argument("--out", "-o",
                               dest="out_file_path",

                               type=str,
                               default="./selected.xyz",
                               help="Output path for selected structures.default ./selected.xyz"
                               )

    group= parser_select.add_argument_group("SOAP","SOAP Parameters")

    group.add_argument("--r_cut", "-r", type=float, help="A cutoff for local region in angstroms,default 6", default=6)
    group.add_argument("--n_max", "-n", type=int, help="The number of radial basis functions,default 8", default=8)
    group.add_argument("--l_max", "-l", type=int, help="The maximum degree of spherical harmonics,default 6", default=6)

def main():
    parser = argparse.ArgumentParser(
        description="""
        NepTrain is a tool for automatically training NEP potential functions""",

    )
    parser.add_argument(
        "-v", "--version", action="version", version=__version__
    )



    subparsers = parser.add_subparsers()


    build_init(subparsers)

    build_perturb(subparsers)

    build_select(subparsers)
    build_dft(subparsers)

    build_vasp(subparsers)

    build_nep(subparsers)
    build_gpumd(subparsers)
    build_train(subparsers)



    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:

        pass


    args = parser.parse_args()

    try:
        _ = args.func
    except AttributeError as exc:
        parser.print_help()
        raise SystemExit("Please specify a command.") from exc
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
