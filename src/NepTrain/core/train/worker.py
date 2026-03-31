"""Job submission utilities for NepTrain.

This module provides synchronous and asynchronous job submission
using dpdispatcher for HPC cluster management.
"""

import asyncio
import logging
from pathlib import Path
from dpdispatcher import Machine, Resources, Task, Submission

logger = logging.getLogger(__name__)


def remove_sub_file(work_path: str = "./") -> None:
    """
    Remove temporary submission files from work directory.
    
    Args:
        work_path: Path to work directory (default: current directory)
    """
    work_dir = Path(work_path)
    # Remove .sub.* files
    all_files = list(work_dir.glob("????????????????????????????????????????.sub.*"))
    for file in all_files:
        try:
            file.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove file {file}: {e}")
    
    # Remove .sub files
    all_files = list(work_dir.glob("????????????????????????????????????????.sub"))
    for file in all_files:
        try:
            file.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove file {file}: {e}")


def submit_job(
    machine_dict: dict,
    resources_dict: dict,
    task_dict_list: list[dict],
    submission_dict: dict,
) -> Submission:
    """
    Submit a job synchronously using dpdispatcher.
    
    Args:
        machine_dict: Machine configuration dictionary
        resources_dict: Resources configuration dictionary
        task_dict_list: List of task configuration dictionaries
        submission_dict: Submission configuration dictionary
        
    Returns:
        Submission object with job results
    """
    machine = Machine.load_from_dict(machine_dict)
    resources = Resources.load_from_dict(resources_dict)
    task_list = [Task(**task_dict) for task_dict in task_dict_list]
    submission = Submission(
        machine=machine,
        resources=resources,
        task_list=task_list,
        **submission_dict,
    )
    submission.run_submission(clean=False)
    
    for job in submission.belonging_jobs:
        logger.info(f"Finished job {job.job_id} in {job.machine.context.remote_root}")

    return submission


async def async_submit_job(
    machine_dict: dict,
    resources_dict: dict,
    task_dict_list: list[dict],
    submission_dict: dict,
) -> None:
    """
    Submit a job asynchronously using dpdispatcher.
    
    Args:
        machine_dict: Machine configuration dictionary
        resources_dict: Resources configuration dictionary
        task_dict_list: List of task configuration dictionaries
        submission_dict: Submission configuration dictionary
    """
    machine = Machine.load_from_dict(machine_dict)
    resources = Resources.load_from_dict(resources_dict)
    task_list = [Task(**task_dict) for task_dict in task_dict_list]
    submission = Submission(
        machine=machine,
        resources=resources,
        task_list=task_list,
        **submission_dict,
    )
    await submission.async_run_submission(check_interval=1, clean=False)
    
    for job in submission.belonging_jobs:
        logger.info(f"Finished job {job.job_id} in {job.machine.context.remote_root}")
