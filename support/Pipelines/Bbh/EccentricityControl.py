# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import subprocess
import time
from pathlib import Path

import click
import numpy as np

import spectre.IO.H5 as spectre_h5
from spectre.Pipelines.Bbh.InitialData import generate_id
from spectre.Pipelines.EccentricityControl.EccentricityControl import (
    coordinate_separation_eccentricity_control,
)
from spectre.support.Schedule import schedule, scheduler_options
from spectre.Visualization.ReadH5 import available_subfiles, to_dataframe

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Predefined values
DEFAULT_RESIDUAL_TOLERANCE = 1e-7
DEFAULT_MAX_ITERATIONS = 4
w0 = 0.015264577062
a0 = -4.61484576462e-05
L = 1
P1 = 4
P2 = 6
D = 12
bbh_id_dir0 = "./bbh_id_ecc_ctrl"
filename = "BbhReductions.h5"
AhARecord = "/ApparentHorizons/ControlSystemAhA_Centers.dat"
AhBRecord = "/ApparentHorizons/ControlSystemAhB_Centers.dat"


def wait_for_file(filepath, interval=120):
    """Wait for a file to exist with a given check interval."""
    while not os.path.exists(filepath):
        logger.info(f"Waiting for file: {filepath}")
        time.sleep(interval)


def run_subprocess(command):
    """Run a subprocess command and handle errors."""
    try:
        result = subprocess.run(
            command, shell=False, capture_output=True, text=True, check=True
        )
        logger.info(f"Command succeeded: {' '.join(command)}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Subprocess failed with error: {e}")
        raise


def autoecc_control(
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
):
    """
    Function to run iterative eccentricity minimization.
    It stops when eccentricity is below residual_tolerance or number of
    iterations exceed max_iterations.

    Arguments:
    - id_input_file_path: Path to the input file of the first initial data run.
    - id_run_dir: Directory of the first initial data run. If not provided, the
      directory of the input file is used.
    - residual_tolerance: Residual tolerance used for termination condition.
      (Default: 1.e-6)
    - max_iterations: Maximum of iterations allowed. Note: each iteration is
      very expensive as it needs to solve an entire initial data problem.
      (Default: 30)
    """
    iteration = 0
    EC = 1
    subdir0 = os.path.join(bbh_id_dir0, "001_InitialData")
    h5InputFile0 = os.path.join(subdir0, filename)

    # First run: generate_id
    # '--submit' needed to submit without being prompted
    # '--silent' silent mode
    # initial_command = [
    #'spectre', '--silent', 'bbh', 'generate-id', '-q', '1', '--chi-A', '0',
    # '0', '0', '--chi-B', '0', '0', '0', '-e', '0',
    #'--num-orbits', '20', '-L', str(L), '-P', str(P1), '-d', bbh_id_dir0,
    # '--submit'
    # ]
    # run_subprocess(initial_command)
    # wait_for_file(h5InputFile0)

    # Run: evolve
    # evolve_command = [
    #     'spectre', '--silent', 'bbh', 'start-inspiral', '-L', str(L), '-P',
    # str(P2), '-O', './segments',
    #     bbh_id_dir0 + 'InitialData.yaml', '-p', 'reservation=sxs_standing',
    # '-N', '6'
    # #]
    # run_subprocess(evolve_command)
    # wait_for_file(h5InputFile)

    tmin = 30
    tmax = 1000
    angular_velocity_from_xcts = 0.014
    expansion_from_xcts = -1

    while (
        iteration < max_iterations and np.max(np.abs(EC)) > residual_tolerance
    ):
        iteration += 1
        logger.info(f"Starting iteration {iteration}")

        eccout = coordinate_separation_eccentricity_control(
            h5InputFile,
            AhARecord,
            AhBRecord,
            tmin,
            tmax,
            angular_velocity_from_xcts,
            expansion_from_xcts,
        )
        EC = eccout["eccentricity"]
        w = eccout["updated xcts values"]["omega"]
        a = eccout["updated xcts values"]["expansion"]
        angular_velocity_from_xcts = w
        expansion_from_xcts = a

        logger.info(
            f"Iteration {iteration} results - Eccentricity: {EC}, "
            f"Updated omega: {w}, Updated expansion: {a}"
        )

        bbh_id_dir = f"{bbh_id_dir0}_iter_{str(iteration).zfill(3)}"

        subdir = os.path.join(bbh_id_dir, "002_Inspiral", "Segment_0000")
        h5InputFile = os.path.join(subdir, filename)

        # Generate_id + start new inspiral
    #     generate_id_command = [
    #         'spectre', '--silent', 'bbh', 'generate-id', '-q', '1', '-w',
    # str(w), '-a', str(a),
    #         '--chi-A', '0', '0', '0', '--chi-B', '0', '0', '0', '-e', '0',
    # '--num-orbits', '20',
    #         '-L', str(L), '-P', str(P1), '-d', bbh_id_dir, '--submit'
    #     ]
    #     run_subprocess(generate_id_command)
    #     wait_for_file(h5InputFile)

    #     start_inspiral_command = [
    #         'spectre', '--silent', 'bbh', 'start-inspiral', '-L', str(L),
    # '-P', str(P2), '-O', './segments',
    #         bbh_id_dir + 'InitialData.yaml', '-p',
    # 'reservation=sxs_standing', '-N', '6', '--submit'
    #     ]
    #     run_subprocess(start_inspiral_command)
    #     wait_for_file(h5InputFile)

    #     subdir = os.path.join(bbh_id_dir, "002_Inspiral", "Segment_0000")
    #     h5InputFile = os.path.join(subdir, filename)
    #     logger.info(f'Completed iteration {iteration}')

    # logger.info('Auto-eccentricity control finished')

    # Schedule!
    return schedule(
        ringdown_input_file_template,
        **autoecc_control,
        **scheduler_kwargs,
        pipeline_dire=pipeline_dir,
        run_dir=run_dir,
        segments_dir=segments_dir,
    )


@click.command(name="autoecc-control", help=autoecc_control.__doc__)
@click.argument(
    "inspiral_input_file_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@scheduler_options
def autoecc_control_command(**kwargs):
    _rich_traceback_guard = True
    autoecc_control(**kwargs)


if __name__ == "__main__":
    autoecc_control_command(help_option_names=["-h", "--help"])
