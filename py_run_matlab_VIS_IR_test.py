import os
import sys
import time
from pathlib import Path
import argparse
import subprocess
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "info": "dim green",
    "warning": "magenta",
    "danger": "bold red"
})
console = Console(theme=custom_theme)

def arg_parser():
    args = argparse.ArgumentParser()
    
    args.add_argument('--fused_dir', '-f', type=str, help='Path to the fused image directory')
    args.add_argument('--dataset_name', '-d', type=str, choices=['llvip', 'msrs', 'm3fd', 'tno', 'roadscene', 
                                                                 'med_harvard_spect_mri', 'med_harvard_pet_mri',
                                                                 'med_harvard_ct_mri'])
    args.add_argument('--method_name', '-m', type=str)
    args.add_argument('--vi_dir', type=str)
    args.add_argument('--ir_dir', type=str)
    args.add_argument('--fused_ext', '-e', type=str)
    args.add_argument('--no_rgb_test', action='store_false', default=True)
    args.add_argument('--no_test_easy_mode', action='store_false', default=True)
    args.add_argument('--no_save_excel', '-s', action='store_false', default=True)
    
    # launch method
    args.add_argument('--use_mp', '-mp', action='store_true', default=False)
    
    return args.parse_args()


def run_matlab_command(args):
    try:
        console.log('Current working directory:', os.getcwd(), style='info')
        os.chdir('VIS_IR_Matlab_Test_Package')
        console.log('Changed working directory to:', os.getcwd(), style='info')
    except Exception as e:
        console.log(f'Error changing directory: {e}', style='danger')
        return
    
    rgb_test = int(args.no_rgb_test)
    test_easy_mode = int(args.no_test_easy_mode)
    
    # check
    if not Path(args.fused_dir).exists():
        console.log('Fused image directory does not exist!', style='danger')
        raise FileNotFoundError
    _exist_img = False
    _check_ext = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    for ext in _check_ext:
        try:
            next(Path(args.fused_dir).glob(f'*.{ext}'))
            _exist_img = True
            break
        except StopIteration:
            if ext == _check_ext[-1]:
                raise FileNotFoundError('No image files found in the fused image directory')
            continue
    
    # matlab command
    if not args.use_mp: 
        start_method = 'runDir'
    else:
        start_method = 'mp_runDir'
    matlab_command = f'{start_method}("{args.fused_dir}", "{args.dataset_name.upper()}", "{args.method_name}", {rgb_test}, {test_easy_mode}'
    matlab_varargin = ","
    if args.vi_dir:
        matlab_varargin += f'"vi_dir", "{args.vi_dir}",'
    if args.ir_dir:
        matlab_varargin += f'"ir_dir", "{args.ir_dir}",'
    if args.fused_ext:
        matlab_varargin += f'"fused_ext", "{args.fused_ext}",'
        
    run_time = time.strftime('%m-%d %H:%M:%S')
    file_name = f'logs/{run_time}_{args.dataset_name}_{args.method_name}.log'
    log_file_dir = Path(file_name).parents[0]
    if not log_file_dir.exists():
        console.log(f'creating log file at: {file_name}', style='info')
        log_file_dir.mkdir(parents=True, exist_ok=True)
        
    matlab_varargin += f'"file_name", "{file_name}",'
    
    matlab_command += matlab_varargin
    matlab_command = matlab_command.rstrip(',')
    matlab_command += ")"
    
    console.log('Running Matlab command', style='info')
    console.log(f'Matlab command: [dim cyan]{matlab_command}[dim cyan]')
    
    console.log(f'log file is saved at: [green underline]{file_name}[/green underline]')
    
    # Run the MATLAB command with nohup and subprocess
    try:
        with open(os.devnull, 'w') as null_output:
            process = subprocess.Popen(
                ['nohup', 'matlab', '-nodisplay', '-nosplash', '-r', matlab_command],
                stdout=null_output,
                stderr=null_output,
                preexec_fn=os.setpgrp  # Ensures the process is not killed with the terminal
            )
            console.log(f'Matlab process pid is [underline]{process.pid}[/underline]', style='info')
            console.log('MATLAB process started successfully.', style='info')
            try:
                p = input('[Y] to kill the process; [N] for running in background:')
                if p in ['Y', 'yes', 'y', 'YES', 'Yes']:
                    process.kill()
                    console.log('MATLAB process terminated successfully.', style='info')
                elif p in ['N', 'no', 'n', 'NO', 'No']:
                    console.log('MATLAB process is running in the background.', style='info')
            except KeyboardInterrupt:
                console.log('MATLAB process interrupted by user termination', style='warning')
                process.kill()
            
    except Exception as e:
        console.log(f'Error running MATLAB command: {e}', style='danger')
    
    
if __name__ == '__main__':
    args = arg_parser()
    
    run_matlab_command(args)
    