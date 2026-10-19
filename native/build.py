"""Build the C ABI libraries with C++17; CUDA is opt-in and never emulated.

CPU: python3 native/build.py
CUDA: python3 native/build.py --cuda-root /path/to/cuda --arch 120
No implicit fast math or contraction. No compiler/toolkit installation here.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def build(cuda_root=None, arch='120'):
    destination = ROOT/'build'/'native'
    destination.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which(os.environ.get('CXX', 'g++'))
    if compiler is None:
        raise RuntimeError('C++ compiler missing; set CXX to a C++17 compiler')
    commands = [[compiler, '-std=c++17', '-O3', '-fPIC', '-shared',
                 '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra',
                 str(ROOT/'native'/'cpu.cpp'), '-o', str(destination/'liblambert_cpu.so')]]
    if cuda_root:
        toolkit = Path(cuda_root).expanduser().resolve()
        nvcc = toolkit/'bin'/'nvcc'
        if not nvcc.is_file() or not arch.isdigit():
            raise ValueError('Expected a CUDA root containing bin/nvcc and a numeric architecture')
        # Static cudart avoids depending on the task-local toolkit's loader path.
        commands.append([str(nvcc), '-std=c++17', '-O3', '--shared', '--cudart=static',
                         '-Xcompiler=-fPIC,-ffp-contract=off,-fno-fast-math,-Wall,-Wextra',
                         '--fmad=false', '--prec-div=true', '--prec-sqrt=true', '--ftz=false',
                         f'-gencode=arch=compute_{arch},code=sm_{arch}',
                         f'-gencode=arch=compute_{arch},code=compute_{arch}',
                         '--ptxas-options=-v,-warn-spills', '-lineinfo',
                         '-ccbin', compiler, f'-L{toolkit/"lib"}',
                         str(ROOT/'native'/'cuda.cu'), str(ROOT/'native'/'cpu.cpp'),
                         '-o', str(destination/'liblambert_cuda.so')])
    metadata = dict(commands=commands, cpu_compiler=subprocess.check_output([compiler, '--version'], text=True))
    if cuda_root:
        metadata['cuda_compiler'] = subprocess.check_output([str(nvcc), '--version'], text=True)
    build_started = time.perf_counter()
    build_seconds = []
    for command in commands:
        print('Building:', ' '.join(command), flush=True)
        # Do not truncate a shared object that a live Python process mapped.
        target = Path(command[-1])
        temporary = target.with_suffix('.building.so')
        build_command = command[:-1]+[str(temporary)]
        try:
            started = time.perf_counter()
            subprocess.run(build_command, check=True)
            build_seconds.append(time.perf_counter()-started)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    metadata['compile_seconds'] = build_seconds
    metadata['build_wall_seconds'] = time.perf_counter()-build_started
    (destination/'build.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Libraries:', destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-root', default=None)
    parser.add_argument('--arch', default='120', help='CUDA compute capability digits, e.g. 120')
    args = parser.parse_args()
    build(args.cuda_root, args.arch)
