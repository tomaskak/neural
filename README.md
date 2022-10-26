# neural
python package for neural net training modules

## util
This namespace contains strctures and logic that is used in training but no AI-specific
such as replay buffers.

## tools
This namespace contains tools for training neural nets such as:
* Parallelization
* GPU/CPU Support

## model
This namespace contains shared code for building reusable models.

## algos
This namespace contains implementations of specific algorithms making use of the code available in model.

## Dependency Chain
A namespace can only depend upon those listed to the left of it:

util <-- tools <-- model <-- algos

## Docker

### To build and run
```
docker build -t --build-arg BASE=(python:bullseye|nvidia/cuda:11.7.1-base-ubuntu22.04) neural .
docker run --rm -it -v=$(pwd)/run_artifacts:/vol/run_artifacts --runtime nvidia --ipc=host neural (-m pytest | sac_run.py --iterations 100000)
```
* BASE default is the nvidia image but can only be used on machines where CUDA is available, all others use python:bullseye
* -v mount is optional but is how to extract the saved models and results from the run
* --runtime nvidia is for CUDA machines
* --ipc=host is for the shared memory used by pytorch in multiprocessing