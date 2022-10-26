This repo contains a work-in-progress set of features for training NNs in RL.
It is alot of custom implementations for the purpose of learning.

The primary algorithm implemented is Soft Actor-Critic (SAC) and uses multprocessing to split work as follows.
```
Base task - creates subtasks, then generates training data and periodically tests the model. (generation and test will be split out in the future)
ReplayStore - takes new observations and writes them to a replay buffer in shared memory.
ReplaySample - creates mini-batches of data by sampling the replay buffer and pushes them to training task.
Train - Pulls mini batches from the queue as available and runs SAC update repeatedly (updates shmem actor model used in test/explore after each update)


Base --> [state,action,reward,etc] --> ReplayStore.    ______SHMEM_______
  |                                       |           |                  |
  | (uses actor)                          | (push)    |    ReplayBuffer  |       ReplaySample --> [ mini-batch ] --> Trainer (updates actor)           
  |_______________________________________|___________|       Actor      |___________|
                                                      |__________________| (reads from replay)
```

# Package layout
## neural
python package for neural net training modules

### util
This namespace contains strctures and logic that is used in training but no AI-specific
such as replay buffers.

### tools
This namespace contains tools for training neural nets such as:
* Parallelization
* GPU/CPU Support

### model
This namespace contains shared code for building reusable models.

### algos
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
