# neural
python package for neural net training modules


## tools
This namespace contains tools for training neural nets such as:
* Visualization
* Parallelization
* GPU/CPU Support

## model
This namespace contains shared code for building reusable models.

## algos
This namespace contains implementations of specific algorithms making use of the code available in model.

## Dependency Chain
A namespace can only depend upon those listed to the left of it:

tools <-- model <-- algos
