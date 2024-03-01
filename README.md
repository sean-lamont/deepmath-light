# Deepmath Light (Deprecated)

## Moved to parent BAIT project

This repository is intended to be a lightweight version of the original deepmath/deephol 
source. The original project has several dependencies which can make it difficult to set up the 
environment locally. It is also written in TensorFlow 1, which further complicates a smooth integration with modern tools.

As a result, this was written with the goal of maintaining the core functionality of the original
framework, while being as simple as possible to get running. TensorFlow dependencies are kept to a minimum, with python builtins
used where possible (in particular, file I/O, logging and tests are now written with standard library functions such as the unittest and logging libraries). 
This was done to reduce the dependence on a particular ML framework.  

The learning algorithms are implemented in PyTorch instead of TensorFlow, however they are made as modular as possible to facilitate alternatives with e.g. TF 2.0, JAX etc. 
