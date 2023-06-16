# Notes for repo
Should be able to try and reuse as much as possible from framework. Problems lie with the implicit assumption of TensorFlow (and indeed, an old version requiring Python3.6)

# DeepHOL 
## DeepHOL root files 

- tf logging/gfile only: action_generator, embedding_store, io_util, main?, proof_search_tree, prover, prover_runner, prover_util, prune_lib
- flags: prover_flags (only flags)
- refactor required: holparam_predictor, 


## Utils module
- TensorFlow dependencies in Util module seem only to be for logging. Try replace with standard logging to get working without TF.
- Need old version of GRPC to be compatible. Just install old version and should be fine
- Don't worry about tests in util for now since they're tf tests. 

## train module/folder

- Will need to rewrite in torch, contains code for models, dataset generation and training

## DeepHOL Loop

- Need to reimplement readers, writers and checkpointing for Apache Beam...


## Misc

- gRPC done for Apache Beam? 


## gRPC install/process

- pip install grpc-tools
- from root (deepmath-light) dir:
  - python -m grpc_tools.protoc -I=. --python_out=. ./deepmath/deephol/deephol.proto
  - python -m grpc_tools.protoc -I=. --python_out=. ./deepmath/proof_assistant/proof_assistant.proto --grpc_python_out=.

## Run HOL light server

- Go to HOL light repo then run:

  - sudo docker build . (returns container_id e.g. 8a85414b942e)
  - sudo docker run -d -p 2000:2000 --name=holist container_id