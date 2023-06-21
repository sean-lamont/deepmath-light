# Notes for repo
Should be able to try and reuse as much as possible from framework. Problems lie with the implicit assumption of TensorFlow (and indeed, an old version requiring Python3.6)

- Abseil (absl) library used instead of TF flags 
- unittest library used in place of TF tests, although absl parametrised tests are preserved
- Standard python logging library used in place of TF logging

# DeepHOL 


## train module/folder

- Will need to rewrite in torch, contains code for models, dataset generation and training

## DeepHOL Loop

- Need to reimplement readers, writers and checkpointing for Apache Beam...


## Misc

- gRPC as the main message protocol, is it done for Apache Beam? 
-  


## gRPC install/process

- pip install grpc-tools
- from root (deepmath-light) dir:
  - python -m grpc_tools.protoc -I=. --python_out=. ./deepmath/deephol/deephol.proto
  - python -m grpc_tools.protoc -I=. --python_out=. ./deepmath/proof_assistant/proof_assistant.proto --grpc_python_out=.

## Run HOL light server

- Go to HOL light repo then run:

  - sudo docker build . (returns container_id e.g. 8a85414b942e)
  - sudo docker run -d -p 2000:2000 --name=holist container_id

## 

- prover_runner runs prover based on prover_options file
- prvoer_options gives theorem_database, tactics, type of proof search, proof search options, action_generator options, 