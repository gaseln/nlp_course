#!/bin/bash

set -euo pipefail

# export CONDA_PREF=$PWD/env_nlp_course

# source ~/HOME_SCRATCH_FOLDER/mambaforge/etc/profile.d/conda.sh
# source ~/HOME_SCRATCH_FOLDER/mambaforge/etc/profile.d/mamba.sh
# mamba create -p $CONDA_PREF -y
# mamba activate $CONDA_PREF

mamba install pip nltk gensim bokeh jupyter scikit-learn matplotlib seaborn -c conda-forge -y
pip install torch=cu124 subword-nmt transformers datasets accelerate
mamba install cuda=12.4 -c nvidia -y
pip install deepspeed protobuf tiktoken sentencepiece optimum auto-gptq bitsandbytes

