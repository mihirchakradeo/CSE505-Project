CSE 505 Project

This is the course project for the course CSE 505, Fall 2017.

DeepMath - Deep Sequence Models for Premise Selection

Original Code From: https://github.com/girving/deepmath/tree/master/holstep_baselines

My Contributions:

1. Unconditional 1D CNN-RNN
2. Conditional 1D CNN-RNN
3. Unconditional 1D CNN-GRU
4. Conditional 1D CNN-GRU
5. Unconditional 1D CNN-Encoder Decoder
6. Conditional 1D CNN-Encoder Decoder

Execution Steps:

Download the holstep data set from here: http://cl-informatik.uibk.ac.at/cek/holstep/

python main.py \
--model_name=<model name> \
--task_name=<conditioned_classification/unconditioned_classification> \
--logdir=experiments/<folder for tensorboard> \
--source_dir=path to holstep dataset
