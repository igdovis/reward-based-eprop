# Reward-based-e-prop


Experiments were done in google colab and on a local laptop.

Laptop specifications: 16GB RAM, Nvidia GeForce RTX 3060 Laptop GPU, AMD RYZEN 7 5800H CPU

`seq_length` variable of RSNN computation steps highly influences training time.

Two types of networks are present: a network solely from LIF neurons and solely from ALIF.

The models are different, but the actor-critic algorithm to optimize the policy is the same for both of them.

Code inspired by the supervised learning implementation of e-prop https://github.com/ChFrenkel/eprop-PyTorch




