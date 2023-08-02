# Reward-based-e-prop


Experiments were done mostly in google colab and on a local laptop.

Laptop specifications: 16GB RAM, Nvidia GeForce RTX 3060 Laptop GPU, AMD RYZEN 7 5800H CPU

## Explanation of code

Spiking neural networks (SNN) are the third generation of neural networks, characterized by their usage of spikes to propagate information, as opposed to floating point values that are used by artificial neural networks (ANN). SNNs work best when implemented in neuromorphic hardware. These two phenomena aim to eliminate the Von Neumann bottleneck that is present in current hardware. Moreover, SNNs are more energy efficient than ANNs and scale better.

With this code, I wanted to implement the e-prop algorithm by:
 > [G. Bellec et al., "A solution to the learning dilemma for recurrent networks of spiking neurons," *Nature communications*, vol. 11, no. 3625, 2020]

E-prop is an online learning algorithm and can be used for reinforcement learning problems. With e-prop, control problems, such as the Cart Pole task can be solved in an online setting. 

In this code, two types of neuron models are present: leaky integrate & fire (LIF) and adapative leaky integrate & fire (ALIF). ALIF uses adaptive thresholds, which are crucial to solve the temporal credit assignment task and are the main focus of the e-prop paper.

The reinforcement learning part is done with the advantage actor-critic algorithm.  

Code inspired by the supervised learning implementation of e-prop https://github.com/ChFrenkel/eprop-PyTorch

The spiking neural network library _Norse_ was used to grab their encoding function.

### Useful information and papers:
- https://www.nature.com/articles/s41467-020-17236-y
- https://huggingface.co/learn/deep-rl-course/unit6/advantage-actor-critic?fw=pt
- https://arxiv.org/abs/2109.12894 
- https://github.com/norse/norse


