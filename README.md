Pytorch Implementation of Adversarial Deep Network Embedding for Cross-Network Node Classification (ACDNE)
====

This is a pytorch implementation of the Adversarial Deep Network Embedding for Cross-Network Node Classification (ACDNE) model presented by Shen et. al (2020, https://arxiv.org/abs/2002.07366)

The official repository for ACDNE (Tensorflow) is available in https://github.com/shenxiaocam/ACDNE. Therefore, if you make advantage of the ACDNE model in your research, please cite the following:

Xiao Shen, Quanyu Dai, Fu-lai Chung, Wei Lu, and Kup-Sze Choi. Adversarial Deep Network Embedding for Cross-Network Node Classification. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), pages 2991-2999, 2020.


Environment Requirement
===
The code has been tested running under Python 3.8.8. The required packages are as follows:

•	python == 3.8.8

•	torch == 1.7.1

•	numpy == 1.19.5

•	scipy == 1.6.2

•	sklearn == 0.24.1


Datasets
===
input/ contains the 5 datasets used in our paper.

Each ".mat" file stores a network dataset, where

the variable "network" represents an adjacency matrix, 

the variable "attrb" represents a node attribute matrix,

the variable "group" represents a node label matrix. 

Code
===
"ACDNE_model.py" is the implementation of the ACDNE model.

"ACDNE_test_Blog.py" is an example case of the cross-network node classification task from Blog1 to Blog2 networks.

"ACDNE_test_citation.py" is an example case of the cross-network node classification task from citationv1 to dblpv7 networks.

