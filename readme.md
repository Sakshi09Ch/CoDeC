# CoDeC: Communication-Efficient Decentralized Continual Learning
This repository contains the source code associated with CoDeC: Communication-Efficient Decentralized Continual Learning.
## Introduction
Training at the edge utilizes continuously evolving data generated at different locations. Privacy concerns prohibit the co-location of this spatially as well as temporally distributed data, deeming it crucial to design training algorithms that enable efficient continual learning over decentralized private data. Decentralized learning allows serverless training with spatially distributed data. A fundamental barrier in such setups is the high bandwidth cost of communicating model updates between agents. Moreover, existing works under this training paradigm are not inherently suitable for learning a temporal sequence of tasks while retaining the previously acquired knowledge. In this work, we propose CoDeC, a novel communication-efficient decentralized continual learning algorithm that addresses these challenges. We mitigate catastrophic forgetting while learning a distributed task sequence by incorporating orthogonal gradient projection within a gossip-based decentralized learning algorithm. Further, CoDeC includes a novel lossless communication compression scheme based on the gradient subspaces. We theoretically analyze the convergence rate for our algorithm and demonstrate through an extensive set of experiments that CoDeC successfully learns distributed continual tasks with minimal forgetting. The proposed compression scheme results in up to 4.8× reduction in communication costs without any loss in performance.


## Requirements
- To create the conda environment for running the experiments --> conda env create -f env_codec.yml

   Activate the environment --> conda activate codec

- The dataset for Split CIFAR-100 and 5-Datasets experiments will be automatically downloaded. Download the data for miniImageNet from the following links (taken from https://github.com/LYang-666/TRGP) and store these files in the folder named 'data_minii': 

   Training data: https://drive.google.com/file/d/1fm6TcKIwELbuoEOOdvxq72TtUlZlvGIm/view

  Testing data: https://drive.google.com/file/d/1RA-MluRWM4fqxG9HQbQBBVVjDddYPCri/view

## Usage
- Please refer to run_exp.sh.

## Authors
[Sakshi Choudhary](https://github.com/Sakshi09Ch), [Sai Aparna Aketi](https://github.com/aparna-aketi), [Gobinda Saha](https://github.com/sahagobinda), Kaushik Roy

All authors are with Purdue University, West Lafayette, IN, USA.

