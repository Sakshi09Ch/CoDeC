# CoDeC
Communication-Efficient Decentralized Continual Learning

Training at the edge utilizes continuously evolving data generated at different locations. Privacy concerns prohibit the co-location of this spatially as well as 
temporally distributed data, deeming it crucial to design training algorithms that enable efficient continual learning over decentralized private data. Decentralized 
learning allows serverless training with spatially distributed data. A fundamental barrier in such distributed learning is the high bandwidth cost of communicating model 
updates between agents. Moreover, existing works under this training paradigm are not inherently suitable for learning a temporal sequence of tasks while retaining the 
previously acquired knowledge. In this work, we propose CoDeC, a novel communication-efficient decentralized continual learning algorithm which addresses these challenges.
We mitigate catastrophic forgetting while learning a task sequence in a decentralized learning setup by combining orthogonal gradient projection with gossip averaging 
across decentralized agents. Further, CoDeC includes a novel lossless communication compression scheme based on the gradient subspaces. We express layer-wise gradients 
as a linear combination of the basis vectors of these gradient subspaces and communicate the associated coefficients. We theoretically analyze the convergence rate for 
our algorithm and demonstrate through an extensive set of experiments that CoDeC successfully learns distributed continual tasks with minimal forgetting. The proposed 
compression scheme results in up to 4.8× reduction in communication costs with iso-performance as the full communication baseline.
