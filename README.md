# GZSL_OS-GZSL

This repository contains the implementation of our approach proposed for Generalized Zero-Shot Learning (GZSL), which combines VAE-GAN-based generative modeling with clustering-based feature selection.

Architecture Diagrams

Ontology schema (img/ontology-schema.png)
Semantic model (img/semantic-model.png)
Architecture overview (img/architecture-overview.png)


📊 Experimental Results
#Zero-Shot Learning (ZSL)
#Generalized ZSL (GZSL)
#Open-Set GZSL (OS-GZSL)

| Dataset | ZSL      | GZSL     | OS-GZSL (70-30) | OS-GZSL (50-50) | 
|---------|----------|----------|-----------------|-----------------|
| AWA2    | 74.0     | 70.1     | 66.8            | 64.3            |
| CUB     | 81.8     | 77.0     | 55.9            | 63.7            |
| SUN     | 66.2     | 42.8     | -               | -               |
| FLO     | 91.8     | 92.4     | 84.5            | 79.0            |


📚 Publications

1. Akdemir, E., Barisci, N., Akcayol, M.A. et al. Selecting generated synthetic features using clustering algorithm for generalized zero-shot learning. Multimedia Systems 31, 402 (2025).
   🔗 https://doi.org/10.1007/s00530-025-01979-z 
2. Akdemir, E., Barisci, N. Generative-based hybrid model with semantic representations for generalized zero-shot learning. SIViP 19, 27 (2025).
   🔗 https://doi.org/10.1007/s11760-024-03734-9
3. E. Akdemir and N. Barışçı, “Ontoloji-Based Generalized Zero-Shot Learning with Generative Networks”, GJES, vol. 10, no. 1, pp. 183–192, 2024.
   🔗 https://doi.org/10.30855/gmbd.0705n15

Referenced Repositories & Acknowledgements
This work builds upon and extends several valuable open-source contributions. We would like to express our sincere thanks to the authors of the following repositories, which we used and/or adapted in the development of our code and experiments:

🔗 https://github.com/akshitac8/tfvaegan – for the base VAE-GAN framework

🔗 https://github.com/uqzhichen/SDGZSL – for semantic description

🔗 https://github.com/genggengcss/OntoZSL – for ontology

🔗 https://github.com/facebookresearch/mixup-cifar10 – for mixup function

In addition, we used the benchmark datasets (AWA2, CUB, FLO) and associated semantic embeddings provided by the above repositories.
