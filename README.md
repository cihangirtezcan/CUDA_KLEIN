# CUDA_KLEIN

These codes are used in the publication "GPU-Based Brute Force Cryptanalysis of KLEIN" to be published in the conference proceedings of ICISSP 2024.

**Abstract**
KLEIN is a family of lightweight block ciphers that supports 64-bit, 80-bit, and 96-bit secret keys. In this work, we provide a CUDA optimized table-based implementation of the KLEIN family which does not contain shared memory bank conflicts. Our best optimization reach more than 45 billion 64-bit KLEIN key searches on an RTX 4090. Our results show that KLEIN block cipher is susceptible to brute force attacks via GPUs. Namely, in order to break KLEIN in a year via brute force, one needs around 13, 1.34 million, and 111 billion RTX 4090 GPUs for 64-bit, 80-bit, and 96-bit secret keys, respectively. We recommend lightweight designs to avoid short keys.
