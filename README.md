# mrcnf

This repository contains code for [Multi-Resolution Continuous Normalizing Flows](https://arxiv.org/abs/2106.08462).

### Examples
The paper applies multi-resolution regularized neural ODEs to density estimation and generative modeling using the FFJORD-RNODE framework. Example training scripts are found in `example-scripts/`

### Citation

Please cite as

```
@article{voleti2021mrcnf,
  author    = {Vikram Voleti and
               Chris Finlay and
               Adam Oberman and
               Christopher Pal},
  title     = {Multi-Resolution Continuous Normalizing Flows},
  journal   = {CoRR},
  volume    = {abs/2106.08462},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.08462},
  archivePrefix = {arXiv},
  eprint    = {2106.08462},
}
```

### Many thanks

FFJORD-RNODE was gratefully forked from https://github.com/cfinlay/ffjord-rnode, FFJORD was gratefully forked from https://github.com/rtqichen/ffjord.
