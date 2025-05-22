# JAX-AMR: A JAX-based adaptive mesh refinement framework

JAX-AMR is an adaptive mesh refinement framework based on dynamically updated multi-layer blocks with fixed positions and fixed shapes. This framework is fully compatible with JIT and vectorized operations.

Authors:
- [Haocheng Wen](https://github.com/thuwen)
- [Faxuan Luo](https://github.com/luofx23)

Correspondence via [mail](mailto:haochengwenson@126.com) (Haocheng Wen).

## Implementation Strategy
The multi-layer blocks and the partitioning and refinement strategies in JAX-AMR are illustrated as follows.

<img src="/docs/images/blocks in JAX-AMR.png" alt="Schematic diagram of multi-layer blocks in JAX-AMR" height="500"/>

For the detailed implementation strategies of JAX-AMR, please refer to our [paper](https://doi.org/10.48550/arXiv.2504.13750).

Note: The last block of each layer in JAX-AMR is marked as a NAN block, with all values in its data and info set to NAN. This is to mark the values of ghost grids without neighbor block as invalid values NAN. Therefore, it is important to note that if the solver involves computing global extrema, averages, or similar operations, the influence of NAN values needs to be removed. For instance, the function jnp.nanmax should be used instead of jnp.max.

## Quick Installation
JAX-AMR modules can be easily installed using pip install git:
```
pip install git+https://github.com/JA4S/JAX-AMR.git
```

## Example
An example for the conjunction of a simple CFD solver with JAX-AMR are provided [here](https://github.com/JA4S/JAX-AMR/tree/main/examples).

Open [jax_amr_basic_example.ipynb](https://github.com/JA4S/JAX-AMR/blob/main/examples/jax_amr_basic_example.ipynb) in Google Colab to run the example.

The density result and refinement level for the example are shown as follows.

<img src="/examples/result.png" alt="result" height="400"/>

<img src="/examples/refinement_level.png" alt="refinement level" height="400"/>

## State of the Project

- [x] 2D AMR, fully jit-compiled ✅
- [x] conjuction with the CFD solver ✅
- [ ] 3D AMR (soon)
- [ ] Immerse boudary method (soon)
- [ ] parallel mannagment (soon)

## Citation
JANC: A cost-effective, differentiable compressible reacting flow solver featured with JAX-based adaptive mesh refinement
```
@article{Wen2025,
   author = {Haocheng Wen and Faxuan Luo and Sheng Xu and Bing Wang},
   doi = {10.48550/arXiv.2504.13750},
   journal = {arXiv preprint},
   title = {JANC: A cost-effective, differentiable compressible reacting flow solver featured with JAX-based adaptive mesh refinement},
   year = {2025}
}
```


## License
This project is licensed under the MIT License - see 
the [LICENSE](LICENSE) file or for details https://en.wikipedia.org/wiki/MIT_License.
