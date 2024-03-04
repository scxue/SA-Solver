# SA-Solver
Official code for SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models (NeurIPS 2023)
<div align="center">
  <a href="https://arxiv.org/pdf/2309.05019.pdf"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv">
</div>

> [**SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models (Neurips 2023)**](https://arxiv.org/pdf/2309.05019.pdf)<br>
> [Shuchen Xue*](https://github.com/scxue), [Mingyang Yi]()&#8224;, 
> [Weijian Luo](), [Shifeng Zhang](), [Jiacheng Sun](),
> [Zhenguo Li](https://scholar.google.com/citations?user=XboZC1AAAAAJ),
> [Zhi-Ming Ma]()
> <br>University of Chinese Academy of Sciences, Huawei Noah’s Ark Lab, Peking University<br>
---

## News

SA-Solver is integrated into [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha)! Try SA-Solver in [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha).

SA-Solver is now also available in 🧨 Diffusers and accesible via the [SASolverScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_sasolver.py).
Diffusers allows you to test SA-Solver in PyTorch in just a couple lines of code.

## 🐱 Abstract
SA-Solver is a stochastic diffusion sampler based on Stochastic Adams Method. It is training-free and can be employed into pretrained diffusion models. It is a multistep SDE solver that can do fast stochastic sampling. 

1. The parameter 'tau function' controls the stochasticity in the sampling process. Inspired by EDM, we choose the 'tau function' to be a piecewise constant function that is greater than 0 in the middle stage of sampling process and equals zero in the start and end stage. Specifically, we choose the default value of this parameter to be

```python
tau_func = lambda t: 1 if t >= 200 and t <= 800 else 0
```

in diffusers library and 

```python
tau_t = lambda t: eta if 0.2 <= t <= 0.8 else 0
```

in ldm library. (The difference is because the time transformation * 1000).

The value '1' represents the magnitude of stochasticity. Higher value are recommended with more NFEs.

If you want to employ deterministic sampling (solving diffusion ODE) in SA-Solver, please set

```python
tau_func = lambda t: 0
```

If you want to employ original stochastic sampling (solving original diffusion SDE) in SA-Solver, please set

```python
tau_func = lambda t: 1
```


2. The parameter 'predictor_order' and 'corrector_order' controls the specific orders of 'SA-Predictor' and 'SA-Corrector'. For unconditional generation and conditional generation with small classifier-free guidance scale, the recommended orders are 'predictor_order = 3' and 'corrector_order = 4'; for conditional generation with large classifier-free guidance scale (e.g. t2i), the recommended orders are 'predictor_order = 2' and 'corrector_order = 2'.

# Acknowledgement

Our code is based on [DPM-Solver](https://github.com/LuChengTHU/dpm-solver) and [UniPC](https://github.com/wl-zhao/UniPC).

# Citation

If you find our work useful in your research, please consider citing:

```
@article{xue2024sa,
  title={SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models},
  author={Xue, Shuchen and Yi, Mingyang and Luo, Weijian and Zhang, Shifeng and Sun, Jiacheng and Li, Zhenguo and Ma, Zhi-Ming},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```