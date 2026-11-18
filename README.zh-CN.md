# Ceviche Challenges：光子器件逆向设计挑战

[English](README.md) | **简体中文**

`ceviche_challenges` 提供一组光子器件逆向设计任务，使用开源 [Ceviche](https://github.com/fancompute/ceviche/) 的频域有限差分法（FDFD）求解器。本分支使用 Ceviche 的 PyTorch 分支计算散射参数、电磁场及其梯度。

这些任务可用于比较不同拓扑优化算法在集成光子器件设计中的表现，涵盖弯波导、分束器、模式转换器和波分复用器（WDM）。

本分支移植的设计任务源自 Ceviche Challenges 的原始实现。该实现用于论文 [Inverse Design of Photonic Devices with Strict Foundry Fabrication Constraints](https://doi.org/10.1021/acsphotonics.2c00313) 的实验。

## 设计任务

### 弯波导

![弯波导](img/waveguide_bend.png)

### 分束器

![分束器](img/beam_splitter.png)

### 模式转换器

![模式转换器](img/mode_converter.png)

### 波分复用器（WDM）

![波分复用器](img/wavelength_divison_multiplexer.png)

## 安装

`pytorch` 分支从源码安装，使用 Python 3.12 和固定在 `external/ceviche` 的 Ceviche PyTorch 子模块。配置好对子模块仓库的 SSH 访问后，在项目根目录执行以下命令。uv 会以可编辑模式安装子模块。

```sh
git submodule update --init --recursive
uv sync --locked
```

以下命令用于在项目根目录进行本地开发检查。

```sh
uv run --locked ruff check ceviche_challenges
uv run --locked ruff format --check ceviche_challenges
uv run --locked ty check
uv run --locked pytest
```

## 使用

将下面两段 Python 示例按顺序保存到项目根目录的 `example.py`，执行 `uv run --locked python example.py`。也可以在装有 Python 扩展的 VS Code 中打开项目文件夹，在命令面板选择 **Python: Select Interpreter**，选中 Windows 下的 `.venv/Scripts/python.exe` 或 Linux 下的 `.venv/bin/python`。打开脚本后，点击 **Run Python File** 即可运行。

以下模块包含 `ceviche_challenges` 的器件预设。

- `ceviche_challenges.beam_splitter.prefabs`
- `ceviche_challenges.mode_converter.prefabs`
- `ceviche_challenges.waveguide_bend.prefabs`
- `ceviche_challenges.wdm.prefabs`

每种器件都有对应的模型类，仿真会返回复数散射参数和电场。下面使用 2 µm × 2 µm 弯波导预设。

```python
import numpy as np
import ceviche_challenges

spec = ceviche_challenges.waveguide_bend.prefabs.waveguide_bend_2umx2um_spec()
params = ceviche_challenges.waveguide_bend.prefabs.waveguide_bend_sim_params()
model = ceviche_challenges.waveguide_bend.model.WaveguideBendModel(params, spec)

# 按模型要求的形状初始化设计变量。
design = np.ones(model.design_variable_shape)
s_params, fields = model.simulate(design)
```

`s_params` 的轴顺序是波长、激励端口、输出端口。`fields` 的轴顺序是波长、激励端口、x、y。`simulate()` 默认激励编号为 0 的端口，并使用仿真参数里的预设波长。可通过 `excite_port_idxs` 和 `wavelengths_nm` 指定端口和波长。

传入 NumPy 数组时，前向仿真返回 NumPy 数组。传入 Torch 张量时，结果保留在输入设备上，可以对实值目标函数求梯度。使用 `float64` 可进行双精度计算。要在可用的 CUDA 设备上运行，创建张量时指定 `device="cuda"`。需要调整迭代求解器时，向 `simulate()` 传入 `solver_options=ceviche.SolverOptions(...)`。

```python
import torch

design = torch.ones(model.design_variable_shape, dtype=torch.float64,
                    requires_grad=True)
s_params, _ = model.simulate(design)
loss = s_params[:, 0, 0].abs().mean() - s_params[:, 0, 1].abs().mean()
loss.backward()
loss_value = loss.item()
loss_grad = design.grad
```

用 `model.density(design)` 可以查看器件的材料分布。更多接口可查阅模型源码和函数文档字符串。

## 引用

如在研究中使用 `ceviche_challenges`，请引用 [原论文](https://doi.org/10.1021/acsphotonics.2c00313)。

```bibtex
@article{schubert_inverse_2022,
  title = {Inverse Design of Photonic Devices with Strict Foundry Fabrication Constraints},
  author = {Schubert, Martin F. and Cheung, Alfred K. C. and Williamson, Ian A. D. and Spyra, Aleksandra and Alexander, David H.},
  journal = {ACS Photonics},
  volume = {9},
  number = {7},
  pages = {2327-2336},
  year = {2022},
  doi = {10.1021/acsphotonics.2c00313},
}
```
