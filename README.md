# Ceviche Challenges: Photonic Inverse Design Suite

The `ceviche_challenges` module contains a suite of photonic inverse design
challenge problems, backed by the open source
[`ceviche`](https://github.com/fancompute/ceviche/) finite difference frequency
domain (FDFD) simulator. This branch uses the PyTorch fork of Ceviche to
compute scattering parameters, fields, and gradients.

The suite of challenge problems may be used to benchmark different topology 
optimization algorithms in designing relevant photonic components. The suite 
includes several integrated photonic components, including a waveguide beam 
splitter, a waveguide mode converter, a waveguide bend, and a wavelength
division multiplexer (WDM).

The code in this module was used to produce the results in the
[Inverse Design of Photonic Devices with Strict Foundry Fabrication Constraints](https://doi.org/10.1021/acsphotonics.2c00313)
paper.

## Challenge problems

### Waveguide bend

![Waveguide bend](/img/waveguide_bend.png)

### Beam splitter

![Beam splitter](/img/beam_splitter.png)

### Mode converter

![Mode converter](/img/mode_converter.png)

### Wavelength division multiplexer (WDM)

![Wavelength division multiplexer (WDM)](/img/wavelength_divison_multiplexer.png)

## Installation

`pytorch` is installed from source. It uses Python 3.12 and the PyTorch fork of
`ceviche` pinned in `external/ceviche`. With SSH access to the submodule
server, run these commands from the repository root. uv installs the
submodule in editable mode.

```sh
git submodule update --init --recursive
uv sync --locked
uv run --locked pytest
```

## Usage

 `ceviche_challenges` provides several "prefab" component configurations, which
 live in the following submodules:

*   `ceviche_challenges.beam_splitter.prefabs`
*   `ceviche_challenges.mode_converter.prefabs`
*   `ceviche_challenges.waveguide_bend.prefabs`
*   `ceviche_challenges.wdm.prefabs`

Each component type (i.e. beam splitter, mode converter, etc) has an associated
model class for performing simulations and returning complex-valued scattering
parameters. An example using the 2 um x 2 um waveguide bend prefab is shown
below.

```python
import numpy as np
import ceviche_challenges

spec = ceviche_challenges.waveguide_bend.prefabs.waveguide_bend_2umx2um_spec()
params = ceviche_challenges.waveguide_bend.prefabs.waveguide_bend_sim_params()
model = ceviche_challenges.waveguide_bend.model.WaveguideBendModel(params, spec)

# The model class provides a convenience property, `design_variable_shape`
# which specifies the design shape it expects.
design = np.ones(model.design_variable_shape)

# The model class has a `simulate()` method which takes the design variable as
# an input and returns scattering parameters and fields.
s_params, fields = model.simulate(design)
```

Pass a Torch tensor to get Torch outputs on the same device and differentiate a
real-valued objective. Use a `float64` tensor for double precision. To run on an
available CUDA device, create the tensor with `device="cuda"`.

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

NumPy inputs return NumPy arrays. Existing HIPS autograd objectives can still
use first-order reverse-mode gradients on CPU.

There are a number of other features in the device model class, such as
returning an image of the structure. These features are best explored by reading
the code and the function docstrings, as we don't have comprehensive standalone
documentation at this time.

## Citation

If you find `ceviche_challenges` to be useful for your research, please consider
citing our [paper](https://doi.org/10.1021/acsphotonics.2c00313). A BibTex
citation is provided below for convenience.

```
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
