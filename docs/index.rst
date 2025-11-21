.. Quantum-inspired Kolmogorov-Arnold Network documentation master file, created by
   sphinx-quickstart on Mon Jun 23 20:09:40 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Quantum-Inspired Kolmogorov-Arnold Network (QKAN)
=================================================

Welcome to the documentation for the **Quantum-Inspired Kolmogorov-Arnold Network (QKAN)**!

This project builds upon the concepts introduced in the paper `Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks`_ and is open-sourced at GitHub repository `Jim137/qkan`_.

.. _Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks: https://arxiv.org/abs/2509.14026
.. _Jim137/qkan: https://github.com/Jim137/qkan

``QKAN`` is a novel neural network architecture that integrates **Quantum Variational Activation Functions (QVAFs)** with the **Kolmogorov-Arnold Network (KAN)** paradigm, designed for expressive and efficient function approximation and machine learning tasks.

Installation
------------

Install via pip:

.. code-block:: bash

   pip install qkan

Or install from source:

.. code-block:: bash

   git clone https://github.com/Jim137/qkan.git
   cd qkan
   pip install -e .

We recommend using a virtual environment to avoid conflicts with other packages:

.. code-block:: bash

   python -m venv qkan-env
   source qkan-env/bin/activate  # On Windows: qkan-env\Scripts\activate
   pip install qkan

Quick Start
-----------

Here's a minimal working example for function fitting using QKAN:

.. code-block:: python

   import torch

   from qkan import QKAN, create_dataset

   device = "cuda" if torch.cuda.is_available() else "cpu"

   f = lambda x: torch.sin(20*x)/x/20 # J_0(20x)
   dataset = create_dataset(f, n_var=1, ranges=[0,1], device=device, train_num=1000, test_num=1000, seed=0)
   
   qkan = QKAN(
      [1, 1], 
      reps=3, 
      device=device, 
      seed=0,
      preact_trainable=True, 
      postact_weight_trainable=True,
      postact_bias_trainable=True, 
      ba_trainable=True,
      save_act=True, # enable to plot from saved activation
   )

   optimizer = torch.optim.LBFGS(qkan.parameters(), lr=5e-2)

   qkan.train_(
      dataset,
      steps=100,
      optimizer=optimizer,
      reg_metric="edge_forward_dr_n",
   )

   qkan.plot(from_acts=True, metric=None)

Citation
--------

If you find this project useful in your research, please consider citing the following biobliographic references:

.. code-block:: bibtex

   @article{jiang2025qkan,
      title={Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks},
      author={Jiang, Jiun-Cheng and Huang, Yu-Chao and Chen, Tianlong and Goan, Hsi-Sheng},
      journal={arXiv preprint arXiv:2509.14026},
      year={2025},
      url={https://arxiv.org/abs/2509.14026}
   }
   @misc{jiang2025qkan_software,
      title={QKAN: Quantum-inspired Kolmogorov-Arnold Network},
      author={Jiang, Jiun-Cheng},
      year={2025},
      publisher={Zenodo},
      doi={10.5281/zenodo.17437425},
      url={https://doi.org/10.5281/zenodo.17437425}
   }

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   api
   examples


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
