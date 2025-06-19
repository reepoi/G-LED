Generative Learning for Forecasting the Dynamics of High Dimensional Complex Systems
====================================================================================

Unofficial code to attempting to replicate the results of [gaoGenerativeLearningForecasting2024]_.

.. note::

   *This is a fork of the official code.*
   Find the official code `here <https://github.com/cselab/G-LED>`_.

Installation
============

#. Clone this repository, and then clone our fork of Imagen - Pytorch into a directory **next to where the directory where you cloned this repository.**

   .. code:: bash

      # This repository
      git clone git@github.com:reepoi/G-LED.git
      # Our fork of Imagen - Pytorch
      git clone git@github.com:reepoi/imagen-pytorch.git

#. ``cd`` into the directory where Imagen - Pytorch is cloned (e.g., `imagen-pytorch`) and checkout the `G-LED` branch.

   .. code:: bash

      git switch G-LED

#. ``cd`` into the directory where this repository is cloned (e.g., `G-LED`) and checkout the `new-conf` branch.

   .. code:: bash

      git switch new-conf

#. Install ``uv``:

   .. code:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh

#. Install Python dependencies using ``uv``:

   .. code:: bash

      uv sync

#. Activate the Python virtual environment:

   .. code:: bash

      source .venv/bin/activate

#. Test your installation:

   .. code:: bash

      pytest tests

#. Edit the ``out_dir`` and ``run_subdir`` fields of the ``Conf`` class in ``src/conf/conf.py`` to the directory where you want the model training output to be saved.

   .. warning::

      Due to a bug in ``hydra-orm``, the configuration settings mentioned here must be edited in their respective Python files.
      Command line overrides for these settings will be ignored.

#. Edit the ``_data_dir`` field of the ``Dataset`` class in ``src/conf/dataset.py`` to the directory where you want the generated datasets to be saved.

Supplementary documentation
===========================

* `Hydra <https://hydra.cc/docs/1.3/intro/>`_: Command-line inferface configuration library for configuring the experiments in this project.
* `Hydra ORM <https://github.com/reepoi/hydra-orm>`_: Library for saving experiment configurations to an `SQLite <https://sqlite.org/>`_ database.
* `PyTorch <https://pytorch.org/docs/2.7/index.html>`_: Library for implementing the models.
* `PyTorch Lightning <https://lightning.ai/docs/pytorch/2.5.1/>`_: Library for handling model training.

Training the models
===================

Examples for the running the code are in the ``Examples`` subsection.

Run the command

.. code:: bash

   python src/g_led/main_sequential.py dataset=<dataset> model=<model> <other_overrides>...

where: todo

Examples
--------

The following are example commands to show how to run the code.

.. code:: bash

   # Generate Kuramoto-Sivashinsky dataset with 590 trajectories
   python src/g_led/datasets.py dataset=KuramotoSivashinsky1DSize1 model=TransformerKuramotoSivashinsky1D

Running experiments in parallel
===============================

.. warning::

   On network file systems (NFS), starting multiple processes running this code can corrupt the SQLite database storing the experiment configurations.
   See question (5) of the `SQLite FAQs <https://sqlite.org/faq.html>`_.
   See the Preflight section below to see how to ensure the experiment configurations are written to the database serially.

Using `GNU parallel <https://www.gnu.org/software/parallel/>`_, multiple experiments can be run in parallel.

.. code:: bash

   parallel --eta --header : python src/g_led/main_sequential.py <override_1>={<param_1>} <override_2>={<param_2>} ... ::: <param_1> <p1value_1> <p1value_2> ... ::: <param_2> <p2value_1> <p2value_2> ...


Preflight
---------

To ensure that experiment configurations are saved to the database serially, run GNU parallel command with ``-j 1`` and the Python command with ``-c job``.

.. code:: bash

   parallel -j 1 --eta --header : python src/g_led/main_sequential.py -c job <override_1>={<param_1>} <override_2>={<param_2>} ... ::: <param_1> <p1value_1> <p1value_2> ... ::: <param_2> <p2value_1> <p2value_2> ...

Once this command has finished, all the experiment configurations have been saved.
Next, run the first GNU parallel command to begin running the experiments in parallel.

References
==========

.. [gaoGenerativeLearningForecasting2024] `H. Gao, S. Kaltenbach, and P. Koumoutsakos, "Generative learning for forecasting the dynamics of high-dimensional complex systems," Nat Commun, vol. 15, no. 1, p. 8904, Oct. 2024, doi: 10.1038/s41467-024-53165-w. <https://www.nature.com/articles/s41467-024-53165-w>`_
