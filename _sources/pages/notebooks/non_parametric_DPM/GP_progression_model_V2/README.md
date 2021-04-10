# GP_progression_model_V2

# python-hello_world

This repository contains an example of a minimal Python package.
The `setup.py` file is central, it uses setuptools to package the python code.
In the following we consider that the Conda environment described in the `conda/environment.yaml` file has been created and activated.
To do so, type the following command line

 ```console

  conda env create -f conda/environment.yaml
  conda activate gppm

 ```

To create a Python package that can be used in this environment, three options are possible:

1. A develop mode.
   Using this mode, all local modifications of source code will be considered in your Python interpreter (when restarted) without any post-installation.
   This is particularly useful when adding new features.
   To install this package in develop mode, type the following command line

   ```console

   python setup.py develop --prefix=${CONDA_PREFIX}

   ```

   Once this step is done, type the following command line for running tests

   ```console

   nosetests test -x -s -v

   ```

   Note that this require to have installed `nose` in your environment.

2. An install mode.
   Using this mode, all local modifications of source code will **NOT** be considered in your Python interpreter (when restarted).
   To consider your local modifications of source code, you must install the package once again.
   This is particularly useful when you want to install a stable version in one Conda environment while creating another environment for using the develop mode.
   To install this package in develop mode, type the following command line

   ```console

   python setup.py install --prefix=${CONDA_PREFIX}

   ```

3. A release mode.
   Using the mode, a Conda package will be created from the install mode and can be distributed with Conda.
   To build this package with Conda (with `conda-build` installed), type the following command line

   ```console

   conda build conda/recipe -c pytorch -c defaults --override-channels

   ```

   Then, you can upload the generated package or just install it.
   To install this conda package, type the following command line
   
   ```console

   conda install gppm -c local -c pytorch -c defaults --override-channels

   ```


  With the previous modes, the Conda environment doesn't know that this python package has been installed.
  But with this method, the `gppm` will appear in your Conda package listing.
