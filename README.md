# Disease Progression Modelling Website

This repo intends to host the disease progression modelling website, hosted at disease-progression-modelling.github.io.


## How to collaborate?

Everyone is welcome to collaborate to the website and to be part of the community.
To that end, you can add yours materials and notebooks to the repo.
These will be reviewed and merged to the current website

Note : You can find a Notebook Template in `pages/notebooks/_template.ipynb` to comply to the website template


## Website structure

The website relies on [Jupyter Book](https://jupyterbook.org/).
To add you Markdown & Jupyter files, add them within the `pages` folder, and link them from the `_toc.yml` file so that they appear in the website content.
Overall, you have :
- `pages/DPM` : files regarding the Disease Progression Models, their history, state of the art models, ...
- `pages/materials` : files containing materials regarding the models, links to papers, etc
- `pages/notebooks` : tutorials related to the DPM models
- `pages/conferences` : different workshop and conference where the DPM tutorials have been used
- `pages/other` : some additional files need for the website


#### Other files
- `README.md` : read me of the github repo
- `environment.yml` : requirements of the
- `_toc.yml`, `_config.yml` : Jupyter book configuration files. Note that `_toc.yml` contains the structure of the website



## Installation

### Clone the repo
```
git clone https://github.com/Disease-Progression-Modelling/disease-progression-modelling.github.io
cd Disease-Progression-Modelling.github.io
```

### Install the conda environment
```
conda env create -f environment.yml
conda activate DPM_website
```

### Add you materials

You are now welcome to add you Markdown and Notebook pages directly to the repo.
The website is build thanks to Jupyter Book, a Python package that you can discover [here](https://jupyterbook.org/).
In a nutshell, you can add Python notebooks and markdown anywhere in the `pages` directory (and subdirectories). Once added, you have to edit the `_toc.yml` file to organize the website table of content accordingly.


Note : Please create your own branch to add your modifications.

### Build the website once you have added the notebooks

To check you materials and how they integrate within the website, run :  
```
jupyter-book build ./
```
You can they see the results by opening the `_build/html/index.html` file in your browser.
You should be able to navigate and see the website with your modifications.

### Add to the website online

Once you are satisfied with your changes, you can open a merge request so that administrators review your code and add it to the website

Note for administrators :
Github tracks the `_build` folder of the `gh-pages` branch. To push the static files to this branch, we use the `ghp-import` package.
To use it, you just have to run the following code :
```
ghp-import -n -p -f _build/html
```
