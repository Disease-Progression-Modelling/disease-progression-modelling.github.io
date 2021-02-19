# Collaboration

TODO : Write the steps to add additional notebooks

# Installation



### Install the write conda environment

- conda env create -f environment.yml
- conda activate DPM_website

### Build the website once you have added the notebooks
- jupyter-book build ./ #Creates all the static files
- ghp-import -n -p -f _build/html #Pushes the static files to the gh-pages branch of the repo, that is the one dedicated to the website

# Files & Folders :

### Project related
- README.md

### Jupyter book related
- _toc.yml
- _config.yml
- _build/ :


### Python environnment related
- environment.yml
