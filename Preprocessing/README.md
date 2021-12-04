# CS-640-Project PREPROCESSING
Project for CS 640 -  Intro to AI - Fall 21 - Boston University

##For Preprocessing Scripts:

### Create a virtual environment
`python3 -m venv <env_name>`

### Source the virtual environment
#### note that windows environments use the other backslash 
#### to separate directories

`source <env_name>/bin/activate`

### Install libraries to the virtual environment
`pip install -r requirements.txt`

### Select the virtual environment as the system's python interpreter 
###### via IDE preferences

### Create two data folders for the indexes 
###### using shell or terminal from the preprocessing directory
`mkdir ../letterdb`

`mkdir ../worddb`

### Run indexer.py to create inverted indexes of letters and words
#### The \ character escapes the space in the directory name. 
###### You can also use quotes around the file path/name.
`python3 ./Preprocessing/indexer.py ./Data\ Twitter/Train/joy-ratings-0to1.train.txt `
`python3 ./Preprocessing/indexer.py ./Data\ Twitter/Train/sadness-ratings-0to1.train.txt `
`python3 ./Preprocessing/indexer.py ./Data\ Twitter/Train/fear-ratings-0to1.train.txt `
`python3 ./Preprocessing/indexer.py ./Data\ Twitter/Train/anger-ratings-0to1.train.txt `