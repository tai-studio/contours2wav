# contours2wav

## Description

A python script that extracts contours from an image and generates a wav file for each contour.

## Installation

```zsh
git clone git@github.com:tai-studio/contours2wav.git
cd contours2wav
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Usage

Before usage, make sure to activate the virtual environment:

```zsh
source venv/bin/activate
```

alternatively, you can use the `venv/bin/python` command:

```zsh
venv/bin/python contour2wav.py -m100 -n50000 -t 110 infile.jpg outDir
```

### single input file
```zsh
python contour2wav.py -m100 -n50000 -t 110 infile.jpg outDir
```

### multiple input files

```zsh
for file in ../_assets/*.jpg; do python contour2wav.py -m100 -n50000 -t 110 $file ../_renders; done
```