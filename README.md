# Set Solver

Using OpenCV to identify cards in an image and solve for valid sets in a game of [Set](https://www.wikiwand.com/en/Set_(game)). Currently only solves for static image

## Installation

Virtualenv required. Will also need OpenCV installation on your machine and symlinks to necessary files.

```
# create virtual environment
virtualenv .venv
source .venv/bin/activate

# install requirements
pip install -r requirements.txt

# install OpenCV and create symlinks
brew tap homebrew/science
brew install opencv
cd .venv/lib/python2.7
ln -s /usr/local/Cellar/opencv/2.4.10/lib/python2.7/site-packages/cv.py cv.py
ln -s /usr/local/Cellar/opencv/2.4.10/lib/python2.7/site-packages/cv2.so cv.so

```

## Run tests

For examples on how to use code, check out `tests.py`. Run it to see the card detection and solving in action:

```
source .venv/bin/activate
python
>>> import tests 
>>> tests.main()
```