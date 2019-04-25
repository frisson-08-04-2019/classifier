#!/bin/bash

sudo apt update
sudo apt-get install python3-pip python3-dev -y
conda create -n classifier pip python=3.7.1 -y
source activate classifier
easy_install -U pip
pip install --ignore-installed --upgrade -r requirements.txt
conda install pytorch=1.0.1 torchvision cudatoolkit=9.0 cudnn=7.1.2 -c pytorch -y
