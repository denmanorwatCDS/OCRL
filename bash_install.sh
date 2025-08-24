#!/bin/bash
git clone git@github.com:richardrl/fetch-block-construction.git
python3.9 -m pip install setuptools==66.0.0
python3.9 -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python3.9 -m pip install timm==0.9.16
python3.9 -m pip install -e ./fetch-block-construction
python3.9 -m pip install -r pip_requirements.txt
