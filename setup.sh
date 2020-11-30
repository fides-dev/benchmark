#!/usr/bin/env bash
python3 -m virtualenv venv --clear
source ./venv/bin/activate
pip install -r requirements.txt
git submodule init
git submodule update
