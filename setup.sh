#!/usr/bin/env bash
python3 -m virtualenv venv
source ./venv/bin/activate
pip install requirements.txt
git submodule init
git submodule update
