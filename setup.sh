#!/usr/bin/env bash
python3 -m virtualvenv venv
source ./venv/bin/activate
pip install requirements.txt
git submodule init
git submodule update
