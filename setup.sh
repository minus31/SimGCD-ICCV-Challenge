#!/bin/bash
pip install -U pip

pip install -r requirements.txt
pip install loguru
cd SSB
pip install -e .