#!/usr/bin/env python 
import os
import sys

if len(sys.argv) != 2:
    print("usage: push.py \"<commit msg>\"")

os.system("git add *")
os.system(f'git commit -m \"{sys.argv[1]}\"')
os.system("git push")
