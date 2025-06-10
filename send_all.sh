#!/bin/bash

find . -type d -name '__pycache__' -exec rm -r {} + -o -type f -name '*.pyc' -exec rm -f {} +

git add .
read -p "Comment: " comment
git commit -m "$comment"
git push
