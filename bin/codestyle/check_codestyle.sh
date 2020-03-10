#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

# Parse -s flag which tells us that we should skip inplace yapf
echo 'parse -s flag'
skip_inplace=""
while getopts ":s" flag; do
  case "${flag}" in
    s) skip_inplace="true" ;;
  esac
done

echo 'isort -rc --check-only --settings-path ./setup.cfg'
isort -rc --check-only --settings-path ./setup.cfg

# stop the build if there are any unexpected flake8 issues
echo 'bash ./bin/codestyle/flake8.sh'
bash ./bin/codestyle/flake8.sh --count \
  --config=./setup.cfg \
  --show-source \
  --statistics

# exit-zero treats all errors as warnings.
echo 'flake8'
flake8 ./bin/codestyle/flake8.sh --count \
  --config=./setup.cfg \
  --max-complexity=10 \
  --show-source \
  --statistics \
  --exit-zero

# test to make sure the code is yapf compliant
if [[ -f ${skip_inplace} ]]; then
  echo 'bash ./bin/codestyle/yapf.sh --all'
  bash ./bin/codestyle/yapf.sh --all
else
  echo 'bash ./bin/codestyle/yapf.sh --all-in-place'
  bash ./bin/codestyle/yapf.sh --all-in-place
fi
