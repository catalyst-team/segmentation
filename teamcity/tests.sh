echo "pip install -r requirements/requirements.txt"
pip install -r requirements/requirements.txt

echo "pip install -r requirements/requirements-dev.txt"
pip install -r requirements/requirements-dev.txt

echo "isort -rc --check-only --settings-path ./setup.cfg"
isort -rc --check-only --settings-path ./setup.cfg

echo "make check-style"
make check-style