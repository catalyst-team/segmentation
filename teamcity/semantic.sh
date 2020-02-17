echo "pip install -r requirements/requirements.txt"
pip install -r requirements/requirements.txt

echo "bash ./bin/tests/_check_semantic.sh"
bash ./bin/tests/_check_semantic.sh

rm -rf ./data ./logs
