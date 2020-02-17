echo "pip install -r requirements/requirements.txt"
pip install -r requirements/requirements.txt

echo "bash ./bin/tests/_check_instance.sh"
bash ./bin/tests/_check_instance.sh

rm -rf ./data ./logs
