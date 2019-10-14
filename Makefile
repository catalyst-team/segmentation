check-style:
	bash ./bin/_check_codestyle.sh -s

codestyle:
	bash ./bin/_check_codestyle.sh

docker-build: ./requirements.txt
	docker build -t catalyst-segmentation:latest . -f docker/Dockerfile

docker-clean:
	rm -rf build/
	docker rmi -f catalyst-segmentation:latest
