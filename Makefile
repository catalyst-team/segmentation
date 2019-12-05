.PHONY: check-style codestyle docker-build clean

check-style:
	bash ./bin/_check_codestyle.sh -s

codestyle:
	pre-commit run

docker-build: ./requirements/requirements-docker.txt
	docker build -t catalyst-segmentation:latest . -f docker/Dockerfile

clean:
	rm -rf build/
	docker rmi -f catalyst-segmentation:latest
