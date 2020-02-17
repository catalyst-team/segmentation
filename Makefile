.PHONY: check-style codestyle docker-build clean

check-style:
	bash ./bin/_check_codestyle.sh -s

codestyle:
	pre-commit run

docker: ./requirements/
	docker build -t catalyst-segmentation:latest . -f ./docker/Dockerfile

clean:
	rm -rf build/
	docker rmi -f catalyst-segmentation:latest
