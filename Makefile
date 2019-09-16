.PHONY: classification

check-style:
	bash ./bin/_check_codestyle.sh -s

codestyle:
	bash ./bin/_check_codestyle.sh

segmentation: requirements.txt
	docker build -t catalyst-segmentation:latest . -f docker/Dockerfile

clean:
	rm -rf build/
	docker rmi -f catalyst-segmentation:latest
