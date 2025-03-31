build_ft_unsloth:
	docker build -t eft-unsloth:latest -f ft_unsloth/Dockerfile .

launch_ft_unsloth:
	docker run -it --rm --name ft_unsloth \
	-v ${PWD}:/workspace \
	-v ${PWD}/.cache:/root/.cache \
	--gpus all \
	eft-unsloth:latest bash

build_generate_data:
	docker build -t eft-gen:latest -f generate_data/Dockerfile .

launch_generate_data:
	docker run -it --rm --name eft-gen \
	-v ${PWD}:/workspace \
	-v ${PWD}/.cache:/root/.cache \
	--gpus all \
	eft-gen:latest bash