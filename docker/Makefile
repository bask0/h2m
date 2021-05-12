IMAGE_NAME=bkraft_pytorch1.8
APP_NAME=bkraft_pytorch1.8

build:

	docker build -t $(IMAGE_NAME) .

rebuild:

	docker build --no-cache -t $(IMAGE_NAME) .

run: build

	docker run  --gpus all -it --shm-size=500g --rm --ipc=host \
	-v /scratch/bkraft:/scratch \
	-v /Net/Groups/BGI/people/bkraft/hydrodl:/workspace/hydrodl \
	-v /Net/Groups/BGI/people/bkraft:/workspace/bkraft \
	-v /Net/Groups/BGI:/workspace/BGI \
	-v /Net/Groups/BGI/work_2/hydro_dl/:/workspace/work_2 \
	-v /scratch/bkraft:/workspace/scratch \
	-p 8888:8888 \
	-p 8443:8443 \
	-p 8442:8442 \
	-p 8441:8441 \
	-p 8265:8265 \
	--name $(APP_NAME) \
	$(IMAGE_NAME)

test: run

	docker port $(APP_NAME) $(HOST_PORT)

stop:

	docker stop $(APP_NAME)

clean: stop

	docker rmi $(IMAGE_NAME)
