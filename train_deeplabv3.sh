CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29700 tools/train.py local_configs/shift_train_800x500.py --work-dir /opt/data/work_dirs_train/deeplabv3_shift_continous_train --gpus 2 --launcher pytorch