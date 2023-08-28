export DATA_SPLIT='iter_latest'
export CONFIG='local_configs/setr/SETR_SAM_PUP.py'
export CHECKPOINT='/opt/data/work_dirs_train/setrsam/iter_64000.pth'
export TEST_SEQ_LEN=5
export CTTA_TYPE='setrsam_cotta_vida_lr5e-81e-5_overall'
# CUDA_VISIBLE_DEVICES=4  python  tools/cotta.py  $CONFIG  $CHECKPOINT --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 0& #--vida_r1 1 --vida_r2 128 --vida_lr 1e-7 --model_lr 1e-6& 
# CUDA_VISIBLE_DEVICES=5  python  tools/cotta.py  $CONFIG  $CHECKPOINT --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 1& #--vida_r1 1 --vida_r2 128 --vida_lr 1e-7 --model_lr 1e-6&
# CUDA_VISIBLE_DEVICES=2  python  tools/cotta.py  $CONFIG  $CHECKPOINT --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 2& #--vida_r1 1 --vida_r2 128 --vida_lr 1e-7 --model_lr 1e-6&
# CUDA_VISIBLE_DEVICES=3  python  tools/cotta.py  $CONFIG  $CHECKPOINT --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 3& #--vida_r1 1 --vida_r2 128 --vida_lr 1e-7 --model_lr 1e-6&
CUDA_VISIBLE_DEVICES=2  python  tools/cotta.py  $CONFIG  $CHECKPOINT --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 0 --vida_r1 1 --vida_r2 128 --vida_lr 5e-8 --model_lr 1e-5& 
CUDA_VISIBLE_DEVICES=3  python  tools/cotta.py  $CONFIG  $CHECKPOINT --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 1 --vida_r1 1 --vida_r2 128 --vida_lr 5e-8 --model_lr 1e-5&
# CUDA_VISIBLE_DEVICES=2  python  tools/cotta.py  $CONFIG  $CHECKPOINT --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index --vida_r1 1 --vida_r2 128 --vida_lr 1e-7 --model_lr 1e-5&
# CUDA_VISIBLE_DEVICES=3  python  tools/cotta.py  $CONFIG  $CHECKPOINT --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index --vida_r1 1 --vida_r2 128 --vida_lr 1e-7 --model_lr 1e-5&

wait
python get_answer.py $CONFIG  $CHECKPOINT --data_split_type $DATA_SPLIT --ctta_type $CTTA_TYPE --e

# export DATA_SPLIT='iter_latest'
# export CONFIG='local_configs/setr/SETR_SAM_PUP.py'
# export CHECKPOINT='/opt/data/work_dirs_train/setrsam/latest.pth'
# export TEST_SEQ_LEN=31
# export CTTA_TYPE='setrsam_cotta_lr5e-5'
# # CUDA_VISIBLE_DEVICES=2  python  tools/cotta_test.py  $CONFIG  $CHECKPOINT --show --show-dir "/opt/data/setrsam_cotta" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 0&
# # CUDA_VISIBLE_DEVICES=3  python  tools/cotta_test.py  $CONFIG  $CHECKPOINT --show --show-dir "/opt/data/setrsam_cotta" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 1&
# CUDA_VISIBLE_DEVICES=4  python  tools/cotta_test.py  $CONFIG  $CHECKPOINT --show --show-dir "/opt/data/setrsam_cotta_lr5e-5" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 0&
# CUDA_VISIBLE_DEVICES=5  python  tools/cotta_test.py  $CONFIG  $CHECKPOINT --show --show-dir "/opt/data/setrsam_cotta_lr5e-5" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 1&
# CUDA_VISIBLE_DEVICES=6  python  tools/cotta_test.py  $CONFIG  $CHECKPOINT --show --show-dir "/opt/data/setrsam_cotta_lr5e-5" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 2&
# CUDA_VISIBLE_DEVICES=7  python  tools/cotta_test.py  $CONFIG  $CHECKPOINT --show --show-dir "/opt/data/setrsam_cotta_lr5e-5" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 3&