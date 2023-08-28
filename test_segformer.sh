# export DATA_SPLIT='iter_latest'
# export CONFIG='local_configs/segformer/B5/segformer.b5.1280x800.shift.160k.py'
# export CHECKPOINT='work_dirs_train/segformer_shift/latest.pth'
# export TEST_SEQ_LEN=5
# export CTTA_TYPE='segformer_tent'
# CUDA_VISIBLE_DEVICES=4  python  tools/tent.py  $CONFIG  $CHECKPOINT  --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 0& 
# CUDA_VISIBLE_DEVICES=5  python  tools/tent.py  $CONFIG  $CHECKPOINT  --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 1&
# # CUDA_VISIBLE_DEVICES=6  python  tools/cotta_test.py  $CONFIG  $CHECKPOINT --show --show-dir "/opt/data/segformer_cotta" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 0&
# # CUDA_VISIBLE_DEVICES=7  python  tools/cotta_test.py  $CONFIG  $CHECKPOINT --show --show-dir "/opt/data/segformer_cotta" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 1&

# wait
# python get_answer.py $CONFIG  $CHECKPOINT --data_split_type $DATA_SPLIT --ctta_type $CTTA_TYPE --e

export DATA_SPLIT='iter_latest'
export CONFIG='local_configs/segformer/B5/segformer.b5.1280x800.shift.160k.py'
export CHECKPOINT='work_dirs_train/segformer_shift/latest.pth'
export TEST_SEQ_LEN=31
export CTTA_TYPE='TEST'
CUDA_VISIBLE_DEVICES=0  python  tools/test_test.py  $CONFIG  $CHECKPOINT --show --show-dir "/opt/data/segformer_test_pil" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 0&
CUDA_VISIBLE_DEVICES=1  python  tools/test_test.py  $CONFIG  $CHECKPOINT --show --show-dir "/opt/data/segformer_test_pil" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 1&
CUDA_VISIBLE_DEVICES=2  python  tools/test_test.py  $CONFIG  $CHECKPOINT --show --show-dir "/opt/data/segformer_test_pil" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 2&
CUDA_VISIBLE_DEVICES=3  python  tools/test_test.py  $CONFIG  $CHECKPOINT --show --show-dir "/opt/data/segformer_test_pil" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 3&
