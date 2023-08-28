# export DATA_SPLIT='iter_latest'
# export CONFIG='local_configs/shift_test_800x500.py'
# export CHECKPOINT='/data/SHIFT/iter_40000.pth'
# export TEST_SEQ_LEN=31
# export CTTA_TYPE='TEST'
# CUDA_VISIBLE_DEVICES=0  python  tools/test_test.py  $CONFIG  $CHECKPOINT --show-dir "/opt/data/deeplabv3_guanfang_test_pil" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 0&
# CUDA_VISIBLE_DEVICES=1  python  tools/test_test.py  $CONFIG  $CHECKPOINT --show-dir "/opt/data/deeplabv3_guanfang_test_pil" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 1&
# CUDA_VISIBLE_DEVICES=2  python  tools/test_test.py  $CONFIG  $CHECKPOINT --show-dir "/opt/data/deeplabv3_guanfang_test_pil" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 2&
# CUDA_VISIBLE_DEVICES=3  python  tools/test_test.py  $CONFIG  $CHECKPOINT --show-dir "/opt/data/deeplabv3_guanfang_test_pil" --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 3&

export DATA_SPLIT='iter_latest'
export CONFIG='local_configs/shift_train_800x500.py'
export CHECKPOINT='/data/SHIFT/iter_40000.pth'
export TEST_SEQ_LEN=15
export CTTA_TYPE='deeplabv3_guanfang_overall'
CUDA_VISIBLE_DEVICES=0  python  tools/test.py  $CONFIG  $CHECKPOINT  --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 0& 
CUDA_VISIBLE_DEVICES=1  python  tools/test.py  $CONFIG  $CHECKPOINT  --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 1&
CUDA_VISIBLE_DEVICES=2  python  tools/test.py  $CONFIG  $CHECKPOINT  --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 2&
CUDA_VISIBLE_DEVICES=3  python  tools/test.py  $CONFIG  $CHECKPOINT  --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 3&

wait
python get_answer.py $CONFIG  $CHECKPOINT --data_split_type $DATA_SPLIT --ctta_type $CTTA_TYPE #--e