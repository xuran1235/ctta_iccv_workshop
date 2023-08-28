export DATA_SPLIT='iter_latest'
export CONFIG='local_configs/mask2former/mask2former_swin-l_shiftdataset.py'
export CHECKPOINT='work_dirs_train/mask2former_shift/iter_100000.pth'
export TEST_SEQ_LEN=15
export CTTA_TYPE='Test'
CUDA_VISIBLE_DEVICES=4  python  tools/test.py  $CONFIG  $CHECKPOINT --show --show-dir ./VIS_TEST/ --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 0& 
CUDA_VISIBLE_DEVICES=5  python  tools/test.py  $CONFIG  $CHECKPOINT --show --show-dir ./VIS_TEST/ --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 1&
CUDA_VISIBLE_DEVICES=6  python  tools/test.py  $CONFIG  $CHECKPOINT --show --show-dir ./VIS_TEST/ --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 2&
CUDA_VISIBLE_DEVICES=7  python  tools/test.py  $CONFIG  $CHECKPOINT --show --show-dir ./VIS_TEST/ --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 3& 

wait
python get_answer.py $CONFIG  $CHECKPOINT --data_split_type $DATA_SPLIT --ctta_type $CTTA_TYPE --e --show-dir ./TEST/
