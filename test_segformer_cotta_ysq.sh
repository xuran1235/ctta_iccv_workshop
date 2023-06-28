export DATA_SPLIT='iter_latest'
export CONFIG='local_configs/segformer/B5/segformer.b5.1280x800.shift.160k.py'
export CHECKPOINT='/home/ChallengeB/SHIFT-Continuous_Test_Time_Adaptation/work_dirs_train/segformer_shift/latest.pth'
export TEST_SEQ_LEN=15
export CTTA_TYPE='CoTTA'
CUDA_VISIBLE_DEVICES=4  python  tools/cotta.py  $CONFIG  $CHECKPOINT --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 0&
CUDA_VISIBLE_DEVICES=5  python  tools/cotta.py  $CONFIG  $CHECKPOINT --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 1&
CUDA_VISIBLE_DEVICES=6  python  tools/cotta.py  $CONFIG  $CHECKPOINT --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 2&
CUDA_VISIBLE_DEVICES=7  python  tools/cotta.py  $CONFIG  $CHECKPOINT --ctta_type $CTTA_TYPE --data_split_type $DATA_SPLIT --test_seq_len $TEST_SEQ_LEN --test_index 3&

wait
python get_answer.py $CONFIG  $CHECKPOINT --data_split_type $DATA_SPLIT --ctta_type $CTTA_TYPE