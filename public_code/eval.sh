
for seed in `seq 1 5`; do
	export TRAIN_PATH="output_ls/dev/HellaSWAG_roberta-base_lambda_1.0_${seed}.json"
	export TEST_PATH="output_ls/test/HellaSWAG_roberta-base_lambda_1.0_${seed}.json"
	CUDA_VISIBLE_DEVICES=2 python calibrate.py \
		--train_path $TRAIN_PATH \
		--test_path $TEST_PATH \
		--do_train \
		--do_evaluate
done;