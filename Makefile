pipeline:
	get-datasets
	train-tokenizer
	train-model

# ----------------------------------
# PRETRAIN
# ----------------------------------

get-datasets:
	python -m basemodel.data.dataset

train-tokenizer:
	almond-train-tokenizer

train-model:
	almond-pretrain

inference-pretrain:
	python -m tests.inference_pretrain

# ----------------------------------
# FINETUNE
# ----------------------------------

get-datasets-ft:
	python -m finetune.data.dataset

train-finetune:
	almond-finetune

inference-finetune:
	python -m tests.inference_finetune

eval-finetune:
	python -m eval.translation_eval