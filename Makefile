pipeline:
	get-datasets
	train-tokenizer
	train-model


get-datasets:
	python -m basemodel.data.dataset

train-tokenizer:
	almond-train-tokenizer

train-model:
	almond-pretrain