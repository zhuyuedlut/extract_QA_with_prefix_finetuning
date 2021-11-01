#### Exact QA with prefix finetuning
This is a project about use prefix finetuning about exact QA question

--model_type
bart
--train_batch_size
32
--model_dir
"./result"
--eval_batch_size
32
--model_name_or_path
"./resource/bart-base-chinese"
--train_file
"./data/cmrc2018_train.json"
--val_file
"./data/cmrc2018_dev.json"
--test_file
"./data/cmrc2018_test.json"
--do_train
--do_test