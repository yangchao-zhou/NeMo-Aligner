#!/bin/bash

# 获取当前日期并格式化为 YYYY-MM-DD-HH-MM-SS 形式
current_date=$(date +'%Y-%m-%d-%H-%M-%S')

# 设置环境变量
export PYTHONPATH="/opt/NeMo:$PYTHONPATH"
export TMPDIR="/maindata/data/shared/public/yangchao.zhou/opt/tmp"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY="5cecedb71fc563c002ee042baa9ba8be32424434"

# 运行 Python 脚本 注意要用绝对路径
python /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_sft.py \
    trainer.log_every_n_steps=1 \
    trainer.precision=bf16 \
    trainer.num_nodes=1 \
    trainer.devices=4 \
    trainer.sft.max_steps=-1 \
    trainer.sft.max_epochs=4 \
    trainer.sft.limit_val_batches=1.0 \
    trainer.sft.val_check_interval=200 \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=4 \
    model.megatron_amp_O2=True \
    model.restore_from_path=/maindata/data/shared/public/rufeng.dai/models/nemo/Mistral-Nemo-Instruct-2407-toconvernemo.nemo \
    model.optim.name=distributed_fused_adam \
    model.optim.lr=3e-6 \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=100 \
    model.optim.sched.constant_steps=0 \
    model.optim.sched.min_lr=1e-7 \
    model.data.chat=True \
    model.data.num_workers=0 \
    model.data.train_ds.micro_batch_size=1 \
    model.data.train_ds.global_batch_size=1 \
    model.data.train_ds.max_seq_length=4096 \
    model.data.train_ds.file_path=/maindata/data/shared/public/yangchao.zhou/opt/spanish/data/nemo_data/cleaned_labeled_data-20241119_dedup-V2/all_data_sft_train.jsonl \
    model.data.validation_ds.micro_batch_size=1 \
    model.data.validation_ds.global_batch_size=1 \
    model.data.validation_ds.file_path=/maindata/data/shared/public/yangchao.zhou/opt/spanish/data/nemo_data/cleaned_labeled_data-20241119_dedup-V2/all_data_sft_train.jsonl \
    model.data.validation_ds.max_seq_length=4096 \
    exp_manager.create_wandb_logger=True \
    exp_manager.explicit_log_dir=/maindata/data/shared/public/yangchao.zhou/opt/NeMo-Aligner/results/mistral-12b-20241119_dedup-V2-${current_date}/ \
    exp_manager.wandb_logger_kwargs.project=mistral-12b-20241119_dedup-V2 \
    exp_manager.wandb_logger_kwargs.name=mistral-12b-20241119_dedup-V2-${current_date} \
    exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
    exp_manager.checkpoint_callback_params.save_top_k=5 \
    exp_manager.checkpoint_callback_params.always_save_nemo=True \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_checkpoint_callback=True \
    exp_manager.checkpoint_callback_params.monitor=val_loss