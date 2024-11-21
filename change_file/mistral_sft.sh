#!/bin/bash

# 设置环境变量
export PYTHONPATH="/opt/NeMo:$PYTHONPATH"
export TMPDIR=/maindata/data/shared/public/yangchao.zhou/opt/tmp
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 运行 Python 脚本
python /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_sft.py \
    # 每隔多少个 step 打印日志
    trainer.log_every_n_steps=1 \
    # 使用 bf16 精度
    trainer.precision=bf16 \
    # 节点数
    trainer.num_nodes=1 \
    # 使用 8 张 GPU
    trainer.devices=8 \
    # 最大训练 step 数，-1 表示不限
    trainer.sft.max_steps=-1 \
    # 最大训练 epoch 数
    trainer.sft.max_epochs=4 \
    # 验证集使用的 batch 比例
    trainer.sft.limit_val_batches=1.0 \
    # 多少 step 进行一次验证
    trainer.sft.val_check_interval=1 \
    # Tensor Model 并行的大小
    model.tensor_model_parallel_size=1 \
    # Pipeline Model 并行的大小
    model.pipeline_model_parallel_size=8 \
    # 启用 AMP O2 优化
    model.megatron_amp_O2=True \
    # 恢复模型路径
    model.restore_from_path=/maindata/data/shared/public/rufeng.dai/models/nemo/Mistral-Nemo-Instruct-2407-toconvernemo.nemo \
    # 使用的优化器
    model.optim.name=distributed_fused_adam \
    # 学习率
    model.optim.lr=1e-6 \
    # 学习率调度器名称
    model.optim.sched.name=CosineAnnealing \
    # 学习率 warmup 阶段步数
    model.optim.sched.warmup_steps=10 \
    # 常量学习率阶段步数
    model.optim.sched.constant_steps=0 \
    # 最小学习率
    model.optim.sched.min_lr=1e-8 \
    # 是否启用 chat 模式数据处理
    model.data.chat=True \
    # 数据加载器的工作线程数
    model.data.num_workers=0 \
    # 训练集的微批大小
    model.data.train_ds.micro_batch_size=1 \
    # 训练集的全局批大小
    model.data.train_ds.global_batch_size=4 \
    # 训练集的最大序列长度
    model.data.train_ds.max_seq_length=8192 \
    # 训练集文件路径
    model.data.train_ds.file_path=data/nemo_data/cleaned_labeled_data-20241119_dedup-V2/all_data_sft_train.jsonl \
    # 验证集的微批大小
    model.data.validation_ds.micro_batch_size=1 \
    # 验证集的全局批大小
    model.data.validation_ds.global_batch_size=4 \
    # 验证集文件路径
    model.data.validation_ds.file_path=data/nemo_data/cleaned_labeled_data-20241119_dedup-V2/all_data_sft_val.jsonl \
    # 验证集的最大序列长度
    model.data.validation_ds.max_seq_length=8192 \
    # 是否启用 wandb 日志记录
    exp_manager.create_wandb_logger=True \
    # 日志文件保存路径
    exp_manager.explicit_log_dir=/maindata/data/shared/public/yangchao.zhou/opt/NeMo-Aligner/results \
    # wandb 项目名称
    exp_manager.wandb_logger_kwargs.project=mistral-12b-20241119_dedup-V2 \
    # wandb 日志记录的实验名称
    exp_manager.wandb_logger_kwargs.name=mistral-12b-20241119_dedup-V2-sft \
    # 是否在训练结束时保存 Nemo 文件
    exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
    # 保存的 checkpoint 数量
    exp_manager.checkpoint_callback_params.save_top_k=20 \
    # 是否始终保存 Nemo 文件
    exp_manager.checkpoint_callback_params.always_save_nemo=True \
    # 如果已有 checkpoint，是否恢复训练
    exp_manager.resume_if_exists=True \
    # 没有 checkpoint 时是否忽略
    exp_manager.resume_ignore_no_checkpoint=True \
    # 是否创建 checkpoint 回调
    exp_manager.create_checkpoint_callback=True \
    # checkpoint 的评估指标
    exp_manager.checkpoint_callback_params.monitor=val_loss
