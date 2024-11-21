export CUDA_DEVICE_MAX_CONNECTIONS=1
python /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_sft.py \
   trainer.log_every_n_steps=1 \
   trainer.precision=bf16 \
   trainer.num_nodes=1 \
   trainer.devices=8 \
   trainer.sft.max_steps=-1 \
   trainer.sft.max_epochs=4 \
   trainer.sft.limit_val_batches=1.0 \
   trainer.sft.val_check_interval= \  # 训练多少step评估一次
   model.tensor_model_parallel_size=1 \
   model.pipeline_model_parallel_size=8 \
   model.megatron_amp_O2=True \
   model.restore_from_path=/mnt/data/nlp_models/mistralai/nemo/Mistral-Nemo-Instruct-2407.nemo \  # mistral12b模型一定要用这个，不要用官方的
   model.optim.name=distributed_fused_adam \
   model.optim.lr=1e-6 \
   model.optim.sched.name=CosineAnnealing \
   model.optim.sched.warmup_steps=10 \
   model.optim.sched.constant_steps=0 \
   model.optim.sched.min_lr=1e-8 \
   model.data.chat=True \
   model.data.num_workers=0 \
   model.data.train_ds.micro_batch_size=1 \
   model.data.train_ds.global_batch_size=4 \
   model.data.train_ds.max_seq_length=8192 \
   model.data.train_ds.file_path= \  # jsonl
   model.data.validation_ds.micro_batch_size=1 \
   model.data.validation_ds.global_batch_size=4 \
   model.data.validation_ds.file_path= \  # jsonl
   model.data.validation_ds.max_seq_length=8192 \
   exp_manager.create_wandb_logger=True \
   exp_manager.explicit_log_dir= \  # results目录
   exp_manager.wandb_logger_kwargs.project= \  # exp_manager
   exp_manager.wandb_logger_kwargs.name= \
   exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
   exp_manager.checkpoint_callback_params.save_top_k=20 \
   exp_manager.checkpoint_callback_params.always_save_nemo=True \  # 一定要这样设置
   exp_manager.resume_if_exists=True \
   exp_manager.resume_ignore_no_checkpoint=True \
   exp_manager.create_checkpoint_callback=True \
   exp_manager.checkpoint_callback_params.monitor=val_loss