"""Baseten training job: all LoRA/QLoRA variants under purged CV and the day holdout, then final checkpoints (H100)."""
from truss.base.truss_config import AcceleratorSpec
from truss_train import CacheConfig, CheckpointingConfig, Compute, Image, Runtime, TrainingJob, TrainingProject

job = TrainingJob(
    image=Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"),
    compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=1)),
    runtime=Runtime(
        start_commands=["./run_baseten.sh"],
        environment_variables={"HF_HUB_ENABLE_HF_TRANSFER": "0"},
        cache_config=CacheConfig(enabled=True),
        checkpointing_config=CheckpointingConfig(enabled=True),
    ),
)
project = TrainingProject(name="flame3-fire-lora", job=job)
