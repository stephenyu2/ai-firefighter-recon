"""Baseten training job: paired precision study on an RTX PRO 6000."""
from truss.base.truss_config import AcceleratorSpec
from truss_train import CacheConfig, CheckpointingConfig, Compute, Image, Runtime, TrainingJob, TrainingProject

job = TrainingJob(
    image=Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"),
    compute=Compute(accelerator=AcceleratorSpec(accelerator="RTX_PRO_6000", count=1)),
    runtime=Runtime(start_commands=["./run_baseten.sh"], cache_config=CacheConfig(enabled=True), checkpointing_config=CheckpointingConfig(enabled=True)),
)
project = TrainingProject(name="flame3-fire-lora", job=job)
