"""Baseten training job: final DINOv2-g QLoRA checkpoint only, threshold from the committed CV scores (H100)."""
from truss.base.truss_config import AcceleratorSpec
from truss_train import CacheConfig, CheckpointingConfig, Compute, Image, Runtime, TrainingJob, TrainingProject

job = TrainingJob(
    image=Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"),
    compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=1)),
    runtime=Runtime(
        start_commands=["./run_baseten.sh"],
        environment_variables={"HF_HUB_ENABLE_HF_TRANSFER": "0", "FINAL_ONLY": "1", "TRAIN_ARGS": "dinov2g_qlora"},
        cache_config=CacheConfig(enabled=True),
        checkpointing_config=CheckpointingConfig(enabled=True),
    ),
)
project = TrainingProject(name="flame3-fire-lora", job=job)
