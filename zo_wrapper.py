import torch
from zo_trainer import OurTrainer
from transformers import TrainingArguments
from dataclasses import dataclass

@dataclass
class ZOTrainingArguments(TrainingArguments):
    output_dir: str="tmp/"
    task_name: str = "3dgs"
    num_train: int = 0
    num_dev: int = None
    num_eval: int = None
    num_train_sets: int = None
    train_set_seed: int = 0
    result_file: str = None

    model_name: str = "3dgs"
    load_float16: bool = False
    load_bfloat16: bool = False
    load_int8: bool = False
    max_length: int = 2048
    no_auto_device: bool = False

    sfc: bool = False
    icl_sfc: bool = False
    template_ver: int = 0

    trainer: str = "zo_sgd"
    optimizer: str = "adam"
    only_train_option: bool = True
    train_as_classification: bool = False
    momentum: float = 0.0

    zo_eps: float = 1e-3
    perturbation_mode: str = "two_side"
    q: int = 1

    prefix_tuning: bool = False
    num_prefix: int = 5
    no_reparam: bool = True
    prefix_init_by_real_act: bool = True

    prompt_tuning: bool = False
    num_virtual_tokens: int = 10
    prompt_init_by_real_tokens: bool = False

    lora: bool = False
    lora_alpha: int = 16
    lora_r: int = 8

    sampling: bool = False
    temperature: float = 1.0
    num_beams: int = 1
    top_k: int = None
    top_p: float = 0.95
    max_new_tokens: int = 50
    eos_token: str = "\n"

    save_model: bool = False
    no_eval: bool = False
    tag: str = ""

    linear_probing: bool = False
    lp_early_stopping: bool = False
    head_tuning: bool = False
    untie_emb: bool = False
    verbose: bool = False
    non_diff: bool = False
    save_on_interrupt: bool = False
    clean_model_at_end: bool = True

    gradient_sparsity: float = None
    sparse_gradient_resample_steps: int = 1
    sparse_gradient_group: str = "layer"

    module_wise_perturbation: bool = False
    perturbed_module_level: str = "transformer-block"
    coordinate_perturbation: bool = True
    report_to: list = None

class ZOWrapper:
    def __init__(self, model):
        self.args = ZOTrainingArguments()
        self.model = model
        # 你可以根据你的训练设置来传入真实的值
        # eval_samples = 4  # 或者其他合适的值
        perturb_module_regex = ".*"  # 表示所有 module 都会被加入扰动

        # 创建 OurTrainer 实例
        self.zo_trainer = OurTrainer(
            model=model,
            args=self.args,
            evaluate_func=None,
            eval_samples=[],
            dev_samples=[],
            perturb_module_regex=[]
        )

    def zo_step(self, inputs):
        return self.zo_trainer.zo_step(self.model, inputs)

    def zo_update(self):
        return self.zo_trainer.zo_update(self.model)
