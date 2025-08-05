import torch
from zo_trainer import OurTrainer
from transformers import TrainingArguments


class ZOTrainingArguments(TrainingArguments):
    def __init__(self):
        self.output_dir = "./tmp"
        self.do_train = False
        self.do_eval = False
        self.per_device_train_batch_size = 1
        self.evaluation_strategy = "no"
        self.save_strategy = "no"
        self.load_best_model_at_end = False
        self.lr_scheduler_type = "constant"
        self.logging_steps = 10
        self.num_train_epochs = 5
        self.learning_rate = 1e-3
        self.weight_decay = 0.0
        self.fp16 = False
        self.seed = 42
        self.full_determinism = False
        self.skip_memory_metrics = True

        # Zero-order 特有参数
        self.zo_eps = 1e-3
        self.q = 1
        self.trainer = "zo_adam"  # "zo_sgd", "zo_sign_opt" 也可以
        self.perturbation_mode = "two_side"
        self.gradient_accumulation_steps = 1
        self.module_wise_perturbation = False
        self.report_to=["all"]

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
