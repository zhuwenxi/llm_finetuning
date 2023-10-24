import os
import sys
from typing import List, Union

import fire
import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from utils.prompter import AlpacaPrompter, PromptSelector
from utils.text import load_text_file


class PeftTrainer(Trainer):
    def _save_checkpoint(self, _, trial, metrics=None):
        """Don't save base model, optimizer etc.
        but create checkpoint folder (needed for saving adapter)"""
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        os.makedirs(output_dir, exist_ok=True)

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


def ours_prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    )
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # if not is_gptq_quantized:
    if not is_gptq_quantized and not loaded_in_kbit:
        # cast all non INT8 parameters to fp32
        for param in model.parameters():
            if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                param.data = param.data.to(torch.float32)

    if (loaded_in_kbit or is_gptq_quantized) and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


class PeftSavingCallback(TrainerCallback):
    """Correctly save PEFT model and not full model"""

    def _save(self, model, folder):
        peft_model_path = os.path.join(folder, "adapter_model")
        model.save_pretrained(peft_model_path)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Save final best model adapter"""
        pass

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        self._save(kwargs["model"], folder)


# noinspection PyTypeChecker
def train(
    # model/data params
    base_model: str = "/data_test/LLM/huggingface/checkpoints/Qwen-14B",  # the only required argument
    data_path: str = "/public/home/macong/neurIPS2023/data/alpaca_data_extra_long.json",
    output_dir: str = "./output_test",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    lr_scheduler_type: str = "linear",
    cutoff_len: int = 2048,
    val_set_size: int = 0.2,
    eval_steps: int = 100,
    save_steps: int = 10,
    logging_steps: int = 10,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = (
        "w1",
        "w2",
        "mlp.c_proj",
    ),
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "huggingface",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
    # memory optimization params
    gradient_checkpointing: bool = True,
    # GPTQ specific params
    gptq_backend: str = "cuda",  # GPTQ backend "cuda" or "triton"
    gptq_groupsize: int = 128,
    # evaluation flag
    eval: bool = False,
    max_steps: int = -1,
    activation_type: str = "bf16",
    mode: Union[int, str] = 16,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"eval: {eval}\n"
            f"gradient_checkpointing: {gradient_checkpointing}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"lr_scheduler_type: {lr_scheduler_type}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"eval_steps: {eval_steps}\n"
            f"logging_steps: {logging_steps}\n"
            f"save_steps: {save_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt_template: {prompt_template}\n"
            f"max_steps: {max_steps}\n"
            f"mode: {mode}\n"
            f"activation_type: {activation_type}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = PromptSelector.from_template_name(prompt_template, verbose=False)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )

    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # setup model and tokenizer
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if activation_type != "bf16" and activation_type != "fp16":
        print("please set activation_type = bf16 or fp16")
        return

    is_use_bf16 = activation_type == "bf16"
    is_use_fp16 = activation_type == "fp16"
    torch_dtype = torch.float16 if activation_type == "fp16" else torch.bfloat16

    kwargs = {"device_map": device_map}
    if mode == 16:
        kwargs["torch_dtype"] = torch_dtype
    elif mode == 4:
        kwargs["torch_dtype"] = torch_dtype
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    else:
        raise NotImplementedError(f"Mode '{mode}' is not supported.")

    # setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=False, trust_remote_code=True
    )
    if tokenizer.__class__.__name__ == "QWenTokenizer":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            fp16=is_use_fp16,
            bf16=is_use_bf16,
            trust_remote_code=True,
            **kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, trust_remote_code=True, **kwargs
        )

    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    if mode != 16:
        # model = prepare_model_for_kbit_training(model)
        model = ours_prepare_model_for_kbit_training(model)

    if lora_config:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # setup model checkpoint if neeeded
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location="cpu")
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    print(f"{model}")

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    if tokenizer.__class__.__name__ == "QWenTokenizer":
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(**data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            assert isinstance(prompter, AlpacaPrompter)
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    # data preparation
    # check if using raw text format (prompter is None)
    if prompter is None:
        train_data = load_text_file(
            data_path, tokenizer, cutoff_len=cutoff_len, overlap_len=cutoff_len // 2
        )

        if val_set_size > 0:
            train_val = train_data.train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = train_val["train"].shuffle(seed=42)
            val_data = train_val["test"]
        else:
            val_data = None
    else:
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=data_path)
        else:
            if os.path.exists(data_path):
                data = load_dataset(
                    "json",
                    data_files={
                        "train": data_path + "/train.json",
                        "test": data_path + "/test.json",
                    },
                )
            else:
                data = load_dataset(data_path)

        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle(seed=42).map(generate_and_tokenize_prompt)
            )
        else:
            train_data = (
                data["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)
            )
            val_data = data["test"].map(generate_and_tokenize_prompt)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    model.config.use_cache = False
    # sanity check of model saving process
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.save_pretrained(output_dir)

    trainArg = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        max_steps=max_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        bf16=is_use_bf16,
        fp16=is_use_fp16,
        logging_steps=10,
        optim="paged_adamw_8bit" if mode in [4, 8] else "adamw_torch",
        evaluation_strategy="steps" if eval_steps > 0 else "no",
        save_strategy="steps",
        eval_steps=eval_steps if eval_steps > 0 else None,
        save_steps=save_steps,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="none",
        run_name=wandb_run_name if use_wandb else None,
    )

    trainer = PeftTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=trainArg,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[PeftSavingCallback],
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if eval:
        eval_results = trainer.evaluate()
        print(eval_results)
    else:
        trainer.train()

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            model.save_pretrained(output_dir)
            print(
                "\n If there's a warning about missing keys above, please disregard :)"
            )


if __name__ == "__main__":
    fire.Fire(train)
