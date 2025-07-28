# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from copy import deepcopy

import torch
from datasets import load_dataset
from transformers.utils import is_peft_available
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers import Qwen2VLForConditionalGeneration
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    Qwen2VLForConditionalGeneration,
)
from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.models import create_reference_model
from trl.models import create_reference_model
from open_r1.model import LazyInitMoEQwen2VLForConditionalGeneration


if is_peft_available():
    from peft import PeftConfig, get_peft_model


def create_memory_efficient_reference_model(model: PreTrainedModel) -> PreTrainedModel:
    """
    Creates a memory-efficient reference model by sharing weights for all frozen parameters
    and only creating copies for the trainable ones.

    This is ideal for scenarios like partial fine-tuning where a large portion of the model is frozen.

    Args:
        model (PreTrainedModel): The trainable model, where some parameters have `requires_grad=False`.

    Returns:
        PreTrainedModel: A new model instance that acts as the reference model.
    """
    # 1. 创建一个与原模型结构相同但权重是随机初始化的“空壳”模型。
    #    这是我们将要填充的目标。
    print("Creating a memory-efficient reference model...")
    ref_model = model.__class__(model.config)

    if hasattr(ref_model, 'initialize_moe_modules') and callable(getattr(ref_model, 'initialize_moe_modules')):
        print("  - Found and calling 'initialize_moe_modules' for the reference model.")
        ref_model.initialize_moe_modules()

    # 2. 遍历原模型的所有模块（例如，一个注意力层、一个MLP块等）。
    #    我们将逐个模块地决定是共享还是复制。
    for name, original_module in model.named_modules():
        if not name:  # 跳过根模块本身
            continue

        # 3. 检查当前模块是否包含任何可训练的参数。
        is_trainable_module = False
        for param in original_module.parameters(recurse=False): # recurse=False 只检查当前模块的直接参数
            if param.requires_grad:
                is_trainable_module = True
                break
        
        # 检查子模块中是否有可训练参数，因为一个block可能自身没参数，但子模块有
        if not is_trainable_module:
            for sub_param in original_module.parameters(recurse=True):
                 if sub_param.requires_grad:
                    is_trainable_module = True
                    break

        # 4. 根据模块是否可训练来决定策略
        try:
            path_parts = name.split('.')
            parent_module_ref = ref_model
            # 导航到 ref_model 中对应的父模块
            for part in path_parts[:-1]:
                parent_module_ref = getattr(parent_module_ref, part)
            
            module_name = path_parts[-1]

            if is_trainable_module:
                # 策略A：如果模块是可训练的，我们需要为 ref_model 创建一个独立的、冻结的副本。
                # print(f"  - Copying trainable module: {name}")
                copied_module = deepcopy(original_module)
                # 确保副本中的所有参数都被冻结
                for param in copied_module.parameters():
                    param.requires_grad = False
                setattr(parent_module_ref, module_name, copied_module)
            else:
                # 策略B：如果模块是完全冻结的，我们直接让 ref_model 引用原模型的模块。
                # 这就是节省内存的关键步骤。
                # print(f"  - Sharing frozen module: {name}")
                setattr(parent_module_ref, module_name, original_module)

        except AttributeError:
            # 某些模块（如 Dropout）可能不存在于所有路径中，可以安全地跳过。
            # print(f"  - Skipping module not found in ref_model: {name}")
            continue

    # 5. 确保整个 ref_model 处于评估模式
    ref_model.eval()
    print("Memory-efficient reference model created successfully.")
    return ref_model


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    labels  = kwargs['answer']

    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, label in zip(contents, labels):
        reward = 0.0

        try:
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r"<answer>(.*?)</answer>", content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()

            # Compare the extracted answers
            if student_answer == label:
                reward = 1.0
        except Exception:
            pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Label: {label}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    else:
        raise NotImplementedError("Only image-based datasets are supported for now.")
        # dataset = dataset.map(make_conversation)
        # dataset = dataset.remove_columns("messages")

    args = training_args
    model = model_args.model_name_or_path
    peft_config=get_peft_config(model_args)
    
    # Args
    if args is None:
        model_name = model if isinstance(model, str) else model.config._name_or_path
        model_name = model_name.split("/")[-1]
        args = GRPOConfig(f"{model_name}-GRPO")

    # Models
    # Trained model
    model_init_kwargs = args.model_init_kwargs or {}
    model_init_kwargs["attn_implementation"] = model_args.attn_implementation
    if isinstance(model, str):
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )

        if re.search(r'-\d+e\d*-', model_id):
            model = LazyInitMoEQwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            for n, p in model.named_parameters():
                if 'moe' in n or 'embed_tokens' in n:
                    p.requires_grad = True
                    print(f'{n}: {p.shape}')
                else:
                    p.requires_grad = False
        elif "qwen2-vl" in model_id:
            model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        elif "Aria" in model_id:
            model_init_kwargs.pop("use_cache")
            model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
    else:
        model_id = model.config._name_or_path
        if args.model_init_kwargs is not None:
            raise ValueError(
                "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                "This argument can only be used when the `model` argument is a string."
            )
        
    if peft_config is not None:
        model = get_peft_model(model, peft_config)

    # Reference model
    if is_deepspeed_zero3_enabled():
        if re.search(r'-\d+e\d*-', model_id):
            raise ValueError("DeepSpeed ZeRO-3 is not supported for MoE models.")
        elif "qwen2-vl" in model_id:
            ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif "Aria" in model_id:
            ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        else:
            ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
    elif peft_config is None:
        # If PEFT configuration is not provided, create a reference model based on the initial model.
        if re.search(r'-\d+e\d*-', model_id):
            ref_model = create_memory_efficient_reference_model(model)
            ref_model.to(dtype=torch.bfloat16)
        else:
            ref_model = create_reference_model(model)
    else:
        # If PEFT is used, the reference model is not needed since the adapter can be disabled
        # to revert to the initial model.
        ref_model = None

    # Processing class
    if "qwen2-vl" in model_id or "Aria" in model_id:
        processing_class = AutoProcessor.from_pretrained(model_id)
        pad_token_id = processing_class.tokenizer.pad_token_id
        processing_class.pad_token_id = pad_token_id
        processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        if "qwen2-vl" in model_id:
            processing_class.image_processor.max_pixels = script_args.max_pixels
            processing_class.image_processor.min_pixels = script_args.min_pixels
    else:
        processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        pad_token_id = processing_class.pad_token_id

    # Reward functions
    if not isinstance(reward_funcs, list):
        reward_funcs = [reward_funcs]
    for i, reward_func in enumerate(reward_funcs):
        if isinstance(reward_func, str):
            reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                reward_func, num_labels=1, **model_init_kwargs
            )

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
