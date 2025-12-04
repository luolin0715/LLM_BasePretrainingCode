from accelerate import PartialState
from datasets import load_dataset
from peft import TaskType, LoraConfig, get_peft_model
from transformers import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dataclasses import dataclass, field
import transformers
from itertools import chain
import torch
import warnings

@dataclass # dataclass 是一个装饰器
class CustomArguments(transformers.TrainingArguments):
    # Lora_r
    lora_r: int = field(default=8)
    # 数据处理时的并行进程数量
    num_proc: int = field(default=1)
    # 最大序列长度
    max_seq_length: int = field(default=32)
    # 验证策略，如不想验证，可以设置为 no
    eval_strategy: int = field(default="steps")
    # 每多少步进行一次验证
    eval_steps: int = field(default=100)
    # 随机种子
    seed: int = field(default=0)
    # 优化器
    optim: str = field(default="adamw_torch")
    # 训练epoch 数量
    num_train_epochs: int = field(default=2)
    # 每个设备上的批量大小
    per_devices_train_batch_size: int = field(default=1)

    # 学习率
    learning_rate: float = field(default=5e-5)
    # 权重衰减
    weight_decay: float = field(default=0)
    # 预热步数
    warmup_steps: int = field(default=10)
    # 学习率规划期望类型
    lr_scheduler_type: str = field(default="linear")
    # 是否使用梯度检查点
    gradient_checkpointing: bool = field(default=False)
    # 是否使用bf16作为混合精度训练类型
    bf16: bool = field(default=True)
    # 梯度累加步数
    gradient_accumulation_steps: int = field(default=1)

    # 日志记录的步长频率
    logging_steps: int = field(default=3)
    # checkpoint 保存策略
    save_strategy: str = field(default="steps")
    # checkpoint 保存的步长频率
    save_steps: int = field(default=3)
    # 总的保存checkpoint的数量
    save_total_limit: int = field(default=2)

parser = transformers.HfArgumentsParser(CustomArguments) # 创建了一个参数解析器
training_args, = parser.parse_args_into_dataclasses() # 解析参数

model_path = ""

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage = True,
    quantization_config = bnb_config,
    device_map = {"": PartialState().process_index}
)

peft_config = LoraConfig(
    r = training_args.lora_r,
    target_modules = [
        "p_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    task_type = TaskType.CAUSAL_LM,
    lora_alpha = 16,
    lora_dropout = 0.05
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

train_dataset = load_dataset("text", data_dir="", split = "train") # "text"：指定加载纯文本格式的数据
eval_dataset = load_dataset("text", data_dir="", split = "train")

def tokenization(example):
    return tokenizer(example["text"])

# main_process_first：确保在主进程中先执行，避免分布式训练中的数据竞争
# map：对数据集中的每个样本应用tokenization函数
# remove_columns=["text"]：移除原始的文本列，只保留分词后的结果
# num_proc：使用多进程并行处理，加速分词过程
with training_args.main_process_first(desc="dataset map tokenization"):
    train_dataset = train_dataset.map(tokenization, remove_columns=["text"], num_proc=training_args.num_proc)
    eval_dataset = eval_dataset.map(tokenization, remove_columns=["text"], num_proc=training_args.num_proc)

def group_texts(examples):
     # 1. 将所有样本拼接成一个长序列
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
     # 假设examples = {"input_ids": [[1,2,3], [4,5,6]]}
    # 拼接后：concatenated_examples = {"input_ids": [1,2,3,4,5,6]}

     # 2. 计算可整除的最大长度
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 3. 按固定长度切分
    total_length = (total_length // training_args.max_seq_length ) * training_args.max_seq_length
    result = {
        k: t[i: i + training_args.max_seq_length] for i in range(0, total_length, training_args.max_seq_length)
        for k, t in concatenated_examples.items()
    }
    
     # 4. 为因果语言建模创建标签
    result["labels"] = result["input_ids"].copy()
    return result

with training_args.main_process_first(desc = "dataset map tokenization"):
    train_dataset = train_dataset.map(group_texts, num_proc=training_args.num_proc, batched=True)
    eval_dataset = eval_dataset.map(group_texts, num_proc=training_args.num_proc, batched=True)

if __name__ == "__main__":
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )
    trainer.train()
    trainer.save_model("output_dir")


# 关于mian_process_first
# 时间线：
        
# GPU 0 (主进程): [main_process_first开始] → [启动4个工作进程] → [等待完成] → [保存缓存] → [训练]
#                           ↓              ↓              ↓           ↓
#                           并行处理数据    并行处理数据    完成处理     缓存到磁盘
                          
# GPU 1:                    [等待]          [等待]         [等待]      [读取缓存] → [训练]