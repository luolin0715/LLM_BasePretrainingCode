import os
os.environ["CUDA_VISIBLE_DEVICES"] = 0
from peft import TaskType, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM, LlamaModel
import torch

model_path = ""
tokenzier = AutoTokenizer.from_pretrained(model_path)

# 1. 从零开始训练大模型
config = LlamaConfig() # 创建一个默认的Llama config
config.num_hidden_layers = 12 # 配置网络结构
config.hidden_size = 1024
config.intermediate_size = 4096
config.num_key_values_heads = 8
# 用配置文件初始化一个大模型
model = LlamaForCausalLM(config)

# 2. 加载一个预训练的大模型

# 4bit load
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True, # 进行两次量化，第一次量化，将权重量化为4位，第二次量化，将量化常数（quantization constants）本身也进行量化
    bnb_4bit_quant_type = "nf4", # 使用NF 4（normal float4）数据类型， 替代选项是fp4
    bnb_4bit_compute_dtype = torch.bfloat16 # 计算时使用的额数据类型，虽然权重是4位的，但在实际计算（矩阵乘法等）时，需要将权重饭量化为更高精度
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage = True,
    quantization_config = bnb_config
)

peft_config = LoraConfig(
    r = 8,
    target_modules = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj"
    ],
    task_type = TaskType.CUASAL_LM,
    lora_alpha = 16,
    lora_dropout = 0.05
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.to("cuda")
optimizer = torch.optim.AdamW(model.parameters())

text = "今天天气不错"
input = tokenzier(text, return_tensors = "pt")
input = {k:v.to("cuda") for k, v in input.items()}

input["labels"] = input["input_ids"].clone()

output = model(**input)


# 获取模型的loss
loss = output.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()

model.save_pretrained("output_dir")

# 关于taskType 的
## 主要的taskType 选项
# 1. CAUSAL_LM -- 因果语言建模
# 用于文本生成、对话、续写等任务
# 模型示例：GPT、Llama 等自回归模型
# 2. SEQ_CLS -- 序列分类
# 用于文本分类、情感分析、意图识别等
# 模型示例：BERT 等编码器模型
# 3. SEQ_2_SEQ_LM -- 序列到序列语言建模
# 用于翻译、摘要、问答等
# 模型示例： T5、BART 等
# 4. TOKEN_CLS -- 标记分类
# 用于命名实体识别、词性标注等
# 特点：为每个token 分配标签
# 5. QUESTION_ANS -- 问答任务
# 用于抽取式问答的任务场景
# 特点：从上下文中找出答案的起始和结束为止
