import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载预训练的GPT-2模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 确保模型在合适的设备上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 设置生成文本的参数
max_length = 100  # 生成文本的最大长度
temperature = 0.7  # 控制生成文本的随机性，值越低生成的文本越确定
top_k = 0  # 只从最有可能的前k个token中选择下一个token
top_p = 0.9  # 核采样，只考虑累积概率为top_p的token

# 文本生成函数
def generate_text(prompt, max_length, temperature, top_k, top_p):
    # 分词
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # 生成文本
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=model.config.eos_token_id,
        do_sample=True  # 允许随机采样
    )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

# 提示文本
prompt = "Once upon a time, in a land far far away,"

# 生成文本
generated_text = generate_text(prompt, max_length, temperature, top_k, top_p)
print(generated_text)