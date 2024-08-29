import torch
from transformers import MarianMTModel, MarianTokenizer

# 加载预训练的模型和分词器
model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 确保模型在合适的设备上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 翻译函数
def translate_text(text, tokenizer, model):
    # 分词
    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    
    # 生成翻译结果
    translated = model.generate(input_ids, max_length=128)
    
    # 解码翻译结果
    result = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return result

# 示例文本
text_to_translate = "Hello, how are you?"

# 执行翻译
translated_text = translate_text(text_to_translate, tokenizer, model)
print("Translated Text:", translated_text)