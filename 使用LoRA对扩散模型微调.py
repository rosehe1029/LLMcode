import torch
from diffusers import StableDiffusionPipeline, LoRAConfig, LoRAScheduler
from transformers import CLIPTokenizer
from accelerate import Accelerator
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# 初始化加速器
accelerator = Accelerator()

# 加载预训练模型
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(accelerator.device)

# 初始化LoRA配置
lora_config = LoRAConfig(r=4, target_modules=["q", "v"], lora_alpha=32, lora_dropout=0.05)

# 初始化LoRA调度器
scheduler = LoRAScheduler(pipe.unet, pipe.text_encoder, lora_config)

# 微调LoRA权重
pipe.unet.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
scheduler.enable_training()

# 准备数据集
dataset = load_dataset("path/to/your/dataset")  # 替换为您的数据集路径
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

# 数据预处理
def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h))
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

# 构建训练函数
def train_function(examples):
    with accelerator.accumulate(scheduler):
        # 处理输入数据
        prompt = examples["prompt"]
        input_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
        
        # 生成图像
        with torch.no_grad():
            latents = pipe.prepare_latents(input_ids.shape[0], pipe.unet.config.in_channels, input_ids.dtype, accelerator.device, generator=None)
        
        # LoRA微调
        for t in range(100):  # 假设100个时间步
            noise_pred = scheduler.unet(latents, t, encoder_hidden_states=input_ids).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # 计算损失
        loss = torch.mean((latents - examples["target_latents"]) ** 2)  # 假设examples中包含目标latents
        accelerator.backward(loss)
        scheduler.step()

# 微调循环
num_epochs = 5
batch_size = 1
for epoch in range(num_epochs):
    for batch in dataset.as_iterable_dataset(batch_size=batch_size, shuffle=True):
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        train_function(batch)
        
        if accelerator.is_main_process and (epoch + 1) % 10 == 0:
            # 保存LoRA权重
            scheduler.save_pretrained(f"lora_weights_epoch_{epoch+1}")
            print(f"Epoch {epoch+1} finished.")

# 结束训练
scheduler.disable_training()