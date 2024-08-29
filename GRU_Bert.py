import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizerFast
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 设定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# 定义带有GRU的模型
class GRUBERTClassifier(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(GRUBERTClassifier, self).__init__()
        self.bert = bert_model
        self.gru = nn.GRU(input_size=768, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # 获取BERT的输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state
        
        # 传递到GRU
        gru_output, _ = self.gru(last_hidden_state)
        
        # 使用GRU的最后一个时间步的输出进行分类
        logits = self.classifier(gru_output[:, -1, :])
        
        return logits

# 实例化模型
model = GRUBERTClassifier(hidden_size=128, num_layers=1, num_classes=2).to(device)

# 加载情感分析的数据集，例如IMDb电影评论数据集
dataset = load_dataset("imdb")

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# 对数据集进行预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 将数据集划分为训练集和验证集
train_dataset = encoded_dataset['train']
eval_dataset = encoded_dataset['test']

# 定义训练参数
batch_size = 16
learning_rate = 2e-5
num_epochs = 3

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # 评估模型
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# 保存模型
torch.save(model.state_dict(), "./grubert_classifier.pth")