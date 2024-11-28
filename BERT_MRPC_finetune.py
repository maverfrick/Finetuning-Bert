import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel
from transformers import BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import Trainer   
import evaluate
import torch.nn as nn
import sys

#clear cache
torch.cuda.empty_cache()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

#load datasets
dataset = load_dataset("glue", "mrpc")


#example_batch = next(iter(train_dataset))


#for k, v in example_batch.items():
#  print({k : v})
  

model_name = "google-bert/bert-base-uncased"
base_model = AutoModel.from_pretrained(model_name, device_map = 'cpu')


class SequenceClassificationModel(nn.Module):
  def __init__(self, base_model, num_labels=2):
    super(SequenceClassificationModel, self).__init__()
    self.bert = base_model
    self.dropout = nn.Dropout(0.1)
    self.sc = nn.Linear(base_model.config.hidden_size, num_labels) # sequence classifier layer with size R^(H * label) and H = 12
    
  def forward(self, input_ids, attention_mask): 
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    CLS_output = outputs[0][:, 0, :] #CLS have token for classification
    CLS_output = self.dropout(CLS_output)
    logits = self.sc(CLS_output) # raw output
    
    return logits
    
model = SequenceClassificationModel(base_model)

#import sys
#device_ids = range(torch.cuda.device_count())
#print(device_ids)
#sys.exit(1)

# If using multiple GPUs, wrap the model in DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0,1,2])

# Check device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
    
# input_ids = torch.tensor([[101, 2054, 2003, 102]]).to(device)  # Example input IDs
# attention_mask = torch.tensor([[1, 1, 1, 0]]).to(device)  # Example attention mask

tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
  return tokenizer(examples['sentence1'], examples['sentence2'], padding="max_length", truncation=True)
  
tokenized_data = dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns and rename the label column
tokenized_data = tokenized_data.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_data = tokenized_data.rename_column("label", "labels")
tokenized_data.set_format("torch")

train_dataset = tokenized_data['train'].shuffle(seed=42).select(range(1000))
val_dataset = tokenized_data['test'].shuffle(seed=42).select(range(1000))

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)
softmax_loss = nn.CrossEntropyLoss() #softmax loss (CW)

# Set up learning rate scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
  epoch_loss = 0
  for batch in train_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # batch = {k: v.to(device) for k, v in batch.items()}
    
    logits = model(input_ids, attention_mask)
    loss = softmax_loss(logits, labels)
    epoch_loss += loss.item()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)
    
  print(f"Epoch {epoch+1} loss: {epoch_loss/len(train_dataloader)}")
  
# Evaluation
metric = evaluate.load("accuracy")
model.eval()
for batch in val_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

# Compute final accuracy
result = metric.compute()
print("Accuracy:", result["accuracy"])

def save_model(model, tokenizer, save_directory):
    # Save model state dict
    torch.save(model.state_dict(), f"{save_directory}/pytorch_model.bin")
    # Save tokenizer
    tokenizer.save_pretrained(save_directory)


model_to_save = model.module if hasattr(model, 'module') else model
save_model(model_to_save, tokenizer, './FineTuneGlue_Model')









  
  
  
  
  
  
  