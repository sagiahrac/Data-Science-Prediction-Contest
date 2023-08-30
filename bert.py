from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
import pandas as pd
import tqdm
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from helpers.getmodel import get_data_for_bert, get_train_val_indices

DATA_PATH = 'data/food_train.csv'
NUM_EPOCHS = 20
LR_RATE = 0.0001
PRETRAINED_MODEL_NAME = 'bert-base-uncased'
N_EPOCHS = 50
BATCH_SIZE=2

text, categories = get_data_for_bert()
train_indices, val_indices = get_train_val_indices()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=6)
model = model.to(device)

# Unfreeze the last two layers
unfreeze_layers = ['encoder.layer.10', 'encoder.layer.11']
layers_count = 0
for name, param in model.named_parameters():
    for layer_name in unfreeze_layers:
        if layer_name in name:
            param.requires_grad = True
            layers_count += 1
            break  # Once we find the layer, move to the next parameter
print(f'Unfrozen {layers_count} layers')


class CustomDataset(Dataset):
    def __init__(self, text, categories):
        self.text = text
        self.categories = categories
        self.dmap = {
            "cakes_cupcakes_snack_cakes": 0,
            "candy": 1,
            "chips_pretzels_snacks": 2,
            "chocolate": 3,
            "cookies_biscuits": 4,
            "popcorn_peanuts_seeds_related_snacks": 5,
        }

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, idx):
        return self.text[idx], self.dmap[self.categories[idx]]

# data = food_train.to_records(index=False).tolist()
dataset = CustomDataset(text, categories)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE, betas=(0.9, 0.999), weight_decay=0.01)

def lr_lambda(current_step):
    return max(0.0, 1.0 - current_step / N_EPOCHS)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

best_loss = float("inf")
no_improvement = 0


# Training loop
for epoch in range(N_EPOCHS):
    print(f'\n\nEpoch {epoch+1}:')
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()  # Set model to training mode
            indices = np.random.choice(train_indices, BATCH_SIZE*100, replace=False)
        else:
            model.eval()  # Set model to evaluate mode
            indices = val_indices

        sampler = SubsetRandomSampler(indices)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

        loop = tqdm.tqdm(data_loader)
        accuracies, losses = [], []
        for batch_texts, batch_labels in loop:
            with torch.set_grad_enabled(phase == "train"):
                batch_input = tokenizer.batch_encode_plus(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=80,
                    return_tensors='pt'
                )
                batch_input = batch_input.to(device)
                batch_labels = batch_labels.to(device)

                input_ids = batch_input['input_ids']
                attention_mask = batch_input['attention_mask']
                labels = batch_labels
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                a = [x.item() for x in (outputs.logits.argmax(dim=-1) == labels).float()]
                b = [x.item() for x in labels.float()]
                tmp = pd.DataFrame({'a': a, 'b': b})
                tmp = tmp[tmp['a']==0]
                tmp['b'].value_counts().to_csv('batch.csv')
                
                if phase == "train":
                    loss.backward()
                    optimizer.step()


                accuracy = (outputs.logits.argmax(dim=-1) == labels).float().mean()
                accuracies.append(accuracy)
                losses.append(loss.item())

                loop.set_description(
                            f"[{phase.capitalize()}] Epoch [{epoch+1}/{N_EPOCHS}]"
                        )
                loop.set_postfix(
                            loss=loss.item(),
                            acc=accuracy.item(),
                        )

        print(f'Average Acc = {sum(accuracies) / len(accuracies)}')
        print(f'Average Loss = {sum(losses) / len(losses)}')
        if phase == 'train':
            scheduler.step()
        if phase == 'val':
            if best_loss > sum(losses) / len(losses):
                best_loss = sum(losses) / len(losses)
                torch.save(model.state_dict(), 'bert_model.pt')
                print('Model saved!')
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement == 6:
                    print('Early stopping!')
                    exit()
