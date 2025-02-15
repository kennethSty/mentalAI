import tiktoken
from torch.utils.data import DataLoader
import torch

from ..GPTModel import GPTModel
from ..SpamDataset import SpamDataset

def finetune_model(model: GPTModel, train_loader: DataLoader, 
    val_loader: DataLoader, optimizer: torch.optim.Optimizer,
    device: str, num_epochs: int, eval_freq: int, eval_iter: int):

    examples_seen, global_step = 0, 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss) 
                print(f"Ep {epoch+1} (step {global_step:06d}): ")
                print(f"Train loss {train_loss:.3f}")
                print(f"Val loss {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches = eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches = eval_iter
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}%")
        print(f"Val accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def evaluate_model(model: GPTModel, train_loader: DataLoader, val_loader: DataLoader, device: str, eval_iter: int):
    model.eval()
    with torch.inference_mode():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches = eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches = eval_iter
        )
    model.train()
    return train_loss, val_loss


def calc_accuracy_loader(data_loader: DataLoader, model: GPTModel, device: str, num_batches = None):

    model.eval()
    correct_predictions, num_examples = 0,0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else: 
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            #take logit of last token because the last token has attn scores to all other tokens
            with torch.inference_mode():
                logits = model(input_batch)[:, -1, :]  
            pred_labels = torch.argmax(logits, dim=-1)
            num_examples += pred_labels.shape[0]
            correct_predictions += (pred_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples 

def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: GPTModel, device: str):
    batch_size, n_tokens = input_batch.shape
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)            
    logits = model(input_batch)[:, -1, :] #take logits of last token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader: DataLoader, model: GPTModel, device: str, num_batches = None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else: 
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def setup_dataloaders(
    batch_size: int, 
    tokenizer: tiktoken.Encoding,
    train_ds_path: str, 
    test_ds_path:str , 
    val_ds_path: str):

    train_dataset = SpamDataset(
        csv_file=train_ds_path, tokenizer=tokenizer
    )    
    test_dataset = SpamDataset(
        csv_file=test_ds_path, 
        max_length=train_dataset.max_length, #ensure consistent padding across train, val, testf
        tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file=val_ds_path, 
        max_length=train_dataset.max_length, 
        tokenizer=tokenizer
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    return train_loader, test_loader, val_loader
