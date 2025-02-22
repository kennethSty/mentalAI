from torch.utils.data import DataLoader
import torch
import os
import torch.nn as nn

from src._3_model_preparation.gpt_architecture.GPTModel import GPTModel
from src._3_model_preparation.psychbert_architecture.PsychBertClassifier import PsychBertClassifier

def finetune_loop(model: nn.Module, train_loader: DataLoader,
                  val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                  device: str, num_epochs: int, eval_freq: int, checkpoint_freq: int, eval_iter: int, checkpoint_dir='../../../gpt2_checkpoints'):

    examples_seen, global_step = 0, 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch, attention_mask_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, attention_mask_batch, model, device
            )
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1
            print("Done learning steps: ", global_step)

            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(
                    model, train_loader, device, eval_iter
                )
                val_loss = calc_loss_loader(
                    model, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss) 
                print(f"Ep {epoch+1} (step {global_step:06d}): ")
                print(f"Train loss {train_loss:.3f}")
                print(f"Val loss {val_loss:.3f}")
                print(f"Saving checkpoint")
                train_accuracy = calc_accuracy_loader(
                    data_loader=train_loader, model=model, device=device, num_batches=eval_iter
                )
                val_accuracy = calc_accuracy_loader(
                    data_loader=val_loader, model=model, device=device, num_batches=eval_iter
                )
                print(f"Training accuracy: {train_accuracy * 100:.2f}%")
                print(f"Val accuracy: {val_accuracy * 100:.2f}%")
                train_accs.append(train_accuracy)
                val_accs.append(val_accuracy)
            if global_step % checkpoint_freq == 0:
                save_checkpoint(model, optimizer, epoch, global_step,
                    train_losses, val_losses, train_accs, val_accs,checkpoint_dir)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, attention_mask_batch: torch.Tensor, model: nn.Module, device: str):
    #batch_size, n_tokens = input_batch.shape
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    if isinstance(model, PsychBertClassifier):
        attention_mask_batch = attention_mask_batch.to(device)
        logits = model(input_batch, attention_mask_batch)
    else:
        logits = model(input_batch) #is logits of last token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader: DataLoader, model: nn.Module, device: str, num_batches = None):
    total_loss = 0
    model.eval()
    for i, (input_batch, target_batch, attention_mask_batch) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            print(f"{num_batches} reached")
            break
        loss = calc_loss_batch(
            input_batch, target_batch, attention_mask_batch, model, device
        )
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches

def evaluate_model_predscores(model: nn.Module, data_loader: DataLoader, device: str, num_batches: int):
    model.eval()
    with torch.inference_mode():
        true_positives, true_negatives, false_positives, false_negatives, false_classified_inputs = calc_predscores_loader(
            data_loader, model, device, num_batches = num_batches
        )

    n_examples = true_positives + true_negatives + false_positives + false_negatives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / n_examples
    f1 = 2 * (precision * recall) / (precision + recall)
    model.train()
    return precision, recall, accuracy, f1, false_classified_inputs

def calc_accuracy_loader(data_loader: DataLoader, model: nn.Module, device: str, num_batches = None):
    model.eval()
    true_positives, true_negatives, false_positives, false_negatives, _ = calc_predscores_loader(data_loader, model, device, num_batches)
    return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

def calc_predscores_loader(data_loader: DataLoader, model: nn.Module, device: str, num_batches = None):
    model.eval()
    false_classified_inputs = []
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

    for i, (input_batch, target_batch, attention_mask_batch) in enumerate(data_loader):
        if i % 500 == 0:
            print(f"evaluated {i} instances")
        if num_batches is not None and i >= num_batches:
            break
        with torch.inference_mode():
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            if isinstance(model, PsychBertClassifier):
                attention_mask_batch = attention_mask_batch.to(device)
                logits = model(input_batch, attention_mask_batch)
            else:
                logits = model(input_batch)
        pred_labels = torch.argmax(logits, dim=-1)
        pred_is_correct = (pred_labels == target_batch)
        pred_is_wrong = (pred_labels != target_batch)
        true_positives += (pred_is_correct & (pred_labels == 1)).sum().item()
        true_negatives += (pred_is_correct & (pred_labels == 0)).sum().item()
        false_positives += (pred_is_wrong & (pred_labels == 1)).sum().item()
        false_negatives += (pred_is_wrong & (pred_labels == 0)).sum().item()

        if false_positives + false_negatives > 0:
            for j in range(len(input_batch)):
                if pred_is_wrong[j]:
                    false_classified_inputs.append({"pred_label": pred_labels[j].cpu().numpy(), "input": input_batch[j].cpu().numpy()})

    model.train()
    return true_positives, true_negatives, false_positives, false_negatives, false_classified_inputs


def print_pretrain_accuracy(model: nn.Module, dataloader: DataLoader, device: str, label: str):
    accuracy = calc_accuracy_loader(
        dataloader, model, device, num_batches=4
    )
    print(f"{label}-accuracy before training:", accuracy)
    return

def save_checkpoint(model, optimizer, epoch, global_step, train_losses, val_losses,
                    train_accs, val_accs, checkpoint_dir):

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")