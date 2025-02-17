from torch.utils.data import DataLoader
import torch
import os
import torch.nn as nn

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

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    train_accs = checkpoint['train_accs']
    val_accs = checkpoint['val_accs']

    print(f"Checkpoint loaded from {checkpoint_path}")
    return epoch, global_step, train_losses, val_losses, train_accs, val_accs

def finetune_loop(model: nn.Module, train_loader: DataLoader,
                  val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                  device: str, num_epochs: int, eval_freq: int, checkpoint_freq: int, eval_iter: int, checkpoint_dir='../../../gpt2_checkpoints'):

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
            print("Done learning steps: ", global_step)

            if global_step % eval_freq == 0:
                train_loss = evaluate_model_loss(
                    model, train_loader, device, eval_iter
                )
                val_loss = evaluate_model_loss(
                    model, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss) 
                print(f"Ep {epoch+1} (step {global_step:06d}): ")
                print(f"Train loss {train_loss:.3f}")
                print(f"Val loss {val_loss:.3f}")
                print(f"Saving checkpoint")
                train_accuracy = evaluate_model_acc(
                    data_loader=train_loader, model=model, device=device, num_batches=eval_iter
                )
                val_accuracy = evaluate_model_acc(
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



def calc_accuracy_loader(data_loader: DataLoader, model: nn.Module, device: str, num_batches = None):

    model.eval()
    correct_predictions, num_examples = 0,0

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            break
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        #take logit of last token because the last token has attn scores to all other tokens
        with torch.inference_mode():
            logits = model(input_batch)
        pred_labels = torch.argmax(logits, dim=-1)
        num_examples += pred_labels.shape[0]
        correct_predictions += (pred_labels == target_batch).sum().item()

    return correct_predictions / num_examples 

def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: nn.Module, device: str):
    #batch_size, n_tokens = input_batch.shape
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)            
    logits = model(input_batch) #is logits of last token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader: DataLoader, model: nn.Module, device: str, num_batches = None):
    total_loss = 0

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            print(f"{num_batches} reached")
            break
        loss = calc_loss_batch(
            input_batch, target_batch, model, device
        )
        total_loss += loss.item()

    return total_loss / num_batches

def assess_pretrain_accuracy(model: nn.Module, dataloader: DataLoader, device: str, label: str):
    model.eval()
    with torch.inference_mode():
        accuracy = calc_accuracy_loader(
            dataloader, model, device, num_batches=4
        )
    print(f"{label}-accuracy before training:", accuracy)
    model.train()
    return

def evaluate_model_loss(model: nn.Module, data_loader: DataLoader, device: str, num_batches: int):
    model.eval()
    with torch.inference_mode():
        accuracy = calc_loss_loader(
            data_loader, model, device, num_batches = num_batches
        )
    return accuracy

def evaluate_model_acc(model: nn.Module, data_loader: DataLoader, device: str, num_batches: int):
    model.eval()
    with torch.inference_mode():
        accuracy = calc_accuracy_loader(
            data_loader, model, device, num_batches = num_batches
        )
    return accuracy

