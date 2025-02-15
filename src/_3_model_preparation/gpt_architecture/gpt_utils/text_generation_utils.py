from ..GPTModel import GPTModel
import tiktoken
import torch

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

def generate_next_tokens(
        model: GPTModel,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        context_size: int,
        temperature=0.0,
        top_k=None,
        eos_id=None):

    batch_size, n_tokens = token_ids.shape
    
    for _ in range(max_new_tokens):
        context_token_ids = token_ids[:, -context_size:] #choose context_size most recent tokens
        with torch.inference_mode():
            logits = model(context_token_ids)
            batch_size, n_tokens, vocab_size = logits.shape
            logits = logits[:, -1, :] #nwl.shape (batch, vocab_size)

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val, 
                    torch.tensor(float("-inf")).to(logits.device),
                    logits
                )

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_word_id = torch.multinomial(probs, num_samples=1)
            else:
                next_word_id = torch.argmax(logits, dim=-1)
                                
            if next_word_id == eos_id:
                break #stops if next word id is end of sequence token
            token_ids = torch.cat((token_ids, next_word_id), dim = 1)

    return token_ids


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
    
