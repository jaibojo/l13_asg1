# Solving for residual std scaling issue
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml
import tiktoken
from datasets import load_dataset

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = torch.cuda.is_available()  # Only True if we have both the package and CUDA
    if not FLASH_ATTENTION_AVAILABLE:
        print("Flash Attention requires CUDA. Using regular attention.")
except ImportError:
    print("Flash Attention not available. Using regular attention.")
    FLASH_ATTENTION_AVAILABLE = False


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 2048          
    vocab_size: int = 49152         
    n_layer: int = 30               
    n_head: int = 9                 
    n_embd: int = 576              
    intermediate_size: int = 1536   
    n_kv_heads: int = 3            
    hidden_act: str = 'silu'       
    rms_norm_eps: float = 1e-5     
    rope_theta: float = 10000.0    


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        # Create inverse frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim

    def forward(self, x, seq_len):
        # Create position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        # Calculate freqs
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Create complex numbers in real space
        emb = torch.cat((freqs, freqs), dim=-1)
        # Match dimensions with input
        emb = emb.view(1, seq_len, 1, self.dim)  # [1, seq_len, 1, dim]
        return emb

class LlamaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaSdpaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_heads
        self.head_dim = config.n_embd // config.n_head
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        
        # Project q, k, v
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # Get rotary embeddings
        rot_emb = self.rotary_emb(x, T)
        
        # Apply rotary embeddings
        q = (q * torch.cos(rot_emb)) + (self.rotate_half(q) * torch.sin(rot_emb))
        k = (k * torch.cos(rot_emb)) + (self.rotate_half(k) * torch.sin(rot_emb))
        
        # Repeat k,v heads to match number of q heads
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        
        # Reshape for attention
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)
        
        # Regular attention for CPU, Flash Attention for CUDA
        if FLASH_ATTENTION_AVAILABLE and x.is_cuda:
            output = flash_attn_func(q, k, v, causal=True)
        else:
            # Regular attention with memory efficient implementation
            scale = 1.0 / math.sqrt(self.head_dim)
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device), 
                diagonal=1
            )
            scores.masked_fill_(causal_mask, float('-inf'))
            
            # Compute attention probabilities
            attn_probs = F.softmax(scores, dim=-1)
            
            # Apply attention to values
            output = torch.matmul(attn_probs, v)
        
        # Reshape output and project
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(output)
    
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.self_attn = LlamaSdpaAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x):
        # Pre-norm architecture
        h = x + self.self_attn(self.input_layernorm(x))
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=2)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = LlamaRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config.n_embd // config.n_head)
        
        # Remove position embeddings as we're using rotary embeddings
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        # Get token embeddings
        x = self.embed_tokens(idx)
        
        # Forward through transformer blocks
        for block in self.layers:
            x = block(x)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# model = GPT.from_pretrained('gpt2')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print(f"Using device: {device}")

if device == 'cpu':
    print("Warning: Running on CPU. This will be very slow!")
    if not torch.cuda.is_available():
        print("CUDA not available. Consider using a GPU for better performance.")

# SEED
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# STOP
num_return_sequences = 5
max_length = 30



import tiktoken

class StreamingDataLoader:
    def __init__(self, config):
        self.batch_size = config['training']['batch_size']
        self.sequence_length = config['training']['sequence_length']
        self.tokenizer = tiktoken.get_encoding(config['data']['tokenizer'])
        self.buffer_size = config['data']['buffer_size']
        self.buffer = []
        
        # Load dataset from HuggingFace
        self.dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "cosmopedia-v2",
            split="train",
            streaming=True
        )
        self.data_iterator = iter(self.dataset)
        
    def fill_buffer(self):
        """Fill the token buffer from dataset"""
        while len(self.buffer) < self.buffer_size:
            try:
                # Get next example from dataset
                example = next(self.data_iterator)
                # Get text from example
                text = example['text']
                # Tokenize text
                tokens = self.tokenizer.encode(text)
                self.buffer.extend(tokens)
            except StopIteration:
                # Reset iterator when we reach the end
                self.data_iterator = iter(self.dataset)
            except Exception as e:
                print(f"Error processing example: {e}")
                continue
    
    def next_batch(self):
        """Get next batch of tokens"""
        # Fill buffer if needed
        if len(self.buffer) < (self.batch_size * self.sequence_length + 1):
            self.fill_buffer()
        
        # Get batch from buffer
        tokens = self.buffer[:(self.batch_size * self.sequence_length + 1)]
        self.buffer = self.buffer[(self.batch_size * self.sequence_length):]
        
        # Ensure tokens are within vocabulary range
        vocab_size = config['model']['model_config']['vocab_size']
        tokens = [min(t, vocab_size-1) for t in tokens]  # Clip tokens to vocab size
        
        # Reshape into batch
        x = torch.tensor(tokens[:-1]).view(self.batch_size, self.sequence_length)
        y = torch.tensor(tokens[1:]).view(self.batch_size, self.sequence_length)
        
        return x, y


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load configuration
try:
    config = load_config()
except FileNotFoundError:
    # Provide default configuration if file doesn't exist
    config = {
        'training': {
            'batch_size': 8,
            'sequence_length': 2048,
        },
        'data': {
            'tokenizer': 'gpt2',
            'buffer_size': 100000,
        },
        'model': {
            'model_config': {
                'block_size': 2048,
                'vocab_size': 49152,
                'n_layer': 30,
                'n_head': 9,
                'n_embd': 576,
                'intermediate_size': 1536,
                'n_kv_heads': 3,
                'hidden_act': 'silu',
                'rms_norm_eps': 1e-5,
                'rope_theta': 10000.0
            }
        },
        'optimizer': {
            'gradient_clip': 1.0
        },
        'logging': {
            'log_interval': 100
        },
        'checkpointing': {
            'checkpoint_dir': 'checkpoints',
            'save_interval': 1000
        }
    }
    # Save default config
    os.makedirs('checkpoints', exist_ok=True)
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)
    print("Created default config.yaml")

def validate_model_config(config_dict):
    """Validate that all required model config parameters are present"""
    required_params = {
        'block_size', 'vocab_size', 'n_layer', 'n_head', 'n_embd',
        'intermediate_size', 'n_kv_heads', 'hidden_act', 'rms_norm_eps',
        'rope_theta'
    }
    
    model_config = config_dict['model']['model_config']
    missing_params = required_params - set(model_config.keys())
    extra_params = set(model_config.keys()) - required_params
    
    if missing_params:
        raise ValueError(f"Missing required parameters in model config: {missing_params}")
    if extra_params:
        print(f"Warning: Extra parameters in model config will be ignored: {extra_params}")
        # Remove extra parameters
        for param in extra_params:
            del model_config[param]
    
    return config_dict

# Add before creating the model:
config = validate_model_config(config)
model = GPT(GPTConfig(**config['model']['model_config']))
model.to(device)

train_loader = StreamingDataLoader(config)

# NEW CODE
# Define training parameters
num_epochs = 25  # Number of epochs
batches_per_epoch = len(train_loader.buffer) // (train_loader.batch_size * train_loader.sequence_length)

# Training hyperparameters
batch_size = 8                     # Changed from 4 to 8
accumulation_steps = 2             # New parameter
sequence_length = 2048             # Changed from 32 to 2048
initial_lr = 3e-3                  # Changed from 6e-4 to 3e-3
weight_decay = 0.01                # Changed from 0.1 to 0.01
num_training_steps = 600000        # New parameter

# Optimizer changes
try:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
        fused=True
    )
except RuntimeError:
    # Fallback to non-fused implementation
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay
    )

# Learning rate scheduler
def get_lr(step):
    # Implement warmup and decay according to YAML
    warmup_steps = 5000
    decay_start = 500000
    decay_steps = 100000
    if step < warmup_steps:
        return initial_lr * (step / warmup_steps)
    elif step < decay_start:
        return initial_lr
    else:
        decay_ratio = (step - decay_start) / decay_steps
        decay_factor = 1.0 / math.sqrt(1.0 + decay_ratio)
        return max(initial_lr * decay_factor, 0)

print(f"Starting training for {num_epochs} epochs, {batches_per_epoch} batches per epoch")

# Create checkpoint directory if it doesn't exist
checkpoint_dir = config['checkpointing']['checkpoint_dir']
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, batch, loss, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config  # Save configuration with checkpoint
    }
    
    # Save regular checkpoint
    if not is_best:
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f'checkpoint_epoch{epoch}_batch{batch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Remove old checkpoints to save space (keep last 3)
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) 
                            if f.startswith('checkpoint_')])
        for old_ckpt in checkpoints[:-3]:  # Keep only last 3 checkpoints
            os.remove(os.path.join(checkpoint_dir, old_ckpt))
    
    # Save best model separately
    if is_best:
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_model_path)
        print(f"Saved best model with loss: {loss:.4f}")

def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
    
    checkpoint = torch.load(checkpoint_path)
    model_config = GPTConfig(**checkpoint['config']['model']['model_config'])
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['batch'], checkpoint['loss']

# Add before training loop
resume_checkpoint = config['checkpointing'].get('resume_checkpoint_path', None)
if resume_checkpoint:
    loaded = load_checkpoint(resume_checkpoint)
    if loaded is not None:
        model, optimizer, start_epoch, _, best_loss = loaded
        print(f"Resumed training from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_loss = float('inf')
else:
    start_epoch = 0
    best_loss = float('inf')

# Update training loop to handle streaming
global_step = 0
best_loss = float('inf')
total_steps = config['training']['num_training_steps']

try:
    total_tokens = 0
    total_time = 0
    losses = []
    
    for i in range(50):  # Using 50 iterations as in your original code
        try:
            t0 = time.time()
            
            x, y = train_loader.next_batch()
            
            # Validate input tokens are within vocabulary range
            if x.max() >= config['model']['model_config']['vocab_size']:
                print(f"Warning: Input contains token {x.max()} which is >= vocab_size {config['model']['model_config']['vocab_size']}")
                continue
                
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            
            loss.backward()
            optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            t1 = time.time()
            dt = (t1 - t0) * 1000  # Convert to milliseconds
            
            # Calculate tokens per second
            batch_tokens = x.numel()  # Number of tokens in this batch
            tokens_per_sec = batch_tokens / (t1 - t0)
            
            total_tokens += batch_tokens
            total_time += (t1 - t0)
            losses.append(loss.item())
            
            print(f'step[{i:3d}] | loss: {loss.item():6.3f} | dt: {dt:7.2f}ms | tok/sec: {tokens_per_sec:10.2f}')
            
        except Exception as e:
            print(f"Error in training step: {e}")
            continue
    
    # Print summary only if we have losses
    if losses:
        avg_loss = sum(losses) / len(losses)
        print(f"\nTraining Summary:")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Total tokens processed: {total_tokens:,}")
        print(f"Total time: {total_time:.2f} seconds")
        if total_time > 0:
            print(f"Average tokens/sec: {total_tokens/total_time:,.2f}")
    else:
        print("No successful training steps completed")

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"\nTraining failed with error: {e}")
finally:
    if 'loss' in locals() and losses:
        print(f'Final loss: {losses[-1]:.4f}')  # Use the last loss from losses list
    else:
        print("No loss value available")

# Remove the print(loss) statement after the training loop
# import sys; sys.exit(0)  # Keep this if you want to exit

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)[0] # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

# Add validation loop to monitor overfitting
# Add early stopping
# Add tensorboard logging

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps)
        return x * self.weight

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Implementation of grouped-query attention
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)

def validate_config(config):
    required_keys = ['training', 'data', 'model', 'optimizer', 'logging', 'checkpointing']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    if config['model']['model_config']['n_head'] % config['model']['model_config']['n_kv_heads'] != 0:
        raise ValueError("n_head must be divisible by n_kv_heads")

# Add after loading config:
validate_config(config)

def check_gpu_memory():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory < 8 * 1024 * 1024 * 1024:  # 8GB
            print("Warning: GPU has less than 8GB memory. You may encounter OOM errors.")

check_gpu_memory()