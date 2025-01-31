# GPT Implementation in PyTorch

A PyTorch implementation of the GPT (Generative Pre-trained Transformer) architecture, featuring both training capabilities and text generation. This implementation includes the core components of the GPT architecture with configurable parameters and training options.

## 🚀 Features

- Complete GPT architecture implementation
- Support for loading pretrained GPT-2 weights
- Configurable model parameters (layers, heads, embeddings)
- Custom lightweight data loader for efficient text processing
- Learning rate scheduling with CosineAnnealing
- Gradient clipping for stable training
- Model checkpointing (saves best model)
- Multi-device support (CPU, CUDA, MPS)

## 📋 Requirements

```bash
torch>=2.0.0
transformers>=4.30.0
tiktoken>=0.5.0
```

## 🏗️ Model Architecture

- **Embedding Layer**: Token and position embeddings
- **Transformer Blocks**: Multiple layers of:
  - Multi-head self-attention
  - Layer normalization
  - Feed-forward neural network
- **Output Layer**: Linear layer for token prediction

### Default Configuration
- Block size: 1024 (max sequence length)
- Vocabulary size: 50,257 (GPT-2 tokenizer)
- Number of layers: 12
- Number of attention heads: 12
- Embedding dimension: 768

## 💾 Training Setup

### Hyperparameters
- Batch size: 4
- Sequence length: 32
- Initial learning rate: 6e-4
- Weight decay: 0.1
- Number of epochs: 25
- Learning rate scheduling: CosineAnnealing
- Gradient clipping: 1.0

### Training Process
1. Data is loaded and tokenized using the GPT-2 tokenizer
2. Model processes text in chunks of size (batch_size × sequence_length)
3. Learning rate is adjusted using CosineAnnealing scheduler
4. Best model is saved based on average epoch loss

## 📊 Usage

1. **Prepare Data**:
   Place your training text in input.txt

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Training**:
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
.
├── main.py           # Main implementation file
├── input.txt         # Training data
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

## 🔍 Model Components

### CausalSelfAttention
- Implements masked self-attention mechanism
- Supports multiple attention heads
- Includes output projection

### MLP (Multi-Layer Perceptron)
- Implements feed-forward network
- Uses GELU activation
- Includes input and output projections

### Block
- Combines attention and MLP layers
- Implements residual connections
- Uses layer normalization

### DataLoaderLite
- Efficient data loading
- Handles token generation
- Implements batch creation

## 🔧 Configuration Options

The model can be configured through the GPTConfig class:
```python
GPTConfig(
    block_size=1024,    # Maximum sequence length
    vocab_size=50257,   # Vocabulary size
    n_layer=12,         # Number of transformer layers
    n_head=12,          # Number of attention heads
    n_embd=768         # Embedding dimension
)
```

## 💡 Training Monitoring

The training process outputs:
- Per-batch loss values
- Current learning rate
- Average epoch loss
- Best model checkpoints

## 🛠️ Advanced Features

1. **Model Checkpointing**:
   - Saves best performing model
   - Includes optimizer state
   - Tracks training progress

2. **Learning Rate Scheduling**:
   - Cosine annealing
   - Configurable minimum learning rate
   - Smooth learning rate decay

3. **Gradient Handling**:
   - Gradient clipping
   - Weight decay
   - Optimizer state management

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.