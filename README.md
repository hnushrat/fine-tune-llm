# Fine Tune Llama 3.1 8B Instruct

This codebase can fine-tune *Llama 3.1 8B Instruct* on your custom dataset.
This implementation is on the HuggingFace dataset, but you can get the idea about preprocessing your data.

## Usage

### Prerequisites

The code is built with the following libraries:

- Python >= 3.8
- [PyTorch](https://github.com/pytorch/pytorch) = 2.0.1
- [tqdm](https://github.com/tqdm/tqdm)
- [trl] = 0.9.4
- [transformers] = 4.41.0
- [bitsandbytes] = 0.43.1
- [peft] = 0.11.1
