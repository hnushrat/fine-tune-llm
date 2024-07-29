import torch

from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

class build_model:
    def __init__(self, name = None, pad_token = None):
        '''
        name: model card from huggingface
        pad_token: (str) the pad token

        returns: peft model
        '''
        self.MODEL_NAME = name       

        quant_conf = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_quant_type = 'nf4', bnb_4bit_compute_dtype = torch.float16)

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, use_fast = True)
        tokenizer.add_special_tokens({"pad_token" : pad_token})
        tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME,
                                                    quantization_config = quant_conf,
                                                    device_map = "auto")
        self.model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of = 8)

        print(self.model.config)
    
    def get_lora_model(self, rank = 64, alpha = 16, dropout = 0.2):
        
        lora_config = LoraConfig(
            r = rank,
            lora_alpha = alpha,
            lora_dropout = dropout,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj"
                ],
            bias="none",
            task_type=TaskType.CAUSAL_LM,)

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)

        print(self.model.print_trainable_parameters())
        return self.model
     