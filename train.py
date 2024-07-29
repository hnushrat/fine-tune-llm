from transformers import AutoTokenizer

from trl import SFTConfig, SFTTrainer

class trainer:
    def __init__(self, seed = None, name = None, max_seq_length = 512, model = None, dataset = None, pad_token = None):
        '''
        seed: (int) seed to set
        name: model card from huggingface
        max_seq_length: (int)
        model: the peft model
        dataset: the dataset
        pad_token: the pad token
        '''
        self.SEED = seed
        self.OUTPUT_DIR = 'outputs'

        self.model = model
        self.dataset = dataset
        
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast = True)
        self.tokenizer.add_special_tokens({'pad_token' : pad_token})
        self.tokenizer.padding_side = "right"

        self.max_seq_length = max_seq_length
        
        self.sft_config = SFTConfig(output_dir = self.OUTPUT_DIR,
                                dataset_text_field = "text",
                                max_seq_length = self.max_seq_length,
                                num_train_epochs = 1,
                                per_device_train_batch_size = 2,
                                per_device_eval_batch_size = 2,
                                gradient_accumulation_steps = 4,
                                optim = "paged_adamw_8bit",
                                eval_strategy = "steps",
                                eval_steps = 0.2,
                                save_steps = 0.2,
                                logging_steps = 10,
                                learning_rate = 1e-4,
                                fp16 = True,  # or bf16 = True,
                                save_strategy = "steps",
                                warmup_ratio = 0.1,
                                save_total_limit = 2,
                                lr_scheduler_type = "constant",
                                report_to = "tensorboard",
                                save_safetensors = True,
                                dataset_kwargs = {
                                "add_special_tokens": False,  # We template with special tokens
                                "append_concat_token": False,  # No need to add additional separator token
                                },
                                seed = self.SEED,
                                )

    def train(self, saved_name = None):
        trainer = SFTTrainer(
        model = self.model,
        args = self.sft_config,
        train_dataset = self.dataset["train"],
        eval_dataset = self.dataset["val"],
        tokenizer = self.tokenizer,
        #data_collator = self.collator,
        )
        trainer.train()
        trainer.save_model(saved_name)

