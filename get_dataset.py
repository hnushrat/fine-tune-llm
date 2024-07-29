import pandas as pd

from textwrap import dedent

from datasets import load_dataset

from transformers import AutoTokenizer

class prepare_data:
    '''
    name: the dataset card from huggingface
    tokenizer: token card from huggingface
    
    returns: dataframe with Llama 3 template for the texts
    '''
    def __init__(self, name = None, tokenizer = None):
        
        self.dataset = load_dataset(name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast = True)

        # Convert to dataframe
        
        q,c,a = [],[],[] # question, context, answer
        for i in self.dataset['train']: # dictionary
            q.append(i['question'])
            c.append(i['context'])
            a.append(i['answer'])
        self.df = pd.DataFrame({'question' : q, 'context' : c, 'answer' : a})

    # format samples
    def format_samples(self, sample):
        prompt = dedent(
            f"""
                {sample['question']}
                
                Information :            
                {sample['context']}
            """
                        )
        messages = [
            {
                'role' : "system",
                'content' : "Use given info to answer",
            },
            {
                'role' : 'user', 
                'content' : prompt

            },
            {
                'role' : 'assistant', # the model
                'content' : sample['answer']
            }        
                    ]
        return self.tokenizer.apply_chat_template(messages, tokenize = False)

    def calc_length(self, sample):
        return len(self.tokenizer(sample['text'], add_special_tokens = True, return_attention_mask = False)['input_ids'])
    
    def get_data(self):
        self.df['text'] = self.df.apply(self.format_samples, axis = 1)
        self.df['total_tokens'] = self.df.apply(self.calc_length, axis = 1)

        # find samples with total tokens more than 512
        print(f"Total samples with token size more than 512 -> {self.df[self.df['total_tokens']>512].shape[0]}")

        self.df = self.df[self.df['total_tokens']<=512]

        return self.df

    
