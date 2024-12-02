import random

import torch

from utils.prompter import Prompter


class DPODataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.args = args
        self.prompter = Prompter(args, args.prompt)
        self.print_result = True

    def tokenize(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                # and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = data_point['instruction']
        tokenized_full_prompt = self.tokenize(full_prompt)
        return tokenized_full_prompt

    def prompting(self, data, mode='train'):
        full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], mode=mode)
        return full_prompt

    def clean(self, input_sentence):
        return input_sentence.replace('\xa0', ' ').replace('  ', ' ').strip()

    def __getitem__(self, idx):
        data = self.dataset[idx]

        prompt = self.clean(self.prompting(data['dialog']))
        chosen = self.clean(data['chosen'] + self.tokenizer)
        rejected = self.clean(data['rejected'] + self.tokenizer)

        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    def __len__(self):
        return len(self.dataset)
