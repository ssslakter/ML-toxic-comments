import torch


class Processor:
    def __init__(self, labels, tokenizer, device: torch.device) -> None:
        self.labels = labels
        self.tokenizer = tokenizer
        self.device = device

    def extract_target(self, df):
        return torch.Tensor(df[self.labels], device=self.device)

    def tokenize_data(self, df) -> torch.Tensor:
        return (self.tokenizer(list(df["comment_text"]),
                               padding="max_length", truncation=True, return_tensors='pt',
                               return_attention_mask=False)['input_ids']).to(self.device)
