"""Module loading the AlpacaInstructPromptTokenizingStrategy class"""

from training.prompt_tokenizers import AlpacaPromptTokenizingStrategy
from training.prompters import AlpacaPrompter, PromptStyle


def load(tokenizer, cfg):
    return AlpacaPromptTokenizingStrategy(
        AlpacaPrompter(PromptStyle.INSTRUCT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
