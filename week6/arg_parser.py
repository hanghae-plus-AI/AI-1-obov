from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default=None)


parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()
print("args", args)
print("training_args", training_args)
