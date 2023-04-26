import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk, load_dataset

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-3.3B",
    use_auth_token=True,
    # src_lang="ron_Latn"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-3.3B", use_auth_token=True,
)

item = "Hello World"
inputs = tokenizer(item, return_tensors="pt")

translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["rus_Cyrl"], max_length=30
)
tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

fields_map = {
    "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/databricks_dolly_15k_translated_fixed": {
        "read_func": load_from_disk,
        "translate_fields": ["context", "instruction", "response"],
    },
    "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/self_instruct_translated": {
        "read_func": load_from_disk,
        "translate_fields": [
            "completion",
            "prompt",
        ],
    },
    "Anthropic/hh-rlhf": {
        "read_func": load_dataset,
        "translate_fields": [
            "prompt",
            "response",
            "chosen",
            "rejected",
        ],
    },
}


with torch.no_grad():
    for i, example in enumerate(data["train"].shuffle()):
        fields = ["context", "instruction", "response"]
        for field in fields:
            print(f"Field name: {field}")
            print("Original: ", example[field])
            item = example[field]
            inputs = tokenizer(item, return_tensors="pt")
            for key in inputs.keys():
                inputs[key] = inputs[key].to("cuda")
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id["rus_Cyrl"],
                max_length=1024,
            )
            translated = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]
            print("Translated: ", translated)
            print(f"Previous translated: {example[f'{field}_translated']}")
            print()
        print("==" * 100)

        if i > 100:
            break
