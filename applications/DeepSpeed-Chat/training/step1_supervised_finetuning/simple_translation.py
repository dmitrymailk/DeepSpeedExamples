import os
import torch
import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from easynmt import EasyNMT
from optimum.bettertransformer import BetterTransformer
from datasets import load_from_disk, load_dataset
import os
from tqdm import tqdm
import json



class Translator:
    def __init__(self, model_name: str, device="cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.max_length = 2048
        self.init()

    def init(self):
        print(f"Init model. {self.device}")
        if self.model_name == "facebook/nllb-200-3.3B":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                use_auth_token=True,
            )
            # self.model = BetterTransformer.transform(self.model)
            
            self.model.eval()
            self.model = torch.compile(self.model)
            self.model = self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=True,
            )
        elif self.model_name == "facebook/wmt21-dense-24-wide-en-x":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                use_auth_token=True,
            )
            self.model = BetterTransformer.transform(self.model)
            self.model.half()
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self.model.eval()
            self.model = self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=True,
            )
        elif self.model_name == "opus-mt":
            self.model = EasyNMT(self.model_name)

        print("Model is initialized.")

    def translate(self, text: str):
        func_map = {
            "facebook/nllb-200-3.3B": self.nllb_translate,
            "opus-mt": self.opusmt_translate,
            "facebook/wmt21-dense-24-wide-en-x": self.wmt21_translate,
        }
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            return func_map[self.model_name](text)

    def __call__(self, text: str):
        return self.translate(text=text)

    def nllb_translate(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = self.to_device(inputs=inputs)
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id["rus_Cyrl"],
        )
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[
            0
        ]

    def opusmt_translate(self, text: str):
        return self.model.translate(text, source_lang="en", target_lang="ru")

    def wmt21_translate(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        inputs = self.to_device(inputs=inputs)
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.get_lang_id("ru"),
        )
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[
            0
        ]

    def to_device(self, inputs):
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)
        return inputs


def translate(params):
    device, dataset_subset = params

    return translated_examples


if __name__ == "__main__":
    print("Start translation")
    base_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/"
    save_folder = "dolly_translated"
    full_path= f"{base_path}{save_folder}/"
    file_name = "dolly_translated.json"
    assert os.path.isdir(full_path)
    

    data = load_dataset("databricks/databricks-dolly-15k")
    data = data["train"]
    # data = data.select(range(5))

    fields = ["context", "instruction", "response"]

    model_name = "facebook/wmt21-dense-24-wide-en-x"
    translator = Translator(
        model_name=model_name,
        device="cuda:1",
    )

    translated_examples = []
    for example in tqdm(data):
        for field in fields:
            text = example[field]
            translated = translator(text=text)
            example[f"{field}_translated"] = translated
        translated_examples.append(example)
    
    with open(f"{full_path}{file_name}", 'w', encoding='utf-8') as outfile:
        json.dump(json.dumps(translated_examples, ensure_ascii=False), outfile)
    
    
    
#  0%|          | 1/15014 [00:03<16:22:26,  3.93s/it]
#   0%|          | 2/15014 [00:09<19:44:49,  4.74s/it]
#   0%|          | 3/15014 [00:15<21:58:34,  5.27s/it]
#   0%|          | 4/15014 [00:20<22:27:03,  5.38s/it]
#   0%|          | 5/15014 [00:27<23:52:14,  5.73s/it]
#   0%|          | 6/15014 [00:33<24:42:53,  5.93s/it]
#   0%|          | 7/15014 [00:42<29:44:43,  7.14s/it]
#   0%|          | 8/15014 [00:48<27:36:45,  6.62s/it]
#   0%|          | 9/15014 [00:53<26:03:59,  6.25s/it]
#   0%|          | 10/15014 [01:03<30:43:18,  7.37s/it]
#   0%|          | 11/15014 [01:14<34:26:24,  8.26s/it]
#   0%|          | 12/15014 [01:19<30:59:08,  7.44s/it]
#   0%|          | 13/15014 [01:26<30:06:17,  7.22s/it]
#   0%|          | 14/15014 [01:34<30:48:13,  7.39s/it]
#   0%|          | 15/15014 [01:42<31:23:02,  7.53s/it]
#   0%|          | 16/15014 [01:48<29:40:08,  7.12s/it]
#   0%|          | 17/15014 [01:54<28:21:29,  6.81s/it]
#   0%|          | 18/15014 [02:01<28:57:33,  6.95s/it]
#   0%|          | 19/15014 [02:06<26:39:10,  6.40s/it]
#   0%|          | 20/15014 [02:12<25:45:43,  6.19s/it]
#   0%|          | 21/15014 [02:18<25:14:35,  6.06s/it]
#   0%|          | 22/15014 [02:23<25:00:08,  6.00s/it]
#   0%|          | 23/15014 [02:29<24:26:50,  5.87s/it]
#   0%|          | 24/15014 [02:34<23:45:08,  5.70s/it]
#   0%|          | 25/15014 [02:42<26:27:15,  6.35s/it]
#   0%|          | 26/15014 [02:51<30:03:10,  7.22s/it]
#   0%|          | 27/15014 [02:58<28:48:39,  6.92s/it]
#   0%|          | 28/15014 [03:04<27:29:12,  6.60s/it]
#   0%|          | 29/15014 [03:07<23:08:14,  5.56s/it]
#   0%|          | 30/15014 [03:12<23:12:21,  5.58s/it]
#   0%|          | 31/15014 [03:21<27:15:40,  6.55s/it]
#   0%|          | 32/15014 [03:27<25:56:17,  6.23s/it]
#   0%|          | 33/15014 [03:34<27:23:43,  6.58s/it]
#   0%|          | 34/15014 [03:44<31:39:38,  7.61s/it]
#   0%|          | 35/15014 [03:54<34:48:59,  8.37s/it]
#   0%|          | 36/15014 [04:03<35:30:55,  8.54s/it]
#   0%|          | 37/15014 [04:11<34:47:48,  8.36s/it]
#   0%|          | 38/15014 [04:16<31:00:49,  7.46s/it]
#   0%|          | 39/15014 [04:22<28:47:57,  6.92s/it]
#   0%|          | 40/15014 [04:32<33:08:30,  7.97s/it]
#   0%|          | 41/15014 [04:45<38:21:34,  9.22s/it]
#   0%|          | 42/15014 [04:54<39:09:02,  9.41s/it]
#   0%|          | 43/15014 [05:05<40:36:58,  9.77s/it]
#   0%|          | 44/15014 [05:15<40:58:22,  9.85s/it]
#   0%|          | 45/15014 [05:21<35:49:52,  8.62s/it]
#   0%|          | 46/15014 [05:27<32:37:47,  7.85s/it]
#   0%|          | 47/15014 [05:32<29:36:17,  7.12s/it]
#   0%|          | 48/15014 [05:38<27:43:54,  6.67s/it]
#   0%|          | 49/15014 [05:44<27:04:34,  6.51s/it]
#   0%|          | 50/15014 [05:53<30:00:23,  7.22s/it]
#   0%|          | 51/15014 [05:59<28:58:47,  6.97s/it]
#   0%|          | 52/15014 [06:05<27:11:57,  6.54s/it]
#   0%|          | 53/15014 [06:14<30:28:32,  7.33s/it]
#   0%|          | 54/15014 [06:20<28:11:35,  6.78s/it]
#   0%|          | 55/15014 [06:25<26:33:30,  6.39s/it]
#   0%|          | 56/15014 [06:31<26:27:23,  6.37s/it]
#   0%|          | 57/15014 [06:42<31:24:27,  7.56s/it]
#   0%|          | 58/15014 [06:48<30:02:29,  7.23s/it]
#   0%|          | 59/15014 [06:58<33:40:03,  8.10s/it]
#   0%|          | 60/15014 [07:08<35:18:16,  8.50s/it]
#   0%|          | 61/15014 [07:14<32:05:02,  7.72s/it]
#   0%|          | 62/15014 [07:24<35:18:22,  8.50s/it]
#   0%|          | 63/15014 [07:30<31:58:00,  7.70s/it]
#   0%|          | 64/15014 [07:38<32:22:46,  7.80s/it]
#   0%|          | 65/15014 [07:44<29:49:17,  7.18s/it]
#   0%|          | 66/15014 [07:53<32:00:59,  7.71s/it]
#   0%|          | 67/15014 [07:57<27:51:05,  6.71s/it]
#   0%|          | 68/15014 [08:04<28:52:52,  6.96s/it]
#   0%|          | 69/15014 [08:11<27:58:39,  6.74s/it]
#   0%|          | 70/15014 [08:17<27:24:17,  6.60s/it]
#   0%|          | 71/15014 [08:23<27:18:27,  6.58s/it]
#   0%|          | 72/15014 [08:30<27:20:33,  6.59s/it]
#   0%|          | 73/15014 [08:34<23:47:55,  5.73s/it]
#   0%|          | 74/15014 [08:45<30:20:51,  7.31s/it]
#   0%|          | 75/15014 [08:50<28:15:32,  6.81s/it]
#   1%|          | 76/15014 [08:55<25:12:14,  6.07s/it]
#   1%|          | 77/15014 [09:06<31:49:03,  7.67s/it]
#   1%|          | 78/15014 [09:12<30:03:28,  7.24s/it]
#   1%|          | 79/15014 [09:18<27:58:35,  6.74s/it]
#   1%|          | 80/15014 [09:25<28:38:39,  6.91s/it]
#   1%|          | 81/15014 [09:32<28:12:57,  6.80s/it]
#   1%|          | 82/15014 [09:38<27:41:58,  6.68s/it]
#   1%|          | 83/15014 [09:46<29:07:38,  7.02s/it]
#   1%|          | 84/15014 [09:54<29:50:06,  7.19s/it]
#   1%|          | 85/15014 [10:01<29:45:20,  7.18s/it]
#   1%|          | 86/15014 [10:07<28:26:33,  6.86s/it]
#   1%|          | 87/15014 [10:13<27:19:34,  6.59s/it]
#   1%|          | 88/15014 [10:21<28:59:04,  6.99s/it]
#   1%|          | 89/15014 [10:27<28:18:09,  6.83s/it]
#   1%|          | 90/15014 [10:32<25:08:09,  6.06s/it]
#   1%|          | 91/15014 [10:37<24:48:49,  5.99s/it]
#   1%|          | 92/15014 [10:44<25:43:17,  6.21s/it]
#   1%|          | 93/15014 [10:50<25:56:06,  6.26s/it]
#   1%|          | 94/15014 [10:57<26:37:13,  6.42s/it]
#   1%|          | 95/15014 [11:02<24:50:28,  5.99s/it]
#   1%|          | 96/15014 [11:08<24:09:49,  5.83s/it]
#   1%|          | 97/15014 [11:19<30:48:01,  7.43s/it]
#   1%|          | 98/15014 [11:29<34:44:08,  8.38s/it]
#   1%|          | 99/15014 [11:35<31:33:26,  7.62s/it]
#   1%|          | 100/15014 [11:45<34:38:35,  8.36s/it]
#   1%|          | 101/15014 [11:52<32:09:09,  7.76s/it]
#   1%|          | 102/15014 [11:58<30:00:06,  7.24s/it]
#   1%|          | 103/15014 [12:08<33:44:57,  8.15s/it]
#   1%|          | 104/15014 [12:15<32:38:37,  7.88s/it]
#   1%|          | 105/15014 [12:22<31:26:17,  7.59s/it]
#   1%|          | 106/15014 [12:27<27:21:05,  6.60s/it]
#   1%|          | 107/15014 [12:31<24:28:08,  5.91s/it]
#   1%|          | 108/15014 [12:38<26:11:00,  6.32s/it]
#   1%|          | 109/15014 [12:43<23:59:29,  5.79s/it]
#   1%|          | 110/15014 [12:49<24:42:24,  5.97s/it]
#   1%|          | 111/15014 [12:55<24:07:07,  5.83s/it]
#   1%|          | 112/15014 [13:01<24:46:09,  5.98s/it]
#   1%|          | 113/15014 [13:11<29:23:54,  7.10s/it]
#   1%|          | 114/15014 [13:21<33:30:40,  8.10s/it]
#   1%|          | 115/15014 [13:32<36:33:30,  8.83s/it]
#   1%|          | 116/15014 [13:41<37:31:17,  9.07s/it]
#   1%|          | 117/15014 [13:47<33:12:35,  8.03s/it]
#   1%|          | 118/15014 [13:57<36:01:33,  8.71s/it]
#   1%|          | 119/15014 [14:01<30:26:34,  7.36s/it]
#   1%|          | 120/15014 [14:09<31:23:44,  7.59s/it]
#   1%|          | 121/15014 [14:14<27:57:06,  6.76s/it]
#   1%|          | 122/15014 [14:22<29:28:16,  7.12s/it]
#   1%|          | 123/15014 [14:28<27:59:50,  6.77s/it]
#   1%|          | 124/15014 [14:38<32:23:01,  7.83s/it]
#   1%|          | 125/15014 [14:43<28:25:58,  6.87s/it]
#   1%|          | 126/15014 [14:49<27:04:59,  6.55s/it]
#   1%|          | 127/15014 [14:57<28:57:08,  7.00s/it]
#   1%|          | 128/15014 [15:03<27:43:35,  6.71s/it]
#   1%|          | 129/15014 [15:09<26:59:09,  6.53s/it]
#   1%|          | 130/15014 [15:15<25:42:05,  6.22s/it]
#   1%|          | 131/15014 [15:21<25:29:36,  6.17s/it]
#   1%|          | 132/15014 [15:29<28:03:00,  6.79s/it]
#   1%|          | 133/15014 [15:35<27:18:36,  6.61s/it]
#   1%|          | 134/15014 [15:43<28:39:34,  6.93s/it]
#   1%|          | 135/15014 [15:52<31:54:23,  7.72s/it]
#   1%|          | 136/15014 [16:00<31:22:24,  7.59s/it]
#   1%|          | 137/15014 [16:05<29:15:23,  7.08s/it]
#   1%|          | 138/15014 [16:11<27:45:43,  6.72s/it]
#   1%|          | 139/15014 [16:17<26:36:12,  6.44s/it]
#   1%|          | 140/15014 [16:26<29:27:57,  7.13s/it]
#   1%|          | 141/15014 [16:36<33:24:21,  8.09s/it]
#   1%|          | 142/15014 [16:43<32:18:51,  7.82s/it]
#   1%|          | 143/15014 [16:50<30:39:16,  7.42s/it]
#   1%|          | 144/15014 [16:56<29:32:14,  7.15s/it]
#   1%|          | 145/15014 [17:04<29:55:05,  7.24s/it]
#   1%|          | 146/15014 [17:11<30:02:03,  7.27s/it]
#   1%|          | 147/15014 [17:20<31:33:21,  7.64s/it]
#   1%|          | 148/15014 [17:25<29:04:50,  7.04s/it]
#   1%|          | 149/15014 [17:33<29:16:35,  7.09s/it]
#   1%|          | 150/15014 [17:39<28:26:55,  6.89s/it]
#   1%|          | 151/15014 [17:49<32:23:06,  7.84s/it]
#   1%|          | 152/15014 [17:56<31:00:26,  7.51s/it]
#   1%|          | 153/15014 [18:02<28:52:52,  7.00s/it]
#   1%|          | 154/15014 [18:07<27:23:38,  6.64s/it]
#   1%|          | 155/15014 [18:14<27:35:19,  6.68s/it]
#   1%|          | 156/15014 [18:21<28:06:51,  6.81s/it]
#   1%|          | 157/15014 [18:26<25:35:26,  6.20s/it]
#   1%|          | 158/15014 [18:32<25:12:02,  6.11s/it]
#   1%|          | 159/15014 [18:41<28:25:25,  6.89s/it]
#   1%|          | 160/15014 [18:46<26:31:39,  6.43s/it]
#   1%|          | 161/15014 [18:57<31:43:24,  7.69s/it]
#   1%|          | 162/15014 [19:04<31:37:15,  7.66s/it]
#   1%|          | 163/15014 [19:11<29:55:18,  7.25s/it]
#   1%|          | 164/15014 [19:18<29:52:32,  7.24s/it]
#   1%|          | 165/15014 [19:25<30:05:00,  7.29s/it]
#   1%|          | 166/15014 [19:32<29:48:55,  7.23s/it]
#   1%|          | 167/15014 [19:39<29:27:39,  7.14s/it]
#   1%|          | 168/15014 [19:46<28:33:03,  6.92s/it]
#   1%|          | 169/15014 [19:55<31:05:26,  7.54s/it]
#   1%|          | 170/15014 [20:04<33:31:23,  8.13s/it]
#   1%|          | 171/15014 [20:14<36:21:23,  8.82s/it]
#   1%|          | 172/15014 [20:19<31:36:24,  7.67s/it]
#   1%|          | 173/15014 [20:29<33:57:30,  8.24s/it]
#   1%|          | 174/15014 [20:39<36:30:03,  8.85s/it]
#   1%|          | 175/15014 [20:45<32:29:00,  7.88s/it]
#   1%|          | 176/15014 [20:55<35:47:03,  8.68s/it]
#   1%|          | 177/15014 [21:01<32:01:15,  7.77s/it]
#   1%|          | 178/15014 [21:08<30:54:09,  7.50s/it]
#   1%|          | 179/15014 [21:15<30:50:37,  7.48s/it]
#   1%|          | 180/15014 [21:19<25:24:39,  6.17s/it]
#   1%|          | 181/15014 [21:24<24:24:19,  5.92s/it]
#   1%|          | 182/15014 [21:30<25:08:27,  6.10s/it]
#   1%|          | 183/15014 [21:37<26:19:47,  6.39s/it]
#   1%|          | 184/15014 [21:43<25:50:15,  6.27s/it]
#   1%|          | 185/15014 [21:51<27:08:09,  6.59s/it]
#   1%|          | 186/15014 [21:56<25:40:18,  6.23s/it]
#   1%|          | 187/15014 [22:04<27:40:21,  6.72s/it]
#   1%|▏         | 188/15014 [22:14<31:49:52,  7.73s/it]
#   1%|▏         | 189/15014 [22:22<32:16:09,  7.84s/it]
#   1%|▏         | 190/15014 [22:28<29:30:34,  7.17s/it]
#   1%|▏         | 191/15014 [22:35<29:24:04,  7.14s/it]
#   1%|▏         | 192/15014 [22:39<25:40:26,  6.24s/it]
#   1%|▏         | 193/15014 [22:45<24:47:36,  6.02s/it]
#   1%|▏         | 194/15014 [22:51<25:19:21,  6.15s/it]
#   1%|▏         | 195/15014 [22:59<27:23:29,  6.65s/it]
#   1%|▏         | 196/15014 [23:06<27:45:54,  6.75s/it]
#   1%|▏         | 197/15014 [23:13<28:19:31,  6.88s/it]
