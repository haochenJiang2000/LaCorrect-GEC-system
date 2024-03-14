from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BertTokenizer, DataCollatorForSeq2Seq
import torch

from data_loader import MyDataset

model_dir = "real_learner_bart_CGEC"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BartForConditionalGeneration.from_pretrained(model_dir)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
model.eval()

inputs = ['只 是 当 时 哦 达 瓦 和 第 哦 啊 为 活 动 撒 达 瓦 达 瓦', '大 哇 撒 大 娃 娃 大 啊 撒 大 网 的 速 度 打 完', '大 苏 打 大 大 哇 大 撒 大 撒 达 娃 大 撒']

# 加载数据
predict_dataset = MyDataset(inputs, tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
predict_dataloader = DataLoader(dataset=predict_dataset, collate_fn=data_collator, batch_size=4)

# input_ids = torch.tensor(tokenizer.encode(input, return_tensors='pt', add_special_tokens=False), device=device)
for batch in predict_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    generated_tokens = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], num_beams=3, max_length=100)
    generated_tokens = generated_tokens.cpu().numpy()
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    predictions.append(cc.convert("".join(result).replace("##", "")))