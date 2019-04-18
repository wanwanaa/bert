import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# Tokenized input
text = "自然语言处理 <S> "
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)


# Convert token to vocabulary indices
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# print(indexed_tokens)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 0, 0]
# segments_ids = [1, 1, 1, 1, 1, 1, 1]
# Convert inputs to PyTorch tensors
# tokens_tensor = torch.tensor([indexed_tokens])
# segments_tensors = torch.tensor([segments_ids])
#
# model = BertModel.from_pretrained('bert-base-uncased')
# model.eval()
# with torch.no_grad():
#     encoded_layers, _ = model(tokens_tensor, segments_tensors)
# print(encoded_layers[-1][0][-2])

# word = tokenizer.convert_ids_to_tokens([104, 105])
# print(word)