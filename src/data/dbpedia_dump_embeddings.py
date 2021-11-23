from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import torch
import pickle

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def calculateEmbeddings(sentences,tokenizer,model):
    tokenized_sentences = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**tokenized_sentences)
    sentence_embeddings = mean_pooling(model_output, tokenized_sentences['attention_mask'])
    return sentence_embeddings


def saveToDisc(embeddings, filename):
    with open(filename, "ab") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

def saveToDisc(sentences, embeddings, filename):
    with open(filename, "ab") as f:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings}, f, protocol=pickle.HIGHEST_PROTOCOL)

dt = datetime.now()
datetime_formatted = dt.strftime('%Y-%m-%d_%H:%M:%S')
batch_size = 1000

input_text_file = 'data/preprocessed/shortened_abstracts_hu_2021_09_01.txt'
output_embeddings_file = f'data/preprocessed/embeddings_{batch_size}_batches_at_{datetime_formatted}.pkl'

multilingual_checkpoint = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
tokenizer = AutoTokenizer.from_pretrained(multilingual_checkpoint)
model = AutoModel.from_pretrained(multilingual_checkpoint)


total_read = 0
total_read_limit = 3 * batch_size
with open(input_text_file) as f:
    while total_read < total_read_limit:
        count = 0
        sentences = []
        line = 'init'
        while line and count < batch_size:
            line = f.readline()
            sentences.append(line)
            count += 1
        
        sentence_embeddings = calculateEmbeddings(sentences,tokenizer,model)
        saveToDisc(sentences, sentence_embeddings,output_embeddings_file)
        total_read += count