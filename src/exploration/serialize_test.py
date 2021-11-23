import torch
import pickle
'''
a = [1,2,3]
b = [4,5,6]
at = torch.tensor([a,a])
bt = torch.tensor([b,b])

with open('serialize_test.pkl', "ab") as f:
    pickle.dump(at,f)
    pickle.dump(bt,f)

with open('serialize_test.pkl', "rb") as f:
    print(pickle.load(f))
    print(pickle.load(f))
'''

def loadFromDiskRaw(batch_number, filename='embeddings.pkl'):
    count = 0
    with open(filename, "rb") as f:
        while count < batch_number:
            stored_data = pickle.load(f)
            print(stored_data.size())
            print(stored_data[0][:15])
            count += 1
    return stored_data

output_embeddings_file = 'data/processed/DBpedia_shortened_abstracts_hu_embeddings.pkl'
loadFromDiskRaw(3, output_embeddings_file)

