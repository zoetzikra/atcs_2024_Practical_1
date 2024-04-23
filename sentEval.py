# import senteval
# from dataset import build_voc
import numpy as np
import torch
import json

device_found = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_samples(params, samples):
    
    params.w2i = params.vocabulary.stoi
    
    return


def get_batch(params, batch):
    batch = [sent if len(sent) > 0 else ["."] for sent in batch]
    sentence_ids = []
    for sent in batch:
        sentence_ids.append(torch.tensor([params.w2i.get(token, 0) for token in sent]))
        
    lengths = torch.tensor([len(s) for s in sentence_ids])
    
    padded_sentences = torch.nn.utils.rnn.pad_sequence(
        sentence_ids, batch_first=True, padding_value=1
    )
    
    embeddings = (params.encoder.forward(padded_sentences.to(device_found), lengths).cpu().detach().numpy())
    
    return embeddings


def sentEval(vocab, model, data_dir, batch_size, path=None):
    params_senteval = {"task_path": data_dir, "usepytorch": True, "kfold": 5}
    params_senteval["classifier"] = {
        "nhid": 0,
        "optim": "rmsprop",
        "batch_size": batch_size,
        "tenacity": 3,
        "epoch_size": 2,
    }
    params_senteval["vocabulary"] = vocab
    
    if torch.cuda.is_available():
        params_senteval["encoder"] = model.encoder.cuda()
    else:
        params_senteval["encoder"] = model.encoder
    

    se = senteval.engine.SE(params_senteval, get_batch, prepare_samples)
    transfer_tasks = [
        "MR",
        "CR",
        "MPQA",
        "SUBJ",
        "SST5",
        "MRPC",
        "SICKEntailment",
        "SICKRelatedness",
        "TREC",
    ]
    results = se.eval(transfer_tasks)
    dictionary = {}
    for i in transfer_tasks:
        print("----------------- ", i, " -----------------")
        print(results[i])
        dictionary[i] = str(results[i])
    if path != None:
        json_object = json.dumps(dict(dictionary))
        with open(path, "w") as outfile:
            outfile.write(json_object)
