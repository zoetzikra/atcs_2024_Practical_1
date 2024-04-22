import os
import sys
import time

import argparse

import numpy as np

import torch
import torch.nn as nn


import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from preprocess_data import get_batch, build_vocab_embeddings, load_snli
from models import AWEsEncoder, LSTMEncoder, BiLSTMEncoder, MLP_classifier

GLOVE_PATH = "/GloVe_embeds/glove.840B.300d.txt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NLIModule(pl.LightningModule):

    def __init__(self, model_name, vocab, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, 
        # AND creates "self.hparams" namespace
        self.save_hyperparameters()

        
        # Create model (no need to put self.hparams.model_name here, but only in the other methods)
        if model_name == 'AWE':
            self.encoder = AWEsEncoder().to(device)
            self.classifier = MLP_classifier().to(device)
        if model_name == 'LSTM':
            self.encoder = LSTMEncoder().to(device)
            self.classifier = MLP_classifier(input_dim=4*2048).to(device)
        if model_name == 'BiLSTM':
            self.encoder = BiLSTMEncoder().to(device)
            self.classifier = MLP_classifier(input_dim=4*2*2048).to(device)
        if model_name == 'BiLSTMMax':
            self.encoder = BiLSTMEncoder(pooling='max').to(device)
            self.classifier = MLP_classifier(input_dim=4*2*2048).to(device)

        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()

        self.glove_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(vocab.vectors))
        self.glove_embeddings.weight.requires_grad = False
        self.glove_embeddings = self.glove_embeddings.to(device)
        #OR
        # self.glove_embeddings = nn.Embedding(len(vocab), 300)
        # self.glove_embeddings.weight.requires_grad = False
        # self.glove_embeddings.weight.data.copy_(torch.from_numpy(vocab.vectors))
        # self.glove_embeddings = self.glove_embeddings.to(device)
        
        self.last_val_acc = None


    def forward(self, batch):
        '''
        The forward method produces the prediction logits for the NLI task.
        First, we use a shared sentence encoder that outputs a representation for the premise and the hypothesis.
        Then we extract 3 relations from the sentences:
        - the concatenation of the two representations
        - the element-wise product of the two representations
        - the absolute element-wise difference of the two representations
        Finally,  The resulting vector, which captures information from is fed into the 3-class classifier
        
        batch - batch of sentences with (premise, hypothesis, label) pairs
        '''
        # Convert word indices to embeddings
        # embedded_sentences = self.embeddings(batch)
        
        # Extract the premise and hypothesis sentences from the batch
        # premises, hypothesis, labels = batch
        (premises_padded, hypotheses_padded, premises_lengths, hypothesis_lengths), _ = batch
        premises = premises_padded
        hypothesis = hypotheses_padded
        # Convert sentences to embeddings
        premises = self.glove_embeddings(premises)
        hypothesis = self.glove_embeddings(hypothesis)
        
        # Get the lengths of the premise and hypothesis sentences
        # for every sentence in the batch, get the length of the sentence and store it in a list
        premise_lengths = premises_lengths
        hypothesis_lengths = hypothesis_lengths
        
        # print("Premise lengths of 4 first premises: ", premise_lengths[:4])
        # print("Hypothesis lengths: ", hypothesis_lengths[:4])        
        # Get the representations for the premise and hypothesis sentences
        premise_rep = self.encoder(premises.to(device), premise_lengths)
        hypothesis_rep = self.encoder(hypothesis.to(device), hypothesis_lengths)

        # Concatenate the representations of the premise and hypothesis sentences
        # concat_rep = torch.cat((premise_rep, hypothesis_rep), dim=1)
        # Absolute element-wise difference of the representations of the premise and hypothesis sentences
        diff_rep = torch.abs(premise_rep - hypothesis_rep)
        # Element-wise product of the representations of the premise and hypothesis sentences
        prod_rep = premise_rep * hypothesis_rep
        # make up the complete sentence representation
        sentence_rep = torch.cat((premise_rep, hypothesis_rep, diff_rep, prod_rep), dim=1)
        # sentence_rep = torch.cat((concat_rep, prod_rep, diff_rep), dim=1)

        # Get the logits for the 3-class classifier
        predictions = self.classifier(sentence_rep).to(device)

        return predictions

    
    def configure_optimizers(self):
        # We will support SGD but also Adam, in case we would like to reproduce Section 5.1 of the original paper
        if self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), self.hparams.optimizer_hparams['lr'])
        elif self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(
                self.parameters(), self.hparams.optimizer_hparams['lr'])
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams.optimizer_hparams['lr_decay'])
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        predictions = self.forward(batch)
        # _, _, labels = batch
        (_, _, _, _), labels = batch
        labels = torch.tensor(labels).to(predictions.device)
        train_loss = self.loss_module(predictions, labels) # potentially batch.label or labels.squeeze() ?? WHY THO TODO: CHECK
        predicted_labels = torch.argmax(predictions, dim=1)
        train_acc = torch.true_divide(torch.sum(predicted_labels == labels), torch.tensor(labels.shape[0]))

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', train_acc, on_step=False, on_epoch=True, batch_size=64)
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, batch_size=64)
        # Return tensor to call ".backward" on
        return train_loss  

    def validation_step(self, batch, batch_idx): 
        prediction_logits = self.forward(batch)
        _, _, labels = batch
        labels = torch.tensor(labels).to(prediction_logits.device)
        val_loss = self.loss_module(prediction_logits, labels)
        predicted_labels = torch.argmax(prediction_logits, dim=1)
        print("predicted labels: ", predicted_labels)
        print("Labels: ", labels)
        val_acc = (labels == predicted_labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # save the validation accuracy
        self.last_val_acc = val_acc.item()  
    

    def test_step(self, batch, batch_idx):
        prediction_logits = self.forward(batch)
        _, _, labels = batch
        labels = torch.tensor(labels).to(prediction_logits.device)
        test_loss = self.loss_module(prediction_logits, labels)
        predicted_labels = torch.argmax(prediction_logits, dim=1)
        test_acc = (labels == predicted_labels).float().mean()
        
        # By default logs it per epoch (weighted average over batches)
        self.log('test_acc', test_acc)
        self.log('test_loss', test_loss)


'''
FYI:
Callbacks are self-contained functions that contain the non-essential logic of your Lightning Module. 
They are usually called after finishing a training epoch, but can also influence other parts of your 
training loop.
'''

'''
    FYI:
    "For all our models trained on SNLI, we use SGD
    with a learning rate of 0.1 and a weight decay of
    0.99. At each epoch, we divide the learning rate
    by 5 if the dev accuracy decreases. We use minibatches of size 64 and training is stopped when the
    learning rate goes under the threshold of 10^(−5)."
    '''

# Pytorch Lightning callback class
class PLCallback(pl.Callback):

    def __init__(self, minlr=1e-5):
        super().__init__()
        self.minlr = 0.095

        # initialize a field for the last validation accuracy so we can use it in the next method for comparison
        self.last_val_acc = None

    def on_train_epoch_end(self, trainer, pl_module):
        '''
        Each parameter group is a dictionary that holds parameters ex. learning rate, weight decay, etc. 
        Since we didn't specify any groups when creating the optimizer, 
        there will be one group that contains all parameters.
        '''
        # If the learning rate is below a certain threshold, stop training
        last_train_lr = trainer.optimizers[0].param_groups[0]['lr']
        print("Last training learning rate: ", last_train_lr)
        print("Minimum learning rate: ", self.minlr)
        if last_train_lr < self.minlr:
            print('Stopping training because the learning rate is below the minimum threshold')
            trainer.should_stop = True
    
class LearningRateMonitor(pl.Callback):
    
    def __init__(self, lr_shrink=5):
        super().__init__()
        self.lr_shrink = lr_shrink

        # initialize a field for the last validation accuracy so we can use it in the next method for comparison
        self.best_val_acc = 0.0

    # VERSION 1
    def on_validation_epoch_end(self, trainer, pl_module):
        print("CAME IN HERE")
        print("self.last_val_acc: ", self.last_val_acc)
        print("pl_module.last_val_acc: ", pl_module.last_val_acc)

        lr = trainer.optimizers[0].param_groups[0]['lr']
        print()
        print("LR:", lr)
        val_acc = pl_module.last_val_acc
        print("val_acc: ", val_acc)

        # If the validation accuracy has decreased, shrink the learning rate
        if pl_module.last_val_acc <= self.best_val_acc :
            new_lr = lr / self.lr_shrink
            trainer.optimizers[0].param_groups[0]['lr'] = new_lr
            
            print('Shrinking the learning rate by a factor of %s' % self.lr_shrink )
            self.best_val_acc = pl_module.last_val_acc

        self.best_val_acc = max(self.best_val_acc, val_acc)

        print("Best eval: ", self.best_val_acc, end=' ')
        print("Current eval:", val_acc)
     
        
    # VERSION 2
    # def on_val_epoch_end(self, trainer, pl_module):

    #     if pl_module.last_val_acc is not None and pl_module.last_val_acc < self.last_val_acc:
    #         current_lr = trainer.optimizers[0].param_groups[0]['lr']
    #         new_lr = current_lr / self.lr_shrink 
        
    #     optimizer_hparams_updated = self.hparams.optimizer_hparams.copy()
    #     optimizer_hparams_updated['lr'] = new_lr

    #     if pl_module.hparams.optimizer_name == "SGD":
    #         new_optimizer = torch.optim.SGD(self.parameters(), **optimizer_hparams_updated)
    #     if pl_module.hparams.optimizer_name == "Adam":
    #         new_optimizer = torch.optim.AdamW(self.parameters(), **optimizer_hparams_updated)
    #     trainer.optimizers = [new_optimizer]
     
    #     # we also need to create a new lr scheduler to use the new optimizer:
    #     new_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=new_optimizer, step_size=1, gamma=self.hparams.optimizer_hparams['lr_decay'])
    #     trainer.lr_schedulers[0] = trainer.configure_schedulers([new_scheduler])
        
    #     # save the last validation accuracy
    #     self.last_val_acc = pl_module.last_val_acc



# function to train the specified model
def train_model(args, model_hparams, optimizer_hparams):
    """
    Function for training and testing a model.
    Inputs:
        args - Namespace object from the argument parser
    """

    # create the logging directory
    os.makedirs(args.log_dir, exist_ok=True)

    # create the vocabulary and datasets for the SNLI task
    # above the train_loader and val_loader are already assumed
    vocab, train_loader, val_loader, test_loader = load_snli(batch_size=args.batch_size, development=args.development)
    
    # check if a checkpoint has been given
    if args.checkpoint_dir is not None:
        # create a PyTorch Lightning trainer because we still need one for testing
        trainer = pl.Trainer(logger=False, gpus=1 if torch.cuda.is_available() else 0)

        # load model from the given checkpoint
        model = NLIModule.load_from_checkpoint(args.checkpoint_dir)
    
    else: # train the model from scratch

        # create the callback for decreasing the learning rate
        pl_callback = PLCallback(minlr=args.minlr)

        # create a learning rate monitor callback
        lr_monitor = LearningRateMonitor(lr_shrink=args.lr_shrink)

        # create a model checkpoint callback
        checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")

        # create a PyTorch Lightning trainer
        trainer = pl.Trainer(
                    default_root_dir=args.log_dir,
                    callbacks=[lr_monitor, pl_callback, checkpoint_callback],
                    max_epochs=100,
                    enable_progress_bar=True)
        trainer.logger._default_hp_metric = None

        # create model
        # passing the vocab as argument because it allows the model to access the pre-trained vectors in vocab.vectors and use them to initialize the glove_embeddings layer.
        pl.seed_everything(args.seed)
        model = NLIModule(model_name=args.encoder_type, vocab=vocab, model_hparams=model_hparams, 
                        optimizer_name=args.optimizer, optimizer_hparams=optimizer_hparams)

        # Move the model to the GPU if one is available
        if torch.cuda.is_available():
            model.cuda()

        # train the model
        trainer.fit(model, train_loader, val_loader)

        # load the best checkpoint
        model = NLIModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # test the model
    model.freeze()
    test_result = trainer.test(model, dataloaders=test_loader, verbose=True)

    # return the test results
    return test_result



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLI training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model hyperparameters
    parser.add_argument('--encoder_type', default='AWE', type=str,
                        help='Type of encoder to use in the model',
                        choices=['AWE', 'LSTM', 'BiLSTM', 'BiLSTMMax'])

    parser.add_argument("--pool_type", type=str, default=None, 
                        help="Type of pooling to use in the model",
                        choices=['max'])

    # optimizer hyperparameters
    parser.add_argument("--optimizer", type=str, default="sgd", 
                        help="Optimizer to use for training",
                        choices=['SGD', 'Adam'])
    parser.add_argument("--lr", type=float, default=0.1, 
                        help="Learning rate for the optimizer")
    parser.add_argument("--lr_decay", type=float, default=0.99, 
                        help="Learning rate decay after each epoch")
    parser.add_argument("--lr_shrink", type=float, default=5, 
                        help="shrink factor for sgd. Factor to divide learning rate by when dev accuracy decreases.")
    parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")

    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, default=30,
                        help="Number of epochs to train the model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for reproducibility")
    parser.add_argument("--log_dir", type=str, default='pl_logs/',
                        help="Directory for saving the PyTorch Lightning logs")
    parser.add_argument("--development", action='store_true',
                        help="Use development dataset")
    
    # in case we want to load a model from a checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory for loading a checkpoint")
    
    args = parser.parse_args()

    # create the model hyperparameters
    model_hparams = {'encoder_type': args.encoder_type, 'pool_type': args.pool_type}

    # create the optimizer hyperparameters
    # optimizer_hparams = {'lr': args.lr}
    optimizer_hparams = {'lr': args.lr, 'lr_decay': args.lr_decay, 'lr_shrink': args.lr_shrink}

    # train the model
    test_result = train_model(args, model_hparams, optimizer_hparams)

    #print the test result returned by the training 
    print(test_result) 


# Run the following command to train the model:
# python main.py --encoder_type AWE --optimizer sgd --lr 0.1 --lr_decay 0.99 --lr_shrink 5 --minlr 1e-5  --seed 42 --log_dir pl_logs/ --development