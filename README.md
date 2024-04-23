# Advanced Topics in Computational Semantics 2024

# Overview 
Provide a github ReadMe file describing the package/installation requirements, a guide to train and evaluate a model, and the structure of your code.


## Prerequisites
* Anaconda. Available at: https://www.anaconda.com/distribution/


## Dependencies installation
1. Open Anaconda prompt and clone this repository (or download and unpack zip):
```
git clone https://github.com/zoetzikra/atcs_2024_Practical_1
```
2. Create the environment:
```
conda env create -f environmentatcs_env_cpu.yml
conda env create -f environmentatcs_env_gpu.yml

```
3. Activate the environment:
```
conda activate atcs_2024
```

## Quick Overview

To see the notebook with the experimental results, see:
```
jupyter notebook demo.ipynb
```

## Replicating the Results

# Training the models

To train the models you can use the following commands:

```
python main.py --encoder_type AWE --optimizer SGD --lr 0.1 --lr_decay 0.99 --lr_shrink 5 --minlr 1e-5  --seed 42 --log_dir pl_logs/ 

python main.py --encoder_type LSTM --optimizer SGD --lr 0.1 --lr_decay 0.99 --lr_shrink 5 --minlr 1e-5  --seed 42 --log_dir pl_logs/ 

python main.py --encoder_type BiLSTM --optimizer SGD --lr 0.1 --lr_decay 0.99 --lr_shrink 5 --minlr 1e-5  --seed 42 --log_dir pl_logs/

python main.py --encoder_type BiLSTMMax --optimizer SGD --lr 0.1 --lr_decay 0.99 --lr_shrink 5 --minlr 1e-5  --seed 42 --log_dir pl_logs/ 
```

To run on the cluster computer, you can find the corresponding Snellius job files in the folder job_files

You can add the argument:
```
--development
```
to run the code on a smaller subset of the dataset. 
Pre-selected size:
    * 64x500 for train
    * 64x250 for val
	* 64x250 for test

# Code Structure

In preprocessing.py, the helper functions that pre-process and load the data are defined. 
The load_snli function does the following:
* loads the SNLI dataset from HuggingFace (and downsampling it if the --development argument is passed)
* calls the preprocess function, whichtokenizes the data (using word_tokenizer) and lowercases it
* constructs the vocabulary of the dataset: a Vocabulary() object which contains the word vectors and string-to-index and index-to-string mappings. For example, index_to_vector is a dictionary that maps indices to vectors.
* creates and returns the DataLoaders. The vocabulary is also returned. 
* The collate function in the DataLoader uses the Vocabulary properties to turn the words into their indices such that they are passed into the encoders in the correct format. It also pads the sentences but it returns the original lengths are parts of the tuple, because they will be needed if the forward function of the model. If the padded lengths are passed into the Encoders then the gradient calculation is corrupted. 

How the embeddings are managed:
* The GloVe vectors are downloaded from the [Stanford website](http://nlp.stanford.edu/data/glove.840B.300d.zip) and unzipped. The text file is stored in the folder GloVeEmbeds.nosync (for mac users, to prevent from adding a .icloud extension to the file) and the zip folder is deleted. This process can be done by running 
```
jupyter notebook get_data.ipynb
```
* In the preprocess.py the vectors are turned into numpy arrays so they can be processed
* Then the build_word_dict function retuns a Counte() object and a dictionary mapping all the words occuring in the SNLI dataset to unique indices
* The dataset vocabulary is constructed by compiling the glove vectors only for the words that occur in the dataset. This makes the process much faster than making a vocabulary for all the glove vectors.


Training and Evaluation
PyTorch Lightning was the framework used for the training, it provided a very streamlined experience with the logging and the saving of the models. 
The structure was closely followed from the Deep Learning 1 PyTorch Lightning tutorial [here](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html?highlight=lightning)

The NLI model was made up of one of the 4 implemented encoders, and an MLP classifier built according to the original paper. 
The training checkpoints was be found in this [Google Drive](https://drive.google.com/drive/folders/15WIsEZc--zIwH7clWICz_nvmiAa0mp_j?usp=sharing).

The demo.ipynb is a short (and unfortunately incomplete) demonstration of the pre-trained results/analysis of the models.



# SentEval Transfer Task Evaluation:

To run the Sent Eval evaluation, I first installed the necessary libraries by doing the following:

Clone the SentEval project:
```
git clone https://github.com/facebookresearch/SentEval.git
```
2. Go to the *SentEval* folder.

3. Install SentEval through the file provided:
```
python setup.py install
```
4. Go to the *data/downstream* folder and run the following script on Snellius, by adapting it to match the format of a job file:
```
get_transfer_data.bash
```
5. Move a copy of the GloVe embeddings text file to the *SentEval/pretrained* folder.

7. Move the entire *SentEval* folder inside the *practical_1* folder

8. Run SentEval by adding the senteval function in the main.py and write the results to a file. 


Unfornunately, it was not possible to run Step 8 because of multiple small technical errors that were the cause either of the code or the sent eval libraries. It is possible to review the senteval script sentEval.py


