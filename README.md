# byteNER

Tested with Python 2.7, Keras v2.0.6, Theano backend. You will likely run into issues if you use a different version of Keras.

All NER labels must be seen in training data.

## Setup
Download https://github.com/rsennrich/subword-nmt into the top-level directory of byteNER.

Make sure you have Keras v2.0.6 installed, using Theano as the backend.

Run ``pip install -r requirements.txt`` to install dependencies.

Run ``tar xvf models/model.tar.gz models/`` to decompress models.

## Tagging
Usage: ``tagger.py [options]``

Example: 
```
python tagger.py -m models/20CNN,dropout0.5,bytedrop0.3,lr0.0001,bytes,bpe,blstm,crf,biocreative.model -i examples/example.src -o examples/example.iob --output_format iob
```

Options:
  ```
  -h, --help            show this help message and exit
  
  -m MODEL, --model=MODEL
                        Model location
  
  -i INPUT, --input=INPUT
                        Input file, one sample per line
  
  -o OUTPUT, --output=OUTPUT
                        Output file location
  
  --output_format=OUTPUT_FORMAT
                        Whether to output predicted tokens in IOB format or
                        src/tgt format. [iob|st]
  
  --get_probs=GET_PROBS
                        Get normalized log likelihoods of each sample
  
  --get_vectors=GET_VECTORS
                        Get output vectors of second-to-last layer in the
                        network. Currently only tested with the CNN-BLSTM-CRF
                        configuration
   ```
                        
If ```output_format=iob```, then the model will output a ```<output>.chr``` file in byte IOB format, where the first column is the byte and the second column is the IOB tag. Spaces are used as the column separators.

If ```output_format=st```, then the model will output ```<output>.src``` and ```<output>.tgt``` files. The ```.src``` file is one sample per line, and the ```.tgt``` file contains entity predictions for the corresponding line in the ```.src``` file. The predictions are of the form:

```S<starting_byte_offset_in_line> L<length_of_entity_in_bytes> <entity_type>```
                   
## Training
Usage: ```train.py [options]```

Example: 
```
python train.py --model_path models/example.model --train_data_file examples/train --dev_data_file examples/dev --test_data_file examples/test
```

Options:
  ```
  -h, --help            show this help message and exit
  --model_path=MODEL_PATH
                        Output file to save model to
  --input_format=INPUT_FORMAT
                        Input format [iob|st]
  --space_token=SPACE_TOKEN
                        If input format is iob, then use space_token for
                        spaces between bytes
  --train_data_file=TRAIN_DATA_FILE
                        Training data: if input_format == "st", then there are
                        two input files: train_data_file.src and
                        train_data_file.tgt
  --dev_data_file=DEV_DATA_FILE
                        Dev data: if input_format == "st", then there are two
                        input files: dev_data_file.src and dev_data_file.tgt
  --test_data_file=TEST_DATA_FILE
                        Test data: if input_format == "st", then there are two
                        input files: test_data_file.src and test_data_file.tgt
  --max_chars_in_sample=MAX_CHARS_IN_SAMPLE
                        Max number of characters in a data sample
  --embedding_input_dim=EMBEDDING_INPUT_DIM
                        Dimension of input vectors
  --embedding_output_dim=EMBEDDING_OUTPUT_DIM
                        Dimension of output embedding vectors
  --embedding_max_len=EMBEDDING_MAX_LEN
                        Set length of input data (in byte characters)
  --cnn_filters=CNN_FILTERS
                        Number of filters in CNN output
  --cnn_kernel_size=CNN_KERNEL_SIZE
                        Kernel size
  --cnn_padding=CNN_PADDING
                        Type of border for CNN
  --cnn_act=CNN_ACT     Activation fn for CNN
  --dense_final_act=DENSE_FINAL_ACT
                        Final activation fn in network
  --optimizer=OPTIMIZER
                        Optimizer
  --tag_scheme=TAG_SCHEME
                        IOBES or IOB2 tag scheme
  --batch_size=BATCH_SIZE
                        Number of samples to process in one batch
  --dropout=DROPOUT     Fraction of input units to dropout
  --lstm_units=LSTM_UNITS
  --lstm_act=LSTM_ACT   Activation fn for LSTM
  --num_byte_layers=NUM_BYTE_LAYERS
                        Number of CNN layers in architecture
  --nb_workers=NB_WORKERS
                        Number of threads or processes to use
  --nb_epochs=NB_EPOCHS
                        Number of epochs to train model for
  --residual=RESIDUAL   Whether to use residual connections
  --skip_residuals=SKIP_RESIDUALS
                        Whether to add a residual connection every other conv
                        layer
  --lr=LR               Learning rate of optimizer
  --use_bpe=USE_BPE     Whether to use byte pair encodings
  --num_operations=NUM_OPERATIONS
                        Number of merge operations for BPE algorithm
  --reload=RELOAD       Whether to reload a previously trained model
  --blstm_on_top=BLSTM_ON_TOP
                        Whether to use a BLSTM layer on top of the CNNs
  --crf_on_top=CRF_ON_TOP
                        Whether to use a CRF layer on top
  --use_word_embeddings=USE_WORD_EMBEDDINGS
                        Whether to also use pretrained word embeddings as
                        input
  --use_bytes=USE_BYTES
                        Whether to use byte embedding as input. Default is
                        true.
  --word_embeddings_file=WORD_EMBEDDINGS_FILE
                        Pretrained word embedding file. This needs to be set
                        in order to use word embeddings.
  --word_embedding_dim=WORD_EMBEDDING_DIM
                        Dimension of pretrained word embeddings
  --bpe_codes_file=BPE_CODES_FILE
                        Pretrained BPE file
  --use_bpe_embeddings=USE_BPE_EMBEDDINGS
                        Whether to use pretrained BPE embeddings as input.
                        Cannot be used with word embeddings
  --bpe_embeddings_file=BPE_EMBEDDINGS_FILE
                        Pretrained BPE embedding file. This needs to be set in
                        order to use BPE embeddings
  --bpe_embedding_dim=BPE_EMBEDDING_DIM
                        Dimension of pretrained BPE embeddings
  --byte_layer_for_embed=BYTE_LAYER_FOR_EMBED
                        Whether to use embedding inputs as inputs for the CNN
                        layers (or only BLSTM layers)
  --layer_for_bytes=LAYER_FOR_BYTES
                        stack of cnns or a blstm layer for bytes
  --temp_dir=TEMP_DIR   Directory to write evaluation scores
  --drop_bytes=DROP_BYTES
                        Whether to drop a fraction of bytes for each input
  --byte_drop_fraction=BYTE_DROP_FRACTION
                        Fraction of byte input to drop
  --train_data_stride=TRAIN_DATA_STRIDE
                        Stride in number of bytes to shift window to get next
                        training sample
  --trainable_bpe_embeddings=TRAINABLE_BPE_EMBEDDINGS
                        Whether BPE embeddings should be traininable
  --override_parameters=OVERRIDE_PARAMETERS
                        Whether to use specified parameters or parameters from
                        saved model, if they exist
  --get_probs=GET_PROBS
                        Get normalized log likelihoods of each sample
  --get_vectors=GET_VECTORS
                        Get output vectors of second-to-last layer in the
                        network. Currently only tested with the CNN-BLSTM-CRF
                        configuration
  --use_tokenization=USE_TOKENIZATION
                        Use tokenization features
  --repickle_data=REPICKLE_DATA
                        Whether to re-process and pickle data even if the
                        pickle file already exists
  --make_samples_unique=MAKE_SAMPLES_UNIQUE
                        Clean training data so that the user-provided samples
                        are all unique
```
