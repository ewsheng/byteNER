"""
Tag unlabeled data using pre-trained model
"""

from optparse import OptionParser

from model import NERModel
import preprocess, postprocess

import utils
import time
import os
import sys


def main(args):

    # User parameters
    parser = OptionParser()
    parser.add_option(
        "-m", "--model", default="",
        help="Model location"
    )
    parser.add_option(
        "-i", "--input", default="",
        help="Input file, one sample per line"
    )
    parser.add_option(
        "-o", "--output", default="",
        help="Output file location"
    )
    parser.add_option(
        "--output_format", default="iob",
        help="Whether to output predicted tokens in IOB format or src/tgt format. [iob|st]"
    )
    parser.add_option('--get_probs', default=0, help="Get normalized log likelihoods of each sample")
    parser.add_option('--get_vectors', default=0,
                      help="Get output vectors of second-to-last layer in the network. Currently only tested with the CNN-BLSTM-CRF configuration")

    opts = parser.parse_args(args)[0]

    # Check parameters validity
    assert opts.output_format in ["iob", "st"]
    assert os.path.isfile(opts.model)
    assert os.path.isfile(opts.model + "_parameters.pkl")  # need params file to reload model
    assert os.path.isfile(opts.input)

    # Add parameters
    parameters = {'reload': True, 'tag': True, 'repickle_data': True}

    # Load existing model
    print "Loading model..."
    model = NERModel(model_path=opts.model, parameters=parameters)
    parameters = model.parameters
    parameters['input'] = opts.input
    parameters['output'] = opts.output
    parameters['output_format'] = opts.output_format
    parameters['model'] = model.model
    parameters['get_probs'] = int(opts.get_probs) == 1
    parameters['get_vectors'] = int(opts.get_vectors) == 1

    print 'Tagging...'
    start = time.time()
    load_data_and_predict(parameters)
    print '---- lines tagged in %.4fs ----' % (time.time() - start)


def load_data_and_predict(parameters):
    """
    Preprocess input data and make predictions.
    :param parameters:
    :return:
    """
    # Convert user input to model input PKL format
    pkl_file = parameters['input'] + '.pkl'
    preprocess.write_user_input_to_model_input(parameters['input'], '', pkl_file)
    parameters['input'] = pkl_file

    max_chars_in_sample = parameters['max_chars_in_sample']
    utils.load_data_stride_x_chars_enc_dec(parameters['input'], parameters, stride=max_chars_in_sample / 2)
    file_ext = parameters['file_ext']
    char_to_num = parameters['char_to_num']

    if parameters['use_bpe']:
        if parameters['bpe_codes_file']:
            codes_file = parameters['bpe_codes_file']
        else:
            print 'BPE codes file does not exist!'
            exit()
        utils.load_bpe_data(parameters['input'] + '.bpe.' + str(max_chars_in_sample) + file_ext, char_to_num, codes_file, parameters['input'] + '.bpe-coded-' + os.path.basename(parameters['bpe_codes_file']) + '.' + str(max_chars_in_sample) + file_ext, parameters)

    if parameters['use_tokenization']:
        tok_char_to_num = parameters['tok_char_to_num']
        utils.load_tok_data(
            parameters['input'] + '.bpe.' + str(max_chars_in_sample) + file_ext,
            tok_char_to_num,
            parameters['input'] + '.tok.' + str(max_chars_in_sample) + file_ext, parameters)

    if parameters['use_word_embeddings']:
        vocab_dict = parameters['word_vocab_dict']
        # run word embedding features on dev and test data
        utils.load_word_embeddings_data(parameters['input'] + '.bpe.' + str(max_chars_in_sample) + '.pkl',
                                        vocab_dict,
                                        parameters['input'] + '.word-coded.' + str(max_chars_in_sample) + '.pkl',
                                        parameters)
    if parameters['use_bpe_embeddings']:
        if parameters['bpe_codes_file']:
            codes_file = parameters['bpe_codes_file']
        else:
            print 'BPE codes file does not exist!'
            exit()
        vocab_dict = parameters['bpe_vocab_dict']
        # run word embedding features on dev and test data
        utils.load_bpe_embeddings_data(parameters['input'] + '.bpe.' + str(max_chars_in_sample) + file_ext,
                                        vocab_dict,
                                        codes_file,
                                        parameters['input'] + '.bpe-embed-coded-' + os.path.basename(codes_file) + '.' + str(max_chars_in_sample) + file_ext,
                                        parameters)

    # determine what will be the combined data file
    combined_data_ext = utils.generate_combined_feat_ext(parameters)

    # Combine data features as necessary
    utils.combine_data(parameters['input'], combined_data_ext, parameters)

    # Make predictions
    metrics = utils.MetricsCheckpoint(parameters['model'], parameters)
    metrics.make_and_format_predictions(parameters['input'], metrics.test_X_data, parameters)

    # Output in specified format
    if parameters['output_format'] == 'iob':
        postprocess.ord_byte_iob_to_byte_iob(parameters['output'])
    else:  # st
        postprocess.byte_iob_to_src_tgt(parameters['output'], ordinal=True)


if __name__ == '__main__':
    main(sys.argv[1:])
