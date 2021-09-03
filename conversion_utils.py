import re
import numpy as np
from tensorflow.keras import backend as K

def convert_token_dict(token_dict):
    MAPPING = {
        '<s>': '[CLS]',
        '</s>': '[SEP]',
        '<pad>': '[PAD]',
        '<unk>': '[UNK]',
        '<mask>': '[MASK]',
    }
    for k in MAPPING:
        if k in token_dict:
            ind = token_dict.pop(k)
            token_dict[MAPPING[k]] = ind
    
    # hard swap of pad to zero
    if token_dict.get('[PAD]', 0) != 0:
        pad_ind = token_dict.pop('[PAD]')
        token_dict['[PAD]'] = 0
    total = len(token_dict)
    keys = list(token_dict.keys())

    # replace symbol for breaking tokens first
    for k in keys:
        if not re.search('^Ġ.*', k) and not re.search('\[.+\]', k):
            ind = token_dict.pop(k)
            if f'##{k}' in token_dict:
                print(f'unable to convert: {k} {ind}')
                token_dict[k] = ind
            else:
                token_dict[f'##{k}'] = ind
    print(len(token_dict))

    # replace symbol for complete tokens
    keys = list(token_dict.keys()) 
    for k in keys:
        if re.search('^Ġ.*', k):
            ind = token_dict.pop(k)
            if re.sub('^Ġ', '', k) in token_dict:
                print(f'unable to convert: {k} {ind}')
                token_dict[k] = ind
            else:
                token_dict[re.sub('^Ġ', '', k)] = ind
    
    for k, ind in token_dict.items():
        if ind == 0 and k != '[PAD]':
            token_dict[k] = pad_ind

    print(len(token_dict))
    return token_dict


def write_list(filename, a):
    with open(filename, "w") as f:
        f.write("\n".join(a))


def set_transformer_weight(keras_model, hf_encoder_layers):
    for ind, encoder in enumerate(hf_encoder_layers):
        print(f'setting layer {ind}')
        self_attention_layer = keras_model.get_layer(f'Transformer-{ind}-MultiHeadSelfAttention')
        param2set = self_attention_layer.count_params()
        att_mapping = [
            (self_attention_layer.q_dense, encoder.attention.self_attention.query),
            (self_attention_layer.k_dense, encoder.attention.self_attention.key),
            (self_attention_layer.v_dense, encoder.attention.self_attention.value),
            (self_attention_layer.o_dense, encoder.attention.dense_output.dense),
        ]
        total_param = [l.count_params() for _, l in att_mapping]
        print(f'total params assigned: {sum(total_param)} ({param2set})')
        for _keras_layer, _hf_layer in att_mapping:
            weights = _hf_layer.weights
            _keras_layer.set_weights([K.eval(w) for w in weights])
        
        att_norm_layer = keras_model.get_layer(f'Transformer-{ind}-MultiHeadSelfAttention-Norm')
        hf_norm_layer = encoder.attention.dense_output.LayerNorm
        att_norm_layer.set_weights([K.eval(w) for w in [hf_norm_layer.beta, hf_norm_layer.gamma]])

        feedforward_layer = keras_model.get_layer(f'Transformer-{ind}-FeedForward')
        param2set = feedforward_layer.count_params()
        feedforward_mapping = [
            (feedforward_layer.i0_dense, encoder.intermediate.dense),
            (feedforward_layer.o_dense, encoder.bert_output.dense)
        ]
        total_param = [l.count_params() for _, l in feedforward_mapping]
        print(f'total params assigned: {sum(total_param)} ({param2set})')
        for _keras_layer, _hf_layer in feedforward_mapping:
            weights = _hf_layer.weights
            _keras_layer.set_weights([K.eval(w) for w in weights])
        
        
        feedfoward_norm_layer = keras_model.get_layer(f'Transformer-{ind}-FeedForward-Norm')
        hf_norm_layer = encoder.bert_output.LayerNorm
        feedfoward_norm_layer.set_weights([K.eval(w) for w in [hf_norm_layer.beta, hf_norm_layer.gamma]])

def compare_2_array(a, b, rtol=1e-4, atol=1e-8):
    return np.isclose(a, b, rtol=rtol, atol=atol).sum() / np.product(a.shape)