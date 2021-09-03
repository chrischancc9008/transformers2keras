from transformers import RobertaTokenizer, TFRobertaModel
from conversion_utils import (
    convert_token_dict, write_list,
    set_transformer_weight, compare_2_array
)
from config import roberta_config
import copy
import tensorflow as tf
from bert4keras.models import build_transformer_model
from model import Roberta
from tensorflow.keras import backend as K
from tokenizer import Tokenizer
import numpy as np
from tensorflow.keras.models import Model
import json


WITH_POOL = 'tanh'

# config
with open('roberta_config.json', 'w') as f:
    json.dump(roberta_config, f)

# prepare result of hugging face roberta
hf_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
hf_model = TFRobertaModel.from_pretrained('roberta-base')
text = " Replace me by any text you'd like."
encoded_input = hf_tokenizer(text, return_tensors='tf')
hf_output = hf_model(encoded_input)
hf_roberta_layer = hf_model.layers[0]
encoded_input = hf_tokenizer(text, return_tensors='tf')
hf_embedding_output = hf_roberta_layer.embeddings(encoded_input['input_ids'])


# vocab dictionary
token_dict = copy.deepcopy(hf_tokenizer.get_vocab())
token_dict = convert_token_dict(token_dict)
write_list('roberta_vocab.txt', [k for k, _ in sorted(token_dict.items(), key=lambda x: x[1])])


# prepare bert4keras roberta
roberta = build_transformer_model(
    config_path=None,
    checkpoint_path=None,
    model=Roberta,
    dropout_rate=0.1,
    with_pool=WITH_POOL,
    # custom_position_ids=False,
    **roberta_config
)

# weight mapping
layer_weight_mapping = {
    # 'Embedding-Token': hf_roberta_layer.embeddings.weight,
    'Embedding-Segment': hf_roberta_layer.embeddings.token_type_embeddings,
    'Embedding-Position': hf_roberta_layer.embeddings.position_embeddings[2:, :],  # as the position embbedding start with pad_idx+1, i.e., 2
    'Embedding-Norm': [hf_roberta_layer.embeddings.LayerNorm.beta,
                       hf_roberta_layer.embeddings.LayerNorm.gamma]
}

if WITH_POOL:
    layer_weight_mapping['Pooler-Dense'] = hf_roberta_layer.pooler.weights

for layer_name, weights in layer_weight_mapping.items():
    if not isinstance(weights, list):
        weights = [weights]
    layer = roberta.get_layer(layer_name)
    print(layer_name, layer.count_params())
    layer.set_weights([K.eval(w) for w in weights])
    
# swap pad and cls as keras4bert mask 0 by default
word_embed_layer = roberta.get_layer('Embedding-Token')
weight = K.eval(hf_roberta_layer.embeddings.weight)
weight[[0, 1]] = weight[[1, 0]]
word_embed_layer.set_weights([weight])

set_transformer_weight(roberta, hf_roberta_layer.encoder.layer)

tokenizer = Tokenizer('roberta_vocab.txt', do_lower_case=False)

input_ids = tokenizer.encode(text)[0]
_input = np.array([input_ids])
inputs = [_input, np.zeros_like(_input)]
output = roberta.predict(inputs)
last_layer_model = Model(roberta.inputs, roberta.get_layer('Transformer-11-FeedForward-Norm').output)
last_layer_output = last_layer_model.predict(inputs)
embedding_model = Model(roberta.inputs, roberta.get_layer('Embedding-Norm').output)
embedding_output = embedding_model.predict(inputs)

print(f'checking embedding match rate: {compare_2_array(embedding_output, K.eval(hf_embedding_output))}')
if WITH_POOL:
    print(f'checking pooling  match rate: {compare_2_array(output, K.eval(hf_output.pooler_output))}')
print(f'checking embedding  match rate: {compare_2_array(last_layer_output, hf_output.last_hidden_state)}')

roberta.save_weights('roberta_weights.h5')