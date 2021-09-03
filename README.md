# Introduction

Conversion of roberta model weights from huggingface transformers package to bert4keras package

# Purpose

"Bert4keras" provides transformer models in keras, which is good for further modification

However, it only supports few models weights, e.g., english roberta weight is NOT supported.

huggingface "Transformers" provides a lot of transformer models but is hard to adjust.

For details, please refer to "Why shouldn't I use transformers?" https://github.com/huggingface/transformers

# Use Case

3 files will be provided after running the following codes.

    python conversion_test.py

1. **vocab file**: roberta_vocab.txt which could be used by Tokenizer in bert4keras with a small modification as shown in tokenizer.py

2. **config file**: roberta_config.json which could be used in build_transformer_model function in bert4keras.

3. **weight file**: roberta_weights.h5 which could be loaded after getting the keras model.


e.g.,

```
from bert4keras.models import build_transformer_model
from model import Roberta
from tokenizer import Tokenizer

roberta = build_transformer_model(
    config_path=roberta_config.json,
    checkpoint_path=None,
    model=Roberta,
    dropout_rate=0.1,
    **roberta_config
)

roberta.load_weights('roberta_weights.h5')

tokenizer = Tokenizer('roberta_vocab.txt', do_lower_case=False)
```

# Environment Set Up

Install packages,
    pip install -r requirement.txt

Export environment variables so that tensorflow.keras will be used by bert4keras
    export TF_KERAS=1

# Checking

    python conversion_test.py


# TODO

Support conversion other models