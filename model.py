from bert4keras.models import build_transformer_model, BERT

class Roberta(BERT):
    def apply(self, inputs=None, layer=None, arguments=None, **kwargs):
        """
        Modify apply function so that the behaviour of LayerNormalization is modified
        inputs: last layers' output；
        layer: layer class to be used；
        arguments: parameters to layer.call；
        kwargs: parameters to layer init
        """
        if layer.__name__ == 'LayerNormalization' and kwargs.get('epsilon') is None:
            kwargs['epsilon'] = 1e-5
        return super().apply(inputs=inputs, layer=layer, arguments=arguments, **kwargs)