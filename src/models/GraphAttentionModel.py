"""Defines the graph attention encoder model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from src.layers import AttentionLayer
from src.layers import EmbeddingLayer
from src.layers import ffn_layer
from src.models.Transformer import DecoderStack
from src.models.Transformer import PrePostProcessingWrapper, LayerNormalization
from src.utils.TransformerUtils import get_padding, get_decoder_self_attention_bias, get_padding_bias, get_position_encoding
from src.utils import beam_search
from src.utils.metrics import MetricLayer


class EncoderStack(tf.keras.layers.Layer):
    """Transformer encoder stack.
    The encoder stack is made up of N identical src.layers. Each layer is composed
    of the sublayers:
        1. Self-attention layer
        2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self, args):
        super(EncoderStack, self).__init__()
        self.args = args
        self.node_role_layer = tf.keras.layers.Dense(args.hidden_size, input_shape=(2 * args.hidden_size,))

        self.layers = []
        for _ in range(args.enc_layers):
            self_attention_layer = AttentionLayer.SelfAttention(args.hidden_size, args.num_heads, args.dropout)
            feed_forward_network = ffn_layer.FeedForwardNetwork(args.hidden_size, args.filter_size, args.dropout)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, args),
                PrePostProcessingWrapper(feed_forward_network, args)
            ])

        self.output_normalization = LayerNormalization(args.hidden_size)

    def call(self, node_tensor, label_tensor, node1_tensor, node2_tensor, attention_bias, inputs_padding, training):
        edge_tensor = tf.concat([node1_tensor, node2_tensor], axis=2)
        edge_tensor = self.node_role_layer(edge_tensor)
        
        node_tensor = tf.add(node_tensor, tf.add(edge_tensor, label_tensor))

        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.name_scope(f"layer_{n}"):
                with tf.name_scope("self_attention"):
                    node_tensor = self_attention_layer(node_tensor, attention_bias, training=training)

                with tf.name_scope("ffn"):
                    node_tensor = feed_forward_network(node_tensor, training=training)

        return self.output_normalization(node_tensor)


class TransGAT(tf.keras.Model):
    """
    Model that uses Graph Attention encoder and RNN decoder (for now)
    """

    def __init__(self, args, src_vocab_size, src_lang,
                 tgt_vocab_size, max_seq_len, tgt_vocab):
        super(TransGAT, self).__init__()
        self.regularizer = tf.keras.regularizers.l2()
        self.emb_layer = EmbeddingLayer.EmbeddingSharedWeights(
            src_vocab_size, args.emb_dim)

        self.tgt_emb_layer = EmbeddingLayer.EmbeddingSharedWeights(
            tgt_vocab_size, args.emb_dim)

        self.metric_layer = MetricLayer(tgt_vocab_size)

        self.encoder = EncoderStack(args)
        self.decoder_stack = DecoderStack(args)
        self.vocab_tgt_size = tgt_vocab_size
        self.target_lang = src_lang
        self.args = args
        self.num_heads = args.num_heads
        self.max_len = max_seq_len

    def _get_symbols_to_logits_fn(self, max_decode_length, training):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = get_position_encoding(
            max_decode_length + 1, self.args.emb_dim)
        decoder_self_attention_bias = get_decoder_self_attention_bias(
            max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.
            Args:
              ids: Current decoded sequences. int tensor with shape [batch_size *
                beam_size, i + 1]
              i: Loop index
              cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.
            Returns:
              Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                 updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.tgt_emb_layer(decoder_input)
            decoder_input += timing_signal[i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_outputs = self.decoder_stack(
                decoder_input,
                cache.get("encoder_outputs"),
                self_attention_bias,
                cache.get("encoder_decoder_attention_bias"),
                training=training,
                cache=cache)
            logits = self.tgt_emb_layer(decoder_outputs, mode="linear")
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn

    def predict(self, encoder_outputs, encoder_decoder_attention_bias, training):
        """Return predicted sequence."""
        encoder_outputs = tf.cast(encoder_outputs, tf.float32)
        batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = self.max_len

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(
            max_decode_length, training)
        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)
        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.args.hidden_size]),
                "v": tf.zeros([batch_size, 0, self.args.hidden_size])
            } for layer in range(self.args.enc_layers)
        }
        cache["encoder_outputs"] = encoder_outputs
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias
        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.vocab_tgt_size,
            beam_size=self.args.beam_size,
            alpha=self.args.beam_alpha,
            max_decode_length=max_decode_length,
            eos_id=6)

        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]
        return {"outputs": top_decoded_ids, "scores": top_scores}

    
    def call(self, nodes, labels, node1, node2, targ, mask=None, training=None):
        """
        Puts the tensors through encoders and decoders
        :param nodes: node features
        :type nodes: tf.Tensor
        :param labels: label features
        :type labels: tf.Tensor
        :param node1: node1 features
        :type node1: tf.Tensor
        :param node2: node2 features
        :type node2: tf.Tensor
        :param targ: target sequences
        :type targ: tf.Tensor
        :param mask: mask sequences
        :type mask: tf.Tensor
        :param training: Whether the model is in training mode, defaults to None
        :type training: bool, optional
        :return: output probability distribution
        :rtype: tf.Tensor
        """
        node_tensor = self.emb_layer(nodes)
        label_tensor = tf.cast(self.emb_layer(labels), dtype=tf.float32)
        node1_tensor = tf.cast(self.emb_layer(node1), dtype=tf.float32)
        node2_tensor = tf.cast(self.emb_layer(node2), dtype=tf.float32)
        
        # Pad shorter sequences to match the length of node_tensor
        max_length = tf.shape(node_tensor)[1]
        label_tensor = tf.pad(label_tensor, [[0, 0], [0, max_length - tf.shape(label_tensor)[1]], [0, 0]])
        node1_tensor = tf.pad(node1_tensor, [[0, 0], [0, max_length - tf.shape(node1_tensor)[1]], [0, 0]])
        node2_tensor = tf.pad(node2_tensor, [[0, 0], [0, max_length - tf.shape(node2_tensor)[1]], [0, 0]])
        
        attention_bias = get_padding_bias(nodes)
        attention_bias = tf.cast(attention_bias, tf.float32)
        inputs_padding = get_padding(node_tensor)

        enc_output = self.encoder(node_tensor, label_tensor, node1_tensor, node2_tensor,
                                attention_bias, inputs_padding, training=training)

        if targ is not None:
            decoder_inputs = tf.cast(self.tgt_emb_layer(targ), dtype=tf.float32)
        else:
            predictions = self.predict(enc_output, attention_bias, training=training)
            return predictions

        with tf.name_scope("shift_targets"):
            # Shift targets to the right, and remove the last element
            decoder_inputs = tf.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        attention_bias = tf.cast(attention_bias, tf.float32)
        with tf.name_scope("add_pos_encoding"):
            length = tf.shape(decoder_inputs)[1]
            pos_encoding = get_position_encoding(length, self.args.hidden_size)
            pos_encoding = tf.cast(pos_encoding, tf.float32)
            decoder_inputs += pos_encoding
        if training:
            decoder_inputs = tf.nn.dropout(decoder_inputs, rate=self.args.dropout)

        decoder_self_attention_bias = get_decoder_self_attention_bias(length, dtype=tf.float32)
        outputs = self.decoder_stack(decoder_inputs, enc_output, decoder_self_attention_bias,
                                    attention_bias, training=training)

        logits = self.tgt_emb_layer(outputs, mode="linear")
        logits = tf.cast(logits, tf.float32)

        return logits
