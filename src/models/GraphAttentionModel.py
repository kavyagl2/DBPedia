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
        self.node_role_layer = tf.keras.layers.Dense(args.hidden_size)

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

        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.vocab_tgt_size,
            beam_size=self.args.beam_size,
            alpha=self.args.alpha,
            max_decode_length=max_decode_length,
            eos_id=1)
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]
        return {
            "outputs": top_decoded_ids,
            "scores": top_scores
        }

    def call(self, node_tensor, label_tensor, node1_tensor, node2_tensor, 
             adj_tensor, trg_input, trg_output, training):
        """Return predicted sequence and metrics."""
        with tf.name_scope("graph_encoder"):
            attention_bias = get_padding_bias(adj_tensor)
            inputs_padding = get_padding(adj_tensor)

            encoder_outputs = self.encoder(
                node_tensor, label_tensor, node1_tensor, node2_tensor, 
                attention_bias, inputs_padding, training)

        with tf.name_scope("targets"):
            targets = tf.cast(trg_output, tf.int32)
            target_weights = tf.cast(tf.not_equal(targets, 0), tf.float32)

        with tf.name_scope("logits"):
            # Save shape to facilitate restoring the encoder-decoder attention bias.
            max_target_length = tf.shape(trg_input)[1]

            decoder_self_attention_bias = get_decoder_self_attention_bias(
                max_target_length)
            decoder_inputs = self.tgt_emb_layer(trg_input)
            decoder_inputs += get_position_encoding(
                tf.shape(decoder_inputs)[1], self.args.emb_dim)

            # Run the decoder stack
            outputs = self.decoder_stack(
                decoder_inputs,
                encoder_outputs,
                decoder_self_attention_bias,
                attention_bias,
                training=training)
            logits = self.tgt_emb_layer(outputs, mode="linear")

        with tf.name_scope("metrics"):
            metrics = self.metric_layer(logits, targets, target_weights)

        return logits, metrics

    def get_config(self):
        return {
            "args": self.args,
            "src_vocab_size": self.emb_layer.vocab_size,
            "tgt_vocab_size": self.tgt_emb_layer.vocab_size,
            "max_len": self.max_len,
            "tgt_vocab": self.vocab_tgt_size
        }
