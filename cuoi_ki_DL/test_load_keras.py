import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Dense, Dropout, LayerNormalization
from keras import ops

# Define custom layers matching the notebook exactly
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = tf.transpose(tf.reshape(query, (batch_size, -1, self.num_heads, self.projection_dim)), perm=[0, 2, 1, 3])
        key = tf.transpose(tf.reshape(key, (batch_size, -1, self.num_heads, self.projection_dim)), perm=[0, 2, 1, 3])
        value = tf.transpose(tf.reshape(value, (batch_size, -1, self.num_heads, self.projection_dim)), perm=[0, 2, 1, 3])
        
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        
        output = tf.matmul(weights, value)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.embed_dim))
        return self.combine_heads(output)

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential([Dense(ff_dim, activation="gelu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class AdaptiveGatingFusion(layers.Layer):
    def __init__(self, num_taps=3, **kwargs):
        super(AdaptiveGatingFusion, self).__init__(**kwargs)
        self.num_taps = num_taps

    def build(self, input_shape):
        self.gating_weights = self.add_weight(
            name="gating_weights", shape=(self.num_taps,), initializer="ones", trainable=True
        )
        super(AdaptiveGatingFusion, self).build(input_shape)

    def call(self, inputs):
        normalized_weights = tf.nn.softmax(self.gating_weights)
        weighted_features = [inputs[i] * normalized_weights[i] for i in range(self.num_taps)]
        return tf.add_n(weighted_features)

def create_pure_mold_vit_model(image_size=256, d_model=768, num_layers=12, num_heads=12, ff_dim=1536):
    inputs = layers.Input(shape=(image_size, image_size, 3), name="input_image")
    
    backbone = keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_tensor=inputs)
    backbone.trainable = False  
    
    features = backbone.output  
    patches = layers.Reshape((64, 2048), name="patch_flatten")(features)
    x = Dense(d_model, name="patch_projection")(patches)
    
    initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
    class_emb = tf.Variable(initial_value=initializer(shape=[1, 1, d_model]), name="class_emb", trainable=True)
    pos_emb = tf.Variable(initial_value=initializer(shape=[1, 64 + 1, d_model]), name="pos_emb", trainable=True)
    
    def add_embeddings(inputs_tensor):
        b_size = ops.shape(inputs_tensor)[0]
        c_emb_broadcasted = ops.broadcast_to(class_emb, [b_size, 1, d_model])
        concated = ops.concatenate([c_emb_broadcasted, inputs_tensor], axis=1)
        return concated + pos_emb

    x = layers.Lambda(add_embeddings, name="vit_embedding_layer")(x)
    
    saved_taps = {}
    for i in range(num_layers):
        x = TransformerEncoderBlock(embed_dim=d_model, num_heads=num_heads, ff_dim=ff_dim, name=f"transformer_layer_{i+1}")(x)
        if i == 0:
            saved_taps['l1'] = layers.Lambda(lambda t: t[:, 0], name="tap_low_L1")(x)
        elif i == 5:
            saved_taps['l6'] = layers.Lambda(lambda t: t[:, 0], name="tap_mid_L6")(x)
            
    saved_taps['l12'] = layers.Lambda(lambda t: t[:, 0], name="tap_high_L12")(x)

    aggregate_feature_vector = AdaptiveGatingFusion(num_taps=3, name="adaptive_gating_fusion")([
        saved_taps['l1'], saved_taps['l6'], saved_taps['l12']
    ])
    
    x_head = LayerNormalization(epsilon=1e-6, name="MLP_Head_LayerNorm")(aggregate_feature_vector)
    x_head = Dense(256, name="MLP_Linear_Layer_1")(x_head)
    x_head = layers.Lambda(lambda v: tf.nn.gelu(v), name="GELU_Activation_1")(x_head)
    x_head = Dropout(0.3)(x_head)
    
    x_head = Dense(128, name="MLP_Linear_Layer_2")(x_head)
    x_head = layers.Lambda(lambda v: tf.nn.gelu(v), name="GELU_Activation_2")(x_head)
    x_head = Dropout(0.3)(x_head)
    
    outputs = Dense(2, activation='softmax', name="Final_Softmax_Output")(x_head)
    return keras.Model(inputs=inputs, outputs=outputs, name="MoLD_Hybrid_ViT")

def test_load():
    PATH_TO_WEIGHTS = r"d:\ffpp\cuoi_ki_DL\checkpoints\best_pure_mold.weights.h5"
    print("Building model graph...")
    model = create_pure_mold_vit_model(image_size=256)
    
    print("Running dummy batch...")
    dummy_input = tf.zeros((1, 256, 256, 3))
    _ = model(dummy_input, training=False)
    
    print("Loading weights...")
    model.load_weights(PATH_TO_WEIGHTS)
    print("SUCCESS! Model loaded perfectly.")

if __name__ == "__main__":
    test_load()
