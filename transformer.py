import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

from imblearn.over_sampling import RandomOverSampler#SMOTE
sm = RandomOverSampler()#SMOTE()

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
                'att':self.att,
                'ffn':self.ffn,
                'layernorm1':self.layernorm1,
                'layernorm2':self.layernorm2,
                'dropout1':self.dropout1,
                'dropout2':self.dropout2
            })
        return config
        
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
                'token_emb':self.token_emb,
                'pos_emb':self.pos_emb
            })
        return config
        
data = pd.read_csv("607nn_50_200_33_s2000.txt", header = None)
data.columns = ['ip','lsb8', 'lsb7', 'lsb6', 'x1','x2', 'x3', 'x4','x5', 'x6', 'x7','x8', 'x9', 'x10','x11', 'x12', 'x13','x14', 'x15', 'x16','x17', 'x18', 'x19','x20', 'x21', 'x22','x23', 'x24', 'x25','x26', 'x27', 'x28','x29', 'x30','x31', 'x32', 'y']

x = data.loc[:,'lsb6':'x32']
y = data.loc[:, 'y']
yy = array(y) + 64 
xx = array(x) +64

x_res,y_res=sm.fit_resample(xx,yy)
X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.2, random_state=1)

vocab_size = 700# Only consider the top 20k words
maxlen = 33  # Only consider the first 200 words of each movie review
#(x_train, y_traxin), (x_val, y_val) = imdb.load_data(num_words=vocab_size)
#print(len(x_train), "Training sequences")
#print(len(x_val), "Validation sequences")
#x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
#x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(129, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
#   X_train, y_train, 
    xx, yy, batch_size=128, epochs=1, shuffle=True, validation_data=(X_test, y_test)
)

Model = "607nn_transformer"
model.save(Model)
