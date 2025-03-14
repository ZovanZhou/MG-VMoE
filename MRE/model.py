import tensorflow as tf
from framework import ZeroShotMRelModel
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Dropout, Layer


class Proto(ZeroShotMRelModel):
    def __init__(self, encoder, use_img: bool = False, d: int = 64):
        ZeroShotMRelModel.__init__(self, encoder, use_img)
        self.ln1 = Dense(d)
        self.ln2 = Dense(d)
        self.dropout = Dropout(0.1)

    @tf.function
    def call(self, data, n_class, training=False):
        data = self.unpack_data(data)
        _, h_entity, h_label = self.encoder(*data)

        hidden_size = h_label.get_shape().as_list()[-1]
        h_label_emb = tf.reshape(h_label, (-1, n_class, hidden_size))

        h_emb = h_entity

        if training:
            h_emb = self.dropout(h_emb, training=training)
            h_label_emb = self.dropout(h_label_emb, training=training)
        h_emb = self.ln1(h_emb)
        h_label_emb = self.ln2(h_label_emb)

        logits = self._batch_dist(h_emb, h_label_emb)
        pred = tf.argmax(logits, axis=-1)
        aux_loss = 0.0
        return logits, pred, aux_loss

class MMProto(ZeroShotMRelModel):
    def __init__(
        self, encoder, use_img: bool = True, d: int = 64, hidden_size: int = 768
    ):
        ZeroShotMRelModel.__init__(self, encoder, use_img)
        self.ln1 = Dense(d)
        self.ln2 = Dense(d)
        self.ln3 = Dense(hidden_size, activation="tanh")
        self.dropout = Dropout(0.1)

    @tf.function
    def call(self, data, n_class, training=False):
        data = self.unpack_data(data)
        h_sentence, h_image, h_entity, h_label, vibmoe_loss = self.encoder(*data)

        hidden_size = h_label.get_shape().as_list()[-1]
        h_label_emb = tf.reshape(h_label, (-1, n_class, hidden_size))

        h_emb = self.ln3(h_entity)

        if training:
            h_emb = self.dropout(h_emb, training=training)
            h_label_emb = self.dropout(h_label_emb, training=training)
        h_emb = self.ln1(h_emb)
        h_label_emb = self.ln2(h_label_emb)

        logits = self._batch_dist(h_emb, h_label_emb)
        pred = tf.argmax(logits, axis=-1)

        aux_loss = vibmoe_loss
        return logits, pred, aux_loss


class MMBilinearProto(ZeroShotMRelModel):
    def __init__(
        self, encoder, use_img: bool = True, d: int = 256
    ):
        ZeroShotMRelModel.__init__(self, encoder, use_img)
        self.ln1 = Dense(d)
        self.ln2 = Dense(d)
        self.dropout = Dropout(0.1)

    @tf.function
    def call(self, data, n_class, training=False):
        data = self.unpack_data(data)
        h_sentence, h_image, h_entity, h_label, aux_loss, gc_loss = self.encoder(*data, training=training)

        hidden_size = h_label.get_shape().as_list()[-1]
        h_label_emb = tf.reshape(h_label, (-1, n_class, hidden_size))

        h_emb = h_entity

        if training:
            h_emb = self.dropout(h_emb, training=training)
            h_label_emb = self.dropout(h_label_emb, training=training)
        h_emb = self.ln1(h_emb)
        h_label_emb = self.ln2(h_label_emb)

        logits = self._batch_dist(h_emb, h_label_emb)
        pred = tf.argmax(logits, axis=-1)

        return logits, pred, aux_loss, gc_loss
