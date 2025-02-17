import tensorflow as tf
from keras_bert import load_trained_model_from_checkpoint
from tensorflow.keras.layers import Bidirectional, LSTM


class BERTEmbedding(tf.keras.models.Model):
    def __init__(
        self, bert_path: str, fine_tune: bool = False,
    ):
        super(BERTEmbedding, self).__init__()
        self.__ckpt_path = f"{bert_path}/bert_model.ckpt"
        self.__config_path = f"{bert_path}/bert_config.json"
        self.bert_model = load_trained_model_from_checkpoint(
            self.__config_path, self.__ckpt_path, seq_len=None,
        )
        for l in self.bert_model.layers:
            l.trainable = fine_tune

    @tf.function
    def call(self, ind, seg):
        return self.bert_model([ind, seg])


class BERTSentenceEncoder(tf.keras.models.Model):
    def __init__(self, bert_path: str, fine_tune: bool = True):
        super(BERTSentenceEncoder, self).__init__()
        self.bert_embedding = BERTEmbedding(bert_path, fine_tune)

    @tf.function
    def get_tensor_by_index(self, tensor, bs, pos):
        indices = tf.concat(
            [tf.expand_dims(tf.range(bs, dtype=tf.int64), axis=1), pos],
            axis=-1,
        )
        _tensor = tf.gather_nd(tensor, indices)
        return _tensor

    @tf.function
    def call(self, s_ind, s_seg, l_ind, l_seg, mask_idx):
        h_sentence = self.bert_embedding(s_ind, s_seg)
        h_label = tf.reduce_mean(self.bert_embedding(l_ind, l_seg), axis=1)
        batch_size = tf.shape(h_sentence)[0]
        h_e = self.get_tensor_by_index(h_sentence, batch_size, mask_idx)
        h_entity = tf.concat([h_sentence[:, 0, :], h_e], axis=-1)
        return h_sentence, h_entity, h_label