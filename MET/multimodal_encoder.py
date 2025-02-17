import tensorflow as tf
from vit_keras import vit
from transformer import EncoderLayer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import initializers
from attention import AttentionWeightedAverage


class SimpleMultimodalEncoder(tf.keras.models.Model):
    def __init__(self, sentence_encoder):
        super(SimpleMultimodalEncoder, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.image_encoder = vit.vit_b16(
            image_size=(384, 384),
            pretrained=True,
            include_top=False,
            pretrained_top=False,
        )
        for l in self.image_encoder.layers:
            l.trainable = False

    @tf.function
    def call(
        self,
        s_ind,
        s_seg,
        l_ind,
        l_seg,
        mask_idx,
        img,
        training=False,
    ):
        h_sentence, h_entity, h_label = self.sentence_encoder(
            s_ind, s_seg, l_ind, l_seg, mask_idx
        )
        h_image = self.image_encoder(img)
        h_image = tf.reduce_mean(h_image, axis=1)
        h_entity = tf.concat([h_entity, h_image], axis=-1)
        aux_loss = 0.0
        return h_sentence, h_image, h_entity, h_label, aux_loss


class VIBMoELayer(tf.keras.models.Model):
    def __init__(self, hidden_size, n_expert):
        super(VIBMoELayer, self).__init__()
        self.hidden_size = hidden_size
        self.n_expert = n_expert
        self.dense4pi = Dense(n_expert, activation="softmax")
        self.dense4mu = Dense(hidden_size * n_expert)
        self.dense4logvar = Dense(hidden_size * n_expert)
    
    @tf.function
    def kl_loss(self, mu, logvar):
        _kl_loss = -0.5 * tf.reduce_sum(
            1.0 + logvar - tf.math.square(mu) - tf.math.exp(logvar), axis=-1
        )
        return tf.reduce_mean(_kl_loss)

    @tf.function
    def call(self, h_mm):
        seq_len_mm = tf.shape(h_mm)[1]

        pi = self.dense4pi(h_mm)
        mu = self.dense4mu(h_mm)
        logvar = self.dense4logvar(h_mm)
        kl_loss = self.kl_loss(mu, logvar)

        mu = tf.reshape(
            mu, shape=(-1, seq_len_mm, self.hidden_size, self.n_expert)
        )
        var = tf.reshape(
            tf.math.exp(logvar),
            shape=(-1, seq_len_mm, self.hidden_size, self.n_expert),
        )
        epsilon = tf.random.normal(tf.shape(mu))
        N = mu + tf.math.sqrt(var) * epsilon
        h_o = tf.squeeze(tf.matmul(N, tf.expand_dims(pi, axis=-1)), axis=-1)

        aux_loss = self.aux_loss(pi)
        loss = kl_loss + aux_loss
        return h_o, loss
    
    @tf.function
    def aux_loss(self, pi):
        entropy = - tf.reduce_sum(pi * tf.math.log(pi), axis=-1)
        return tf.reduce_mean(entropy)


class MultimodalEncoder(tf.keras.models.Model):
    def __init__(self, sentence_encoder, hidden_size: int = 768, n_expert: int = 8):
        super(MultimodalEncoder, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.image_encoder = vit.vit_b16(
            image_size=(384, 384),
            pretrained=True,
            include_top=False,
            pretrained_top=False,
        )
        for l in self.image_encoder.layers:
            l.trainable = False
        self.ibmoe = VIBMoELayer(hidden_size, n_expert)
        self.perturbation = self.add_weight(
            shape=(1, hidden_size * 2),
            name="perturbation",
            initializer=initializers.get("uniform"),
        )

        self.ln = Dense(1)
    
    def construct_graph(self, h):
        inner_product = h @ tf.transpose(h)
        instance_norm = tf.norm(h, ord=2, axis=-1, keepdims=True)
        instance_norm_matrix = instance_norm @ tf.transpose(instance_norm)
        g = (inner_product / instance_norm_matrix + 1.0) / 2.0
        return g

    def graph_contrastive_loss(self, h, g):
        h_p = g @ h
        h_n = (1.0 - g) @ h
        p_logits = tf.reduce_sum(h * h_p, axis=-1, keepdims=True)
        n_logits = tf.reduce_sum(h * h_n, axis=-1, keepdims=True)
        logits = tf.nn.softmax(tf.concat([p_logits, n_logits], axis=-1), axis=-1)
        return logits

    def attention(self, q, v):
        max_len = tf.shape(v)[1]
        h_q = tf.tile(tf.expand_dims(q, axis=1), [1, max_len, 1])
        logits = tf.nn.softmax(self.ln(tf.concat([h_q, v], axis=-1)), axis=1)
        h_v = tf.reduce_sum(logits * v, axis=1)
        return h_v

    @tf.function
    def infonec_loss(self, h_mm_text, h_mm_image):
        bs = tf.shape(h_mm_text)[0]
        labels = tf.eye(bs)
        logits = h_mm_text @ tf.transpose(h_mm_image)
        prob1 = tf.nn.softmax(logits, axis=0)
        prob2 = tf.nn.softmax(logits, axis=1)
        loss1 = tf.losses.categorical_crossentropy(labels, tf.transpose(prob1))
        loss2 = tf.losses.categorical_crossentropy(labels, prob2)
        return tf.reduce_mean((loss1 + loss2) / 2.0)

    @tf.function
    def call(
        self,
        s_ind,
        s_seg,
        l_ind,
        l_seg,
        mask_idx,
        img,
        training=False,
    ):
        h_sentence, h_e, h_label = self.sentence_encoder(
            s_ind, s_seg, l_ind, l_seg, mask_idx
        )
        h_image = self.image_encoder(img)

        h_mm_text, ibmoe_loss4text = self.ibmoe(h_sentence)
        h_mm_image, ibmoe_loss4image = self.ibmoe(h_image)
        ibmoe_loss = (ibmoe_loss4text + ibmoe_loss4image) / 2.0

        h_mm = tf.concat([h_mm_image, h_mm_text], axis=1)
        h_mm_e = self.attention(h_e, h_mm)
        h_entity = tf.concat([h_mm_e, h_e], axis=-1)

        h_mm_text = tf.reduce_mean(h_mm_text, axis=1)
        h_mm_image = tf.reduce_mean(h_mm_image, axis=1)
        infonec_loss = self.infonec_loss(h_mm_text, h_mm_image)

        h_mm = tf.concat([h_mm_image, h_mm_text], axis=-1)
        graph = self.construct_graph(h_mm)
        graph_logits_p = self.graph_contrastive_loss(h_mm, graph)
        graph_logits_n = self.graph_contrastive_loss(h_mm + self.perturbation, graph)
        graph_contrastive_loss = tf.reduce_mean(tf.losses.kl_divergence(graph_logits_p, graph_logits_n))
        perturbation_reg_loss = tf.norm(self.perturbation, ord=2)

        aux_loss = ibmoe_loss + infonec_loss + graph_contrastive_loss

        return h_sentence, h_image, h_entity, h_label, aux_loss, 0.0 - graph_contrastive_loss + perturbation_reg_loss
