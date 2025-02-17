import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score, precision_score, recall_score
from rich.progress import (
    SpinnerColumn,
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def get_ranking_loss(margin: float = 1.0):
    def _loss(y_true, y_pred):
        def _loss_elem(i):
            scores, pos_label = y_pred[i], tf.cast(y_true[i], dtype="int32")
            pos_score = scores[pos_label]
            loss = 0.0
            for j in tf.range(tf.shape(scores)[0]):
                if j != pos_label:
                    neg_score = scores[j]
                    loss += tf.maximum(0, margin - pos_score + neg_score)
            return loss

        return tf.map_fn(_loss_elem, tf.range(tf.shape(y_true)[0]), dtype=K.floatx())

    return _loss


class ZeroShotMNetModel(tf.keras.models.Model):
    def __init__(self, encoder, use_img):
        super(ZeroShotMNetModel, self).__init__()
        self.encoder = encoder
        self.use_img = use_img
        self.loss_func = get_ranking_loss()

    def __dist__(self, x, y, dim):
        return tf.reduce_sum(x * y, axis=dim)

    def _batch_dist(self, X, Y):
        if len(tf.shape(X)) == 2:
            X = tf.expand_dims(X, axis=1)
        return self.__dist__(X, Y, 2)

    def unpack_data(self, data):
        return data[:-1] if not self.use_img else data

    def call(self, data, n_class):
        raise NotImplementedError

    def loss(self, logits, label):
        return tf.reduce_mean(self.loss_func(label, logits))

    def accuracy(self, pred, label):
        return tf.reduce_mean(tf.cast(pred == label, tf.float32))

    def metrics(self, pred, label):
        macro_p  = precision_score(label, pred, average="macro")
        macro_r  = recall_score(label, pred, average="macro")
        macro_f1 = f1_score(label, pred, average="macro")
        micro_f1 = f1_score(label, pred, average="micro")
        return macro_p, macro_r, macro_f1, micro_f1


class ZeroShotMNetFramework(object):
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_n_class,
        val_n_class,
        test_n_class,
        beta,
    ) -> None:
        self.__train_dataloader = train_dataloader
        self.__val_dataloader = val_dataloader
        self.__test_dataloader = test_dataloader
        self.__train_n_class = train_n_class
        self.__val_n_class = val_n_class
        self.__test_n_class = test_n_class
        self.__beta = beta

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold red]{task.fields[info]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

    def __get_data(self, dataloader):
        (
            s_ind,
            s_seg,
            l_ind,
            l_seg,
            mask_idx,
            label,
            img,
        ) = next(dataloader)
        s_len = s_ind.get_shape().as_list()[-1]
        l_len = l_ind.get_shape().as_list()[-1]
        dim_img_feature = img.get_shape().as_list()[-3:]
        data = (
            tf.reshape(s_ind, shape=(-1, s_len)),
            tf.reshape(s_seg, shape=(-1, s_len)),
            tf.reshape(l_ind, shape=(-1, l_len)),
            tf.reshape(l_seg, shape=(-1, l_len)),
            tf.reshape(mask_idx, shape=(-1, 1)),
            tf.reshape(img, shape=(-1, *dim_img_feature)),
        )
        label = tf.reshape(label, shape=(-1,))
        return (data, label)

    def __train_model_with_batch(self, model, optimizer, dataloader, n_class):
        train_data, train_label = self.__get_data(dataloader)
        with tf.GradientTape() as tape:
            gc_loss = model(train_data, n_class, training=True)[-1]
        grad = tape.gradient(gc_loss, model.encoder.perturbation)
        optimizer.apply_gradients(zip([grad], [model.encoder.perturbation]))
        with tf.GradientTape() as tape:
            logits, pred, aux_loss = model(train_data, n_class, training=True)[:3]
            loss = model.loss(logits, train_label)
            overall_loss = loss + self.__beta * aux_loss
        grads = tape.gradient(overall_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        acc = model.accuracy(pred, train_label)
        return loss, acc

    def train(
        self,
        model,
        lr: float,
        epoch: int,
        patience: int,
        train_iter: int,
        val_iter: int,
        model_path: str,
    ):
        dataloader = self.__train_dataloader
        n_class = self.__train_n_class
        losses = []
        train_accs = []
        val_loss = 0.0
        f1_micro, f1_macro = 0.0, 0.0
        min_val_loss = np.inf
        n_patience = 0
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        for e in range(1, epoch + 1):
            losses.clear()
            train_accs.clear()
            train_tqdm = self.progress.add_task(
                description=f"Training epoch {e}",
                total=train_iter,
                info="train_loss:--.--, train_acc:--.--%, val_loss:--.--",
            )
            self.progress.start()
            for _ in range(train_iter):
                loss, train_acc = self.__train_model_with_batch(
                    model, optimizer, dataloader, n_class
                )
                train_accs.append(train_acc)
                losses.append(loss)
                info = "train_loss: {0:2.6f}, train_acc: {1:3.2f}%, val_loss: {2:2.6f}".format(
                    np.mean(losses), 100 * np.mean(train_accs), min_val_loss
                )
                self.progress.advance(train_tqdm, advance=1)
                self.progress.update(train_tqdm, info=info)
            val_metrics, val_loss = self.eval(model, val_iter)
            if f1_macro <= val_metrics["F1-macro"] or f1_micro <= val_metrics["F1-micro"]:
                f1_macro = val_metrics["F1-macro"]
                f1_micro = val_metrics["F1-micro"]
                n_patience = 0
                min_val_loss = val_loss
                self.progress.log("[bold green]Best checkpoint")
                info = "F1-micro {0:3.2f} F1-macro {1:3.2f}".format(
                    val_metrics["F1-micro"] * 100, val_metrics["F1-macro"] * 100
                )
                self.progress.log("[bold blue] Valid result: " + info)
                model.save_weights(model_path)
            else:
                n_patience += 1
                if n_patience == patience:
                    break
        self.progress.log("[bold red]Finish training " + model_path)

    def _load_model(self, model, model_path):
        if os.path.exists(model_path):
            optimizer = tf.optimizers.Adam()
            self.__train_model_with_batch(
                model, optimizer, self.__train_dataloader, self.__train_n_class
            )
            model.load_weights(model_path, by_name=True)
        else:
            print(f"The model file [{model_path}] are not found !")

    def eval(self, model, val_iter, model_path: str = ""):
        if model_path:
            n_class = self.__test_n_class
            self._load_model(model, model_path)
            dataloader = self.__test_dataloader
        else:
            n_class = self.__val_n_class
            dataloader = self.__val_dataloader

        eval_tqdm = self.progress.add_task(
            description="Evaluating", total=val_iter, info="val_loss:--.--"
        )
        self.progress.start()

        labels, preds, logits = [], [], []
        for _ in range(val_iter):
            val_data, val_label = self.__get_data(dataloader)
            logit, pred = model(val_data, n_class)[:2]
            labels.append(val_label)
            logits.append(logit)
            preds.append(pred)
            self.progress.advance(eval_tqdm, advance=1)
        labels = tf.concat(labels, axis=0)
        logits = tf.concat(logits, axis=0)
        preds = tf.concat(preds, axis=0)
        loss = model.loss(logits, labels)
        p, r, f1, micro_f1 = model.metrics(preds.numpy(), labels.numpy())

        info = "val_loss:{0:2.6f}".format(loss)
        self.progress.update(eval_tqdm, info=info)
        return {
            "p": p,
            "r": r,
            "F1-macro": f1,
            "F1-micro": micro_f1,
        }, loss
