import os
import re
import codecs
import json
import hashlib
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from threading import Thread
from vit_keras import vit
from typing import List, Dict
from tensorflow.data import Dataset
from keras_bert.tokenizer import Tokenizer
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from keras_bert import TOKEN_CLS, TOKEN_MASK, TOKEN_SEP


class Entity(object):
    """
    Named Entity
    """

    def __init__(self, name, type, h_pos, t_pos, url):
        self.__name = name
        self.__type = type
        self.__h_pos = h_pos
        self.__t_pos = t_pos
        self.__url = url

    @property
    def name(self):
        return self.__name

    @property
    def type(self):
        return self.__type

    @property
    def pos(self):
        return [self.__h_pos, self.__t_pos]

    @property
    def url(self):
        return self.__url

    def __str__(self) -> str:
        return str({"name": self.name, "type": self.type, "pos": self.pos})


class MNetSample(Thread):
    """
    Multimodal Named Entity Typing Sample
    """

    def __init__(self, **kwargs) -> None:
        Thread.__init__(self)
        if "sample" not in kwargs:
            self.__sentence = kwargs["sentence"]
            self.__img_url = kwargs["img_url"]
            self.__data_path = kwargs["data_path"]
            self.__topic = kwargs["topic"]
            self.__entities = [
                Entity(name, type, h_pos, t_pos, url)
                for name, type, h_pos, t_pos, url in kwargs["entity"]
            ]
        else:
            sample = kwargs["sample"]
            self.__sentence = sample.sentence
            self.__image = sample.image
            self.__topic = sample.topic
            self.__entities = kwargs["entity"]

    def run(self):
        file_name = self.__parse_img_file_name(self.__img_url)
        self.__image = self.__load_image(f"{self.__data_path}/wikinewsImgs/{file_name}")

    @property
    def sentence(self):
        return self.__sentence

    @property
    def image(self):
        return self.__image

    @property
    def topic(self):
        return self.__topic

    @property
    def entity(self):
        return self.__entities

    def __str__(self) -> str:
        entities = "".join([str(e) for e in self.entity])
        return str({"sentence": self.sentence, "topic": self.topic, "entity": entities})

    def __parse_img_file_name(self, url: str):
        m_img = url.split("/")[-1]
        prefix = hashlib.md5(m_img.encode()).hexdigest()
        suffix = re.sub(
            r"(\S+(?=\.(jpg|JPG|png|PNG|svg|SVG)))|(\S+(?=\.(jpeg|JPEG)))", "", m_img
        )
        m_img = prefix + suffix
        m_img = m_img.replace(".svg", ".png").replace(".SVG", ".png")
        return m_img

    def __load_image(self, fname: str, image_size: int = 384):
        if not os.path.exists(fname):
            fname = f"{self.__data_path}/wikinewsImgs/17_06_4705.jpg"
        try:
            img = image.load_img(fname, target_size=(image_size, image_size))
        except Exception:
            fname = f"{self.__data_path}/wikinewsImgs/17_06_4705.jpg"
            img = image.load_img(fname, target_size=(image_size, image_size))
        x = image.img_to_array(img)
        # x = preprocess_input(x)
        x = vit.preprocess_inputs(x).reshape(image_size, image_size, 3)
        return x


class MNetDataset(object):
    def __init__(self, data_path: str):
        self.__samples = self.__collect_data(data_path)
        self.__dict_type2samples = self.__parse_relation_dict(self.__samples)

    @property
    def types(self):
        return list(self.__dict_type2samples.keys())

    def type2samples(self, type: str) -> List[MNetSample]:
        samples = []
        for i, idx in self.__dict_type2samples[type].items():
            sample = self.__samples[i]
            entities = [sample.entity[j] for j in idx]
            samples.append(MNetSample(sample=sample, entity=entities))
        return samples

    def __parse_relation_dict(self, samples: List[MNetSample]) -> Dict:
        dict_type2samples = {}
        for i, sample in enumerate(samples):
            for j, entity in enumerate(sample.entity):
                type = entity.type
                if type not in dict_type2samples:
                    dict_type2samples[type] = {}
                if i not in dict_type2samples[type]:
                    dict_type2samples[type][i] = []
                dict_type2samples[type][i].append(j)
        return dict_type2samples

    def __collect_data(self, data_path: str) -> List[MNetSample]:
        samples = []
        for dtype in ["train", "valid", "test"]:
            with open(f"{data_path}/{dtype}.json", "r") as fr:
                json_data = json.load(fr)
                for sentence, image_url, topic, entities in tqdm(
                    json_data, ascii=True, ncols=80
                ):
                    sample = MNetSample(
                        sentence=sentence,
                        img_url=image_url,
                        topic=topic,
                        entity=entities,
                        data_path=data_path,
                    )
                    sample.start()
                    samples.append(sample)
                for sample in samples:
                    sample.join()
        return samples


class TextProcessor(object):
    def __init__(self, bert_path: str, max_length: int = 128):
        self.__max_length = max_length
        self.__tokenizer = self.__load_tokenizer(bert_path)

    def encode(self, sentence: str, max_seq_len: int):
        indices, segments = self.__tokenizer.encode(first=sentence, max_len=max_seq_len)
        return indices, segments

    def tokenize(self, sentence: str, pos: List):
        h_in_idx, t_in_idx = pos
        context1 = sentence[0:h_in_idx]
        entity = sentence[h_in_idx:t_in_idx]
        context2 = sentence[t_in_idx:]

        tokens_context1 = self.__tokenizer.tokenize(context1)[1:-1]
        tokens_entity = (
            ["[unused1]"] + self.__tokenizer.tokenize(entity)[1:-1] + ["[unused2]"]
        )
        tokens_context2 = self.__tokenizer.tokenize(context2)[1:-1]

        tokens = (
            [TOKEN_CLS]
            + tokens_context1
            + tokens_entity
            + tokens_context2
            + [TOKEN_SEP]
        )
        mask_in_index = tokens.index("[unused1]")

        indices = self.__tokenizer._convert_tokens_to_ids(tokens)
        while len(indices) < self.__max_length:
            indices.append(0)
        indices = indices[: self.__max_length]
        segments = np.zeros_like(indices).tolist()

        return (indices, segments, mask_in_index)

    def __load_tokenizer(self, bert_path):
        token_dict = {}
        with codecs.open(f"{bert_path}/vocab.txt", "r", "utf8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return Tokenizer(token_dict)


class ZeroShotMNetDataset(object):
    def __init__(
        self,
        batch_size: int,
        classes: List[str],
        dataset: MNetDataset,
        textProcessor: TextProcessor,
        shuffle: bool = False,
        balanced_sampling: bool = False,
        n_shots: int = -1,
    ):
        self.__batch_size = batch_size
        self.__classes = classes
        self.__n_shots = n_shots
        self.__textProcessor = textProcessor
        (
            self.__dataset,
            self.__sample_index,
            self.__total_step,
        ) = self.__generate_data_index(dataset)
        self.__cur_step = 0
        self.__shuffle = shuffle
        self.__balanced_sampling = balanced_sampling

    @property
    def step(self):
        if self.__batch_size == 1:
            return self.__total_step
        else:
            return self.__total_step // self.__batch_size + 1

    def _generate_data(self):
        dict_sample_index = {}
        if self.__balanced_sampling:
            for i, j, k in self.__sample_index:
                if i not in dict_sample_index:
                    dict_sample_index[i] = []
                dict_sample_index[i].append((i, j, k))
        else:
            sample_index = self.__sample_index
            if self.__shuffle:
                np.random.shuffle(sample_index)
        del self.__sample_index
        while True:
            if self.__balanced_sampling:
                target_class_idx = dict_sample_index[
                    self.__cur_step % len(self.__classes)
                ]
                idx = target_class_idx[
                    np.random.choice(np.arange(len(target_class_idx)))
                ]
            else:
                idx = sample_index[self.__cur_step]
            packed_data = self.__getitem(idx)
            self.__cur_step += 1
            if self.__cur_step == self.__total_step:
                self.__cur_step = 0
                if self.__shuffle:
                    np.random.shuffle(sample_index)
            yield packed_data["sentence_indices"], packed_data[
                "sentence_segments"
            ], packed_data[
                "label_indices"
            ], packed_data[
                "label_segments"
            ], packed_data["mask_in_index"], packed_data[
                "label"
            ], packed_data[
                "img"
            ]

    def __generate_data_index(self, original_dataset):
        dataset = {}
        total_step = 0
        sample_index = []
        for i, class_name in enumerate(self.__classes):
            if class_name not in dataset:
                dataset[class_name] = []
            samples = original_dataset.type2samples(class_name)
            for j, sample in enumerate(samples):
                dataset[class_name].append(sample)
                entities = sample.entity
                for k, entity in enumerate(entities):
                    total_step += 1
                    sample_index.append((i, j, k))
        if self.__n_shots > 0:
            dict_idx_samples = {}
            for sample in sample_index:
                if sample[0] not in dict_idx_samples:
                    dict_idx_samples[sample[0]] = []
                dict_idx_samples[sample[0]].append(sample)
            shot_sample_index = []
            for v in dict_idx_samples.values():
                idx = np.arange(len(v), dtype=int).tolist()
                selected_idx = np.random.choice(
                    idx, size=self.__n_shots, replace=False
                ).tolist()
                shot_sample_index.extend([v[i] for i in selected_idx])
            sample_index = shot_sample_index
        return dataset, sample_index, total_step

    def __getitem(self, sample_index):
        i, j, k = sample_index
        target_type = self.__classes[i]
        packed_data = {
            "sentence_indices": [],
            "sentence_segments": [],
            "mask_in_index": [],
            "img": [],
            "label_indices": [],
            "label_segments": [],
            "label": [],
        }
        sample = self.__dataset[target_type][j]
        sentence = sample.sentence
        image = sample.image
        entity = sample.entity[k]
        (
            sentence_indices,
            sentence_segments,
            mask_in_index
        ) = self.__textProcessor.tokenize(sentence, entity.pos)
        packed_data["sentence_indices"].append(sentence_indices)
        packed_data["sentence_segments"].append(sentence_segments)
        packed_data["mask_in_index"].append(mask_in_index)
        packed_data["img"].append(image)
        packed_data["label"].append(self.__classes.index(target_type))

        for class_name in self.__classes:
            label_indices, label_segments = self.__textProcessor.encode(class_name, 5)
            packed_data["label_indices"].append(label_indices)
            packed_data["label_segments"].append(label_segments)
        return packed_data


def get_loader(
    classes,
    dataset,
    textProcessor,
    batch_size: int,
    shuffle: bool = False,
    balanced_sampling: bool = False,
    n_shots: int = -1,
):
    zeroShotMNetDataset = ZeroShotMNetDataset(
        batch_size, classes, dataset, textProcessor, shuffle, balanced_sampling, n_shots=n_shots
    )
    dataloader = Dataset.from_generator(
        zeroShotMNetDataset._generate_data,
        tuple([tf.int64] * 6 + [tf.float32]),
        tuple(
            [tf.TensorShape([None, None])] * 4
            + [tf.TensorShape([None])] * 2 + [tf.TensorShape([None] * 4)]
        ),
    )
    dataloader = dataloader.batch(batch_size)
    return iter(dataloader), zeroShotMNetDataset.step


if __name__ == "__main__":
    dataset = MNetDataset("./dataset")
    print(dataset.types)
    # def load_tokenizer(bert_path):
    #     token_dict = {}
    #     with codecs.open(f"{bert_path}/vocab.txt", "r", "utf8") as reader:
    #         for line in reader:
    #             token = line.strip()
    #             token_dict[token] = len(token_dict)
    #     return Tokenizer(token_dict)

    # tokenizer = load_tokenizer("../POS-LB-SF-ID/wwm_uncased_L-24_H-1024_A-16")

    # def tokenize_sample(tokenizer, sentence, pos1, pos2):
    #     context1 = sentence[0:pos1]
    #     entity = sentence[pos1:pos2]
    #     context2 = sentence[pos2:]

    #     tokens_context1 = tokenizer.tokenize(context1)[1:-1]
    #     tokens_entity = tokenizer.tokenize(entity)[1:-1]
    #     tokens_context2 = tokenizer.tokenize(context2)[1:-1]
    #     print(tokens_context1)
    #     print(tokens_entity)
    #     print(tokens_context2)
    #     tokens = (
    #         ["[CLS]"]
    #         + tokens_context1
    #         + ["[unused0]"]
    #         + tokens_entity
    #         + ["[unused1]"]
    #         + tokens_context2
    #         + ["[SEP]"]
    #     )
    #     print(tokens)
    #     pos_in_index = len(tokens_context1) + 1
    #     print(pos_in_index)
    #     assert pos_in_index == tokens.index("[unused0]")

    #     indices = tokenizer._convert_tokens_to_ids(tokens)
    #     print(indices)

    #     # while len(indices) < self.__max_length:
    #     #     indices.append(0)
    #     # indices = indices[: self.__max_length]
    #     segments = np.zeros_like(indices).tolist()

    # # sentence = "10 Chris Bond, 11 Ryan Scott and teammates during a time out."
    # # pos1, pos2 = 3, 13
    # # sentence = "Paul Schlesselman self-portrait photo from his MySpace page."
    # # pos1, pos2 = 0, 17
    # sentence = "The home page of nytimes.com."
    # pos1, pos2 = 17, 28
    # tokenize_sample(tokenizer, sentence, pos1, pos2)
