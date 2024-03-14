"""Tweaked AllenNLP dataset reader."""
import logging
import re
from random import random
from typing import Dict, List
from pypinyin import lazy_pinyin
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides
from utils.get_parallel_from_nlpcc import split_char
from utils.helpers import SEQ_DELIMETERS, START_TOKEN
from utils.helpers import FILTER

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def check(s):
    for ch in FILTER:
        if ch in s:
            return False
    return True

@DatasetReader.register("seq2labels_datareader")
class Seq2LabelsDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n

    and converts it into a ``Dataset`` suitable for sequence tagging. You can also specify
    alternative delimiters in the constructor.

    Parameters
    ----------
    delimiters: ``dict``
        The dcitionary with all delimeters.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    max_len: if set than will truncate long sentences
    """
    # fix broken sentences mostly in Lang8
    BROKEN_SENTENCES_REGEXP = re.compile(r'\.[a-zA-RT-Z]')

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 delimeters: dict = SEQ_DELIMETERS,
                 pinyin: bool = False,
                 skip_correct: bool = False,
                 skip_complex: int = 0,
                 lazy: bool = False,
                 max_len: int = None,
                 test_mode: bool = False,
                 mtl_pos: bool = False,
                 tag_strategy: str = "keep_one",
                 tn_prob: float = 0,
                 tp_prob: float = 0,
                 broken_dot_strategy: str = "keep",
                 granularity: str = "char") -> None:
        """
        数据读取器类
        :param token_indexers: token编号器
        :param delimeters: 分隔符
        :param skip_correct: 是否跳过正确的句子（在GECToR训练的第二阶段，只训练错误的句子）
        :param skip_complex: 是否跳过过于复杂的句子（是否含有编辑次数大于skip_complex的source_token）
        :param lazy: 是否为懒加载模式（节约内存）
        :param max_len: 是否对句子中长于max_len的token做截断
        :param mtl_pos: 是否进行预测此行的多任务学习机制
        :param test_mode: 是否测试模式
        :param tag_strategy:  两种抽取编辑label的策略：1）只保留各source token第一个的编辑label（GECToR的做法）。2）将各source token 所有的编辑label合并为一个（PIE的做法）。
        :param tn_prob: 按照某一概率阈值，跳过TN
        :param tp_prob: 按照某一概率阈值，跳过TP
        :param broken_dot_strategy: 针对破碎的句子的策略（尤其是lang-8中的数据）
        :param granularity: 模型的粒度
        """
        super().__init__(lazy)
        self.mtl_pos = mtl_pos
        self.granularity = granularity
        self._token_indexers = token_indexers if self.granularity == "char" else {'tokens': SingleIdTokenIndexer(namespace='tokens')}
        self._char_indexers = token_indexers if self.granularity == "word" else None
        self.pinyin = pinyin
        if self.pinyin:
            self._pinyin_indexers = {'pinyin_tokens': SingleIdTokenIndexer(namespace='pinyin_tokens')}
        self._delimeters = delimeters
        self._max_len = max_len
        self._skip_correct = skip_correct
        self._skip_complex = skip_complex
        self._tag_strategy = tag_strategy
        self._broken_dot_strategy = broken_dot_strategy
        self._test_mode = test_mode
        self._tn_prob = tn_prob
        self._tp_prob = tp_prob
        # self.pre_instance = None

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, "r") as label_file:
            with open(file_path+'.pos', 'r') as pos_file:
                logger.info("Reading instances from lines in file at: %s", file_path)
                for line in label_file:
                    line = line.strip("\n")
                    pos_line = pos_file.readline().strip("\n")
                    # skip blank and broken lines
                    if not line or (not self._test_mode and self._broken_dot_strategy == 'skip'
                                    and self.BROKEN_SENTENCES_REGEXP.search(line) is not None):
                        continue
                    tokens_and_tags = [pair.rsplit(self._delimeters['labels'], 1)
                                       for pair in line.split(self._delimeters['tokens']) if check(pair)]
                    char_tokens = None
                    pinyin_tokens = None
                    pos_tags = None
                    if self.mtl_pos:
                        pos_tags = [pair.split(self._delimeters['pos_tags'])[1]
                                           for pair in pos_line.split(self._delimeters['labels'])]
                        try:
                            assert len(tokens_and_tags) == len(pos_tags) + 1
                        except AssertionError:
                            # print(tokens_and_tags, len(tokens_and_tags))
                            # print(pos_line, len(pos_tags))
                            continue
                    try:
                        if self.granularity == "char":
                            tokens = [Token(token) for token, tag in tokens_and_tags]
                        elif self.granularity == "word":
                            tokens = [Token(token) for token, tag in tokens_and_tags]
                            char_tokens = [[Token(char) for char in split_char(token)] for token, tag in tokens_and_tags[1:]]
                        else:
                            raise Exception("Wrong Granularity")
                        tags = [tag for token, tag in tokens_and_tags]
                    except ValueError:
                        # print(tokens_and_tags)
                        continue

                    if tokens and tokens[0] != Token(START_TOKEN):
                        tokens = [Token(START_TOKEN)] + tokens

                    if char_tokens:
                        char_tokens = [[Token(START_TOKEN)]] + char_tokens

                    words = [x.text for x in tokens]
                    if self.pinyin:
                        pinyin_tokens = []
                        for word in words:
                            py = lazy_pinyin(word, errors="ignore")
                            if not py:
                                pinyin_tokens.append(Token("<MASK>"))
                            else:
                                pinyin_tokens.append(Token(py[0]))
                    if self._max_len is not None:
                        tokens = tokens[:self._max_len]
                        tags = None if tags is None else tags[:self._max_len]
                        if self.mtl_pos:
                            pos_tags = None if pos_tags is None else pos_tags[:self._max_len - 1]
                        if char_tokens:
                            char_tokens = char_tokens[:self._max_len]
                        if pinyin_tokens:
                            pinyin_tokens = pinyin_tokens[:self._max_len]
                    instance = self.text_to_instance(tokens, char_tokens, tags, words, pinyin_tokens, pos_tags)
                    if instance:
                        # self.pre_instance = instance
                        yield instance

    def extract_tags(self, tags: List[str]):
        op_del = self._delimeters['operations']

        labels = [x.split(op_del) for x in tags]  # 将每个source token的所有编辑label分开
        labels2 = []

        complex_flag_dict = {}
        # get flags
        for i in range(5):
            idx = i + 1
            complex_flag_dict[idx] = sum([len(x) > idx for x in labels])  # 统计编辑次数大于idx的source token个数

        if self._tag_strategy == "keep_one":  # 两种抽取编辑label的策略：1）只保留各source token第一个的编辑label（GECToR的做法）。2）将各source token 所有的编辑label合并为一个（PIE的做法）。
            # get only first candidates for r_tags in right and the last for left
            for x in labels:  # 按照论文中的做法，如果当前token有多个编辑，选择第一个不是$KEEP的
                if len(x) == 1:
                    labels2.append(x[0])
                else:
                    labels2.append(x[1] if x[0] == "$KEEP" else x[0])
            labels = [x[0] for x in labels]

        elif self._tag_strategy == "merge_all":
            # consider phrases as a words
            pass
        else:
            raise Exception("Incorrect tag strategy")

        detect_tags = ["CORRECT" if label == "$KEEP" else "INCORRECT" for label in labels]  # 抽取一下当前token是否有误的标签，后面要用到。
        return labels2, detect_tags, complex_flag_dict

    def text_to_instance(self, tokens: List[Token],
                         char_tokens: List[List[Token]] = None,
                         tags: List[str] = None,
                         words: List[str] = None,
                         pinyin_tokens: List[Token] = None,
                         pos_tags: List[str] = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": words})
        if self.pinyin:
            fields["pinyin_tokens"] = TextField(pinyin_tokens, self._pinyin_indexers)
        if char_tokens:
            fields["char_tokens"] = TextField([chars[0] for chars in char_tokens], self._char_indexers)
        if tags is not None:
            labels, detect_tags, complex_flag_dict = self.extract_tags(tags)
            if self._skip_complex and complex_flag_dict[self._skip_complex] > 0:  # 跳过过于复杂的句子（句子的复杂程度定义为是否有需要很多次编辑的source token）
                return None
            rnd = random()
            # skip TN
            if self._skip_correct and all(x == "CORRECT" for x in detect_tags):
                if rnd > self._tn_prob:
                    return None
            # skip TP
            else:
                if rnd > self._tp_prob:
                    return None

            fields["labels"] = SequenceLabelField(labels, sequence,
                                                  label_namespace="labels")
            fields["d_tags"] = SequenceLabelField(detect_tags, sequence,
                                                  label_namespace="d_tags")
            if pos_tags is not None:
                fields["pos_tags"] = SequenceLabelField(pos_tags, TextField(tokens[1:], self._token_indexers),
                                                  label_namespace="pos_tags")
        return Instance(fields)
