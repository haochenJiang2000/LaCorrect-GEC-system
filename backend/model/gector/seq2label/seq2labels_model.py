"""Basic model. Predicts tags for every token
GECToR真正意义上的模型，对一个源句子序列使用Transformer做encode，然后在每个token处使用MLP预测最可能的编辑label
"""
# -*- coding: utf-8

import os
from typing import *

from model.gector.seq2label.dynamic_crf_layer import DynamicCRF
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from model.gector.seq2label.seq2labels_metric import Seq2LabelsMetric
from overrides import overrides
from torch.nn.modules.linear import Linear
from allennlp.nn import util

@Model.register("seq2labels")
class Seq2Labels(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 pinyin_embedder: TextFieldEmbedder = None,
                 pinyin: bool = False,
                 predictor_dropout=0.0,
                 word_dropout=0.0,
                 labels_namespace: str = "labels",
                 detect_namespace: str = "d_tags",
                 label_smoothing: float = 0.0,
                 confidence: float = 0.0,
                 model_dir: str = "",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 bert_dim: int = 1024,
                 pinyin_dim: int = 200,
                 hidden_layers: int = 0,
                 hidden_dim: int = 512,
                 cuda_device: int = 3,
                 crf: bool = False,
                 low_rank_dim: int = 32,
                 crf_beam_approx: int = 64,
                 word_ins_loss_factor: float = 0.5,
                 dev_file: str = None,
                 logger=None,
                 vocab_path: str = None,
                 weight_name: str = None,
                 save_metric: str = "dev_m2",
                 mtl_pos: bool = False,
                 ) -> None:
        """
        seq2labels模型的构造函数
        :param vocab: 词典对象
        :param text_field_embedder: 嵌入器，这里采用预训练的类bert模型作为嵌入器对token进行embedding（即论文模型结构中的Transformer-Encoder）
        :param predictor_dropout: 预测器dropout概率（防止过拟合）
        :param labels_namespace: labels的命名空间（GECToR解码端的labels输出指的是编辑label，如$KEEP等）
        :param detect_namespace: detect的命令空间（GECToR解码端的d_tags输出指的是探测当前token是否出错的一个二分类标签）
        :param label_smoothing: 一个正则化的trick，减少分类错误带来的惩罚
        :param confidence:  $KEEP标签的偏差项
        :param model_dir:  模型保存路径
        :param initializer: 初始化器
        :param regularizer: 正则化器
        """
        super(Seq2Labels, self).__init__(vocab, regularizer)
        self.mtl_pos = mtl_pos
        self.save_metric = save_metric
        self.weight_name = weight_name
        self.cuda_device = cuda_device
        self.device = torch.device("cuda:" + str(cuda_device) if int(cuda_device) >= 0 else "cpu")
        self.label_namespaces = [labels_namespace,
                                 detect_namespace]  # 需要解码预测的标签的命名空间
        self.text_field_embedder = text_field_embedder
        self.pinyin = pinyin
        if self.pinyin:
            self.pinyin_embedder = pinyin_embedder
        assert 0 <= word_dropout <= 1
        self.word_dropout = word_dropout
        self.num_labels_classes = self.vocab.get_vocab_size(labels_namespace)
        self.num_detect_classes = self.vocab.get_vocab_size(detect_namespace)
        self.num_pos_tag_classes = self.vocab.get_vocab_size("pos_tags")
        self.label_smoothing = label_smoothing
        self.confidence = confidence
        self.incorr_index = self.vocab.get_token_index("INCORRECT",
                                                       namespace=detect_namespace)  # 获取INCORRECT标签的索引
        self.vocab_path = vocab_path
        self.best_metric = 0.0
        self.epoch = 0
        self.model_dir = model_dir
        self.logger = logger
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(predictor_dropout))
        # TimeDistributed模块是临时将(batch_size, time_steps, [rest])的time_steps维度暂时降维，变为(batch_size * time_steps, [rest])的矩阵，提供给某个模块（如维度为([rest], output_dim)的Linear层），
        # 然后再将该模块的输出(batch_size * time_steps, output_dim)变为(batch_size, time_steps, output_dim)的形式
        # TimeDistributed层在需要并行解码时非常常用。因为decoder端的MLP等需要同时应用在当前序列各time_step上，同步解码。
        self.dev_file = dev_file
        self.crf = crf
        self.word_ins_loss_factor = word_ins_loss_factor  # CRF Loss 的权重
        if self.crf:
            self.crf_layer = DynamicCRF(
                num_embedding=self.vocab.get_vocab_size(namespace="labels"),
                low_rank=low_rank_dim,
                beam_size=crf_beam_approx,
            )
        self.tag_labels_hidden_layers = []
        self.tag_detect_hidden_layers = []
        if pinyin:
            input_dim = bert_dim + pinyin_dim
        else:
            input_dim = bert_dim
        if hidden_layers > 0:
            self.tag_labels_hidden_layers.append(TimeDistributed(
                Linear(input_dim,
                       hidden_dim)).cuda(self.device))
            self.tag_detect_hidden_layers.append(TimeDistributed(
                Linear(input_dim,
                       hidden_dim)).cuda(self.device))
            for _ in range(hidden_layers - 1):
                self.tag_labels_hidden_layers.append(TimeDistributed(
                    Linear(hidden_dim, hidden_dim)).cuda(self.device))
                self.tag_detect_hidden_layers.append(TimeDistributed(
                    Linear(hidden_dim, hidden_dim)).cuda(self.device))
            self.tag_labels_projection_layer = TimeDistributed(
                Linear(hidden_dim,
                       self.num_labels_classes)).cuda(self.device)  # 编辑label预测线性投影层
            self.tag_detect_projection_layer = TimeDistributed(
                Linear(hidden_dim,
                       self.num_detect_classes)).cuda(self.device)  # 编辑label预测线性投影层
        else:
            self.tag_labels_projection_layer = TimeDistributed(
                Linear(input_dim, self.num_labels_classes)).to(self.device)  # 编辑label预测线性投影层
            self.tag_detect_projection_layer = TimeDistributed(
                Linear(input_dim, self.num_detect_classes)).to(self.device)  # 是否错误tag预测线性投影层

        if self.mtl_pos:
            self.tag_pos_projection_layer = TimeDistributed(
                Linear(input_dim, self.num_pos_tag_classes)).to(self.device)  # 编辑label预测线性投影层
            # self.tag_pos_crf_layer = ConditionalRandomField(self.num_pos_tag_classes)
        # self.metrics = {"accuracy": CategoricalAccuracy()}
        self.metric = Seq2LabelsMetric()
        # 模型的预测评价指标采用分类准确率
        self.UNKID = self.vocab.get_vocab_size("labels") - 2

        initializer(self)



    def source_word_dropout(self, encoded_text: torch.LongTensor, drop_prob):
        """
        对源句子序列中的词进行dropout：对每个词的bert表示以prob概率变为全0向量，其余词的bert表示以(1/(1-prob))的概率放大
        :param encoded_text: 句子的编码表示，维度：[batch_size, max_len, bert_output_dim]
        :param drop_prob: dropout概率
        :return: dropout结果
        """
        if drop_prob == 0:
            return encoded_text
        keep_prob = 1 - drop_prob
        scale_martix = torch.ones_like(encoded_text) * (1 / keep_prob)
        encoded_text = torch.mul(scale_martix, encoded_text)
        mask = (torch.randn((encoded_text.size()[:-1])) < keep_prob).unsqueeze(-1).cuda(self.device)
        encoded_text = encoded_text * mask.eq(1)
        return encoded_text

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                char_tokens: Dict[str, torch.LongTensor] = None,
                pinyin_tokens: Dict[str, torch.LongTensor] = None,
                labels: torch.LongTensor = None,
                d_tags: torch.LongTensor = None,
                pos_tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        "bert":[batch_szie,sen_len(wordpiece)]
        "bert-offsets":[batch_size,sen_len(token)]
        "mask":[batch_size,sen_len(token)]

        labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        d_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containing the original words in the sentence to be tagged under a 'words' key.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        if char_tokens:  # 词模式预测输出的情况（用每个词第一个字的BERT表示作为整个词的表示）
            encoded_text = self.text_field_embedder(
                char_tokens)  # 返回bert模型的输出。维度：[batch_size,seq_len,encoder_output_dim]
        else:
            encoded_text = self.text_field_embedder(tokens)  # 返回bert模型的输出。维度：[batch_size,seq_len,encoder_output_dim]

        if self.training:
            encoded_text = self.source_word_dropout(encoded_text, self.word_dropout)  # 源词dropout（微软2018论文）

        if pinyin_tokens:  # 是否对拼音进行嵌入和拼接
            encoded_pinyin = self.pinyin_embedder(pinyin_tokens)
            encoded_text = torch.cat((encoded_text, encoded_pinyin), dim=-1)

        batch_size, sequence_length, _ = encoded_text.size()
        mask = get_text_field_mask(tokens)  # [batch_size,seq_len]  # 返回mask标记（防止因句子长度不一致而padding的影响）

        # 训练模式（训练集）
        if self.training:
            ret_train = self.decode(encoded_text, batch_size, sequence_length, mask, labels, d_tags, pos_tags, metadata)
            _loss = ret_train['loss']
            output_dict = {'loss': _loss}
            logits_labels = ret_train["logits_labels"]
            logits_d = ret_train["logits_d_tags"]
            # for metric in self.metrics.values():
            #     metric(logits_labels, labels, mask.float())  # 计算编辑label预测准确率
            #     metric(logits_d, d_tags, mask.float())  # 计算d_tags预测准确率
            self.metric(logits_labels, labels, logits_d, d_tags, mask.float())
            return output_dict

        # 评测模式（开发集）
        training_mode = self.training
        self.eval()
        with torch.no_grad():
            ret_train = self.decode(encoded_text, batch_size, sequence_length, mask, labels, d_tags, pos_tags, metadata)
        self.train(training_mode)
        logits_labels = ret_train["logits_labels"]
        logits_d = ret_train["logits_d_tags"]
        if labels is not None and d_tags is not None:  # 如果没有提供golden labels标签和d_tags标签，那么就是预测模式（测试集），无需计算accuracy
            # for metric in self.metrics.values():
            #     metric(logits_labels, labels, mask.float())  # 计算编辑label预测准确率
            #     metric(logits_d, d_tags, mask.float())  # 计算d_tags预测准确率
            self.metric(logits_labels, labels, logits_d, d_tags, mask.float())
        if self.crf:
            _scores, _tokens = self.crf_layer.forward_decoder(logits_labels, mask)
            ret_train["predict_labels"] = _tokens
        return ret_train

    def decode(self, encoded_text: torch.LongTensor = None,
               batch_size: int = 0,
               sequence_length: int = 0,
               mask: torch.LongTensor = None,
               labels: torch.LongTensor = None,
               d_tags: torch.LongTensor = None,
               pos_tags: torch.LongTensor = None,
               metadata: List[Dict[str, Any]] = None) -> Dict:
        if self.tag_labels_hidden_layers:
            encoded_text_labels = encoded_text.clone().to(self.device)
            for layer in self.tag_labels_hidden_layers:
                encoded_text_labels = layer(encoded_text_labels)
            logits_labels = self.tag_labels_projection_layer(
                self.predictor_dropout(
                    encoded_text_labels))  # 用一个简单的全连接层预测当前token处的label得分，[batch_size,seq_len,num_labels_classes]
            for layer in self.tag_detect_hidden_layers:
                encoded_text = layer(encoded_text)
            logits_d = self.tag_detect_projection_layer(
                self.predictor_dropout(
                    encoded_text))  # 用一个简单的全连接层预测当前token处的label得分，[batch_size,seq_len,num_labels_classes]
        else:
            logits_labels = self.tag_labels_projection_layer(
                self.predictor_dropout(
                    encoded_text))  # 用一个简单的全连接层预测当前token处的label得分，[batch_size,seq_len,num_labels_classes]
            logits_d = self.tag_detect_projection_layer(
                encoded_text)  # 用一个简单的全连接层预测当前token处的label得分，[batch_size,seq_len,num_detect_classes]
        if pos_tags is not None:
            logits_pos_tags = self.tag_pos_projection_layer(encoded_text[:, 1:, :])

        class_probabilities_labels = F.softmax(logits_labels, dim=-1).view(
            [batch_size, sequence_length, self.num_labels_classes])  # 利用Softmax函数，将得分转为概率

        class_probabilities_d = F.softmax(logits_d, dim=-1).view(
            [batch_size, sequence_length, self.num_detect_classes])  # 利用Softmax函数，将得分转为概率

        error_probs = class_probabilities_d[:, :,
                      self.incorr_index] * mask  # [batch_size,sen_qen]点乘[batch_size,sen_qen]=[batch_size,sen_qen]，获得每个句子的每个token错误的概率
        incorr_prob = torch.max(error_probs, dim=-1)[
            0]  # [batch_szie]:取每个句子所有token的错误概率最大者，作为此句子的错误概率（用于min_error_probability的trick）

        if self.confidence > 0:  # 给$KEEP标签添加一个置信度bias，优先预测$KEEP，防止模型过多地纠错，属于一个小trick
            probability_change = [self.confidence] + [0] * (self.num_labels_classes - 1)
            offset = torch.FloatTensor(probability_change).repeat(
                (batch_size, sequence_length, 1)).to(self.device)
            class_probabilities_labels += util.move_to_device(offset, self.device)

        # 输出前向传播计算的结果
        output_dict = {"logits_labels": logits_labels,
                       "logits_d_tags": logits_d,
                       "class_probabilities_labels": class_probabilities_labels,
                       "class_probabilities_d_tags": class_probabilities_d,
                       "max_error_probability": incorr_prob}
        # 以上是训练阶段和预测阶段共享的过程

        # bias_keep = 0.5
        # bias_delete = 1e-5
        # bias_UNK = 1e-3
        # 下面时训练阶段独占的过程，因为需要计算loss进行反向传播更新参数
        if labels is not None and d_tags is not None and pos_tags is not None:  # 如果没有提供golden labels标签和d_tags标签，那么就是预测模式，无需计算loss
            # weights = util.move_to_device(torch.ones(mask.shape), 3 if torch.cuda.is_available() else -1) * mask
            # label_mask = labels == 0
            # weights[label_mask] = bias_keep
            # label_mask = labels == 1
            # weights[label_mask] = bias_delete
            # label_mask = labels == self.UNKID
            # weights[label_mask] = bias_UNK
            # crf_loss_labels = -self.crf_layer(logits_labels, labels, mask)  # 基于CRF层的负对数似然损失函数
            # crf_loss_labels = (crf_loss_labels / mask.type_as(crf_loss_labels).sum(-1)).mean()  # 对当前batch取平均
            loss_labels = sequence_cross_entropy_with_logits(logits_labels, labels, mask,
                                                             label_smoothing=self.label_smoothing)
            loss_d = sequence_cross_entropy_with_logits(logits_d, d_tags, mask)
            if self.crf:
                crf_loss_labels = -self.crf_layer(logits_labels, labels, mask)  # 基于CRF层的负对数似然损失函数
                crf_loss_labels = (crf_loss_labels / mask.type_as(crf_loss_labels).sum(-1)).mean()  # 对当前batch取平均
                output_dict[
                    "loss"] = loss_labels + loss_d + self.word_ins_loss_factor * crf_loss_labels  # 采用了一种类似多任务学习的机制，预测d_tags又可以方便地使用min_error_probability的trick
            else:
                output_dict[
                    "loss"] = loss_labels + loss_d  # 采用了一种类似多任务学习的机制，预测d_tags又可以方便地使用min_error_probability的trick
            # output_dict["loss"] = loss_labels + loss_d + self.word_ins_loss_factor * crf_loss_labels  # 采用了一种类似多任务学习的机制，预测d_tags又可以方便地使用min_error_probability的trick
            if pos_tags is not None:
                pos_mask = mask[:, 1:]
                loss_pos_tags = sequence_cross_entropy_with_logits(logits_pos_tags, pos_tags, pos_mask)
                output_dict["loss"] += loss_pos_tags
                # crf_loss_pos_tags = -self.tag_pos_crf_layer(logits_pos_tags, pos_tags, pos_mask)
                # crf_loss_pos_tags = (crf_loss_pos_tags / pos_mask.type_as(crf_loss_pos_tags).sum(-1)).mean()
                # output_dict["loss"] += 0.01 * crf_loss_pos_tags
                # output_dict["pos_tags"] = self.tag_pos_crf_layer.viterbi_tags(logits_pos_tags, pos_mask)
        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        获取模型的评级指标
        :param reset: 是否重置模型的评价指标
        :return: 模型的评级指标
        """
        metrics_to_return = self.metric.get_metric(reset)
        # reset设为True，则会将模型当前累计的评价指标数据清空。一般来说，allennlp的训练器会在每个epoch结束时的那个batch，调用reset为True的get_metrics，以便在下一轮重新计算指标。
        if self.metric is not None and not self.training:
            if reset:
                labels_accuracy = float(metrics_to_return['labels_accuracy'].item())
                print('The accuracy of predicting for edit labels is: ' + str(labels_accuracy))
                labels_except_keep_accuracy = float(metrics_to_return['labels_accuracy_except_keep'].item())
                self.logger.info('The accuracy of predicting for edit labels is: ' + str(labels_accuracy))
                print('The accuracy of predicting for edit labels except keep label is: ' + str(labels_except_keep_accuracy))
                self.logger.info('The accuracy of predicting for edit labels except keep label is: ' + str(labels_except_keep_accuracy))
                tmp_model_dir = "/".join(self.model_dir.split('/')[:-1]) + "/Temp_Model.th"  # 懒得改预测的程序了，直接把当前epoch的模型保存到硬盘上，重新用predict脚本读取作推理
                self.save(tmp_model_dir)
                out = "/".join(self.model_dir.split('/')[:-1]) + "/epoch{:d}_dev_result.out".format(self.epoch)
                if os.path.exists(self.dev_file + ".m2") and os.path.exists(self.dev_file + ".origin"):
                    dev_predict_res = os.popen("/home/zhenghuali/anaconda3/envs/zhangyue/bin/python -u /home/zhenghuali/zhangyue/GECToR_Chinese/predict.py --model_path {:s} --vocab_path {:s} --input_file {:s} --output_file {:s} --cuda_device {:d} --weight_name {:s} --golden_file {:s}".format(tmp_model_dir, self.vocab_path, self.dev_file + ".origin", out, int(self.cuda_device), self.weight_name, self.dev_file + ".m2"))
                    lines = dev_predict_res.readlines()
                    cnt_corrections = int(lines[1].lstrip("Produced overall corrections: ").rstrip('\n'))
                    dev_perplexity = float(lines[2].lstrip("Model perplexity: ").rstrip('\n'))
                    m2_metrics = eval(lines[3].rstrip('\n'))
                    metrics_to_return['dev_m2_precision'] = m2_metrics['Precision']
                    metrics_to_return['dev_m2_recall'] = m2_metrics['Recall']
                    metrics_to_return['dev_m2_f_0.5'] = m2_metrics['F_0.5']
                    metrics_to_return['dev_perplexity'] = dev_perplexity
                    print("This Model correct {:d} errors in dev-set.".format(cnt_corrections))
                    self.logger.info("This Model correct {:d} errors in dev-set.".format(cnt_corrections))
                    print('The dev_perplexity of GEC results in dev-set is: ' + str(dev_perplexity))
                    self.logger.info('The perplexity of GEC results in dev-set is: ' + str(dev_perplexity))
                    print('The m2 score of GEC results in dev-set is: {:f} Precision, {:f} Recall, {:f} F_0.5.'.format(float(metrics_to_return['dev_m2_precision']), float(metrics_to_return['dev_m2_recall']), float(metrics_to_return['dev_m2_f_0.5'])))
                    self.logger.info('The m2 score of GEC results in dev-set is: {:f} Precision, {:f} Recall, {:f} F_0.5.'.format(float(metrics_to_return['dev_m2_precision']), float(metrics_to_return['dev_m2_recall']), float(metrics_to_return['dev_m2_f_0.5'])))
                if self.save_metric == "-dev_perplexity" and os.path.exists(self.dev_file + ".m2") and os.path.exists(self.dev_file + ".origin"):
                    if self.epoch == 0 or self.best_metric >= dev_perplexity:
                        print('(best)Saving Model...')
                        self.logger.info('(best)Saving Model...')
                        self.best_metric = dev_perplexity
                        self.save(self.model_dir)
                    print('best dev_perplexity till now:' + str(self.best_metric))
                    self.logger.info('best dev_perplexity till now:' + str(self.best_metric))
                elif self.save_metric == "+labels_accuracy":
                    if self.best_metric <= labels_accuracy:
                        print('(best)Saving Model...')
                        self.logger.info('(best)Saving Model...')
                        self.best_metric = labels_accuracy
                        self.save(self.model_dir)
                    print('best labels_accuracy till now:' + str(self.best_metric))
                    self.logger.info('best labels_accuracy till now:' + str(self.best_metric))
                elif self.save_metric == "+dev_m2_f_0.5" and os.path.exists(self.dev_file + ".m2") and os.path.exists(self.dev_file + ".origin"):
                    if self.best_metric <= metrics_to_return['dev_m2_f_0.5']:
                        print('(best)Saving Model...')
                        self.logger.info('(best)Saving Model...')
                        self.best_metric = metrics_to_return['dev_m2_f_0.5']
                        self.save(self.model_dir)
                    print('best dev_m2_f_0.5 till now:' + str(self.best_metric))
                    self.logger.info('best best dev_m2_f_0.5 till now:' + str(self.best_metric))
                else:
                    raise Exception("Wrong metric!")
                self.epoch += 1
                print(f'\nepoch: {self.epoch}')
                self.logger.info(f'epoch: {self.epoch}')

        return metrics_to_return

    def save(self, model_dir):
        """
        保存模型
        :param model_dir: 模型保存文件夹
        """
        with open(model_dir, 'wb') as f:
            torch.save(self.state_dict(), f)
        print("Model is dumped")
        self.logger.info("Model is dumped")
