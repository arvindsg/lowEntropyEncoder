import logging
from typing import Any, Dict, List, Optional

import torch
from torch import sigmoid

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder,Seq2VecEncoder,  SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
import math

@Model.register("sequenceAutoEncoder")
class SequenceAutoEncoder(Model):
    def __init__(self, sequenceEmbedder) -> None:
        super(BidirectionalAttentionFlowWithNoAnswerOption, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
                                                      num_highway_layers))
        self._phrase_layer = phrase_layer
        self._matrix_attention = LegacyMatrixAttention(attention_similarity_function)
        self._modeling_layer = modeling_layer
        self._span_end_encoder = span_end_encoder

        encoding_dim = phrase_layer.get_output_dim()
        modeling_dim = modeling_layer.get_output_dim()
        span_start_input_dim = encoding_dim * 4 + modeling_dim
        self._span_start_predictor = TimeDistributed(torch.nn.Linear(span_start_input_dim, 1))

        span_end_encoding_dim = span_end_encoder.get_output_dim()
        span_end_input_dim = encoding_dim * 4 + span_end_encoding_dim
        self._span_end_predictor = TimeDistributed(torch.nn.Linear(span_end_input_dim, 1))
        
        
        # Bidaf has lots of layer dimensions which need to match up - these aren't necessarily
        # obvious from the configuration files, so we check here.
        check_dimensions_match(modeling_layer.get_input_dim(), 4 * encoding_dim,
                               "modeling layer input dim", "4 * encoding dim")
        check_dimensions_match(text_field_embedder.get_output_dim(), phrase_layer.get_input_dim(),
                               "text field embedder output dim", "phrase layer input dim")
        check_dimensions_match(span_end_encoder.get_input_dim(), 4 * encoding_dim + 3 * modeling_dim,
                               "span end encoder input dim", "4 * encoding dim + 3 * modeling dim")
        self._answer_impossible_accuracy=BooleanAccuracy()
        self._span_start_accuracy = BooleanAccuracy()
        self._span_end_accuracy = BooleanAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                answer_impossible:torch.LongTensor = None,
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.

        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        embedded_question = self._highway_layer(self._text_field_embedder(question))
        embedded_passage = self._highway_layer(self._text_field_embedder(passage))
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.last_dim_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        modeled_passage = self._dropout(self._modeling_layer(final_merged_passage, passage_lstm_mask))
        modeling_dim = modeled_passage.size(-1)

        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
        span_start_input = self._dropout(torch.cat([final_merged_passage, modeled_passage], dim=-1))
        # Shape: (batch_size, passage_length)
        span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)
        # Shape: (batch_size, passage_length)
        span_start_probs = sigmoid(span_start_logits)

        # Shape: (batch_size, modeling_dim)
        span_start_representation = util.weighted_sum(modeled_passage, span_start_probs)
        # Shape: (batch_size, passage_length, modeling_dim)
        tiled_start_representation = span_start_representation.unsqueeze(1).expand(batch_size,
                                                                                   passage_length,
                                                                                   modeling_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim * 3)
        span_end_representation = torch.cat([final_merged_passage,
                                             modeled_passage,
                                             tiled_start_representation,
                                             modeled_passage * tiled_start_representation],
                                            dim=-1)
        # Shape: (batch_size, passage_length, encoding_dim)
        encoded_span_end = self._dropout(self._span_end_encoder(span_end_representation,
                                                                passage_lstm_mask))
        # Shape: (batch_size, passage_length, encoding_dim * 4 + span_end_encoding_dim)
        span_end_input = self._dropout(torch.cat([final_merged_passage, encoded_span_end], dim=-1))
        span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)
        
        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)
        span_start_probs = sigmoid(span_start_logits)
        span_end_probs = sigmoid(span_end_logits)
        best_span = self.get_best_span(span_start_probs,span_end_probs)
        
        
        output_dict = {
                "passage_question_attention": passage_question_attention,
                "span_start_logits": span_start_logits,
                "span_start_probs": span_start_probs,
                "span_end_logits": span_end_logits,
                "span_end_probs": span_end_probs,
                "best_span": best_span,
                
                }

        # Compute the loss for training.
        if answer_impossible is not None:
            
            target_start=torch.arange(0,span_start_logits.size(1),device=span_start_logits.device,dtype=torch.long)
            target_start=target_start.squeeze(0).expand(span_start_logits.size(0),-1)==span_start
            target_start=target_start.long()*(-1*(answer_impossible-1).unsqueeze(1).expand(-1,target_start.size(-1)))
            
            target_end=torch.arange(0,span_end_logits.size(1),device=span_end_logits.device,dtype=torch.long)
            target_end=target_end.squeeze(0).expand(span_end_logits.size(0),-1)==span_end
            target_end=target_end.long()*(-1*(answer_impossible-1).unsqueeze(1).expand(-1,target_start.size(-1)))
            
            span_start_logits_for_loss=torch.stack([-1*span_start_logits,span_start_logits],dim=-1)
            
            loss = util.sequence_cross_entropy_with_logits(span_start_logits_for_loss,target_start, passage_mask)
            
            span_end_logits_for_loss=torch.stack([-1*span_end_logits,span_end_logits],dim=-1)
            loss += util.sequence_cross_entropy_with_logits(span_end_logits_for_loss,target_end, passage_mask)
                
            
            
            
            self._span_start_accuracy((span_start_logits>0).long(), target_start)
            self._span_end_accuracy((span_end_logits>0).long(), target_end)
            self._answer_impossible_accuracy(((best_span.narrow(1,0, 1)==-1)*(best_span.narrow(1,1, 1)==-1)).long(), answer_impossible)
#             self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
            output_dict["loss"] = loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        
        if metadata is not None:
            output_dict['best_span_str'] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                try:
                    if predicted_span[0]!=-1:
                        start_offset = offsets[predicted_span[0]][0]
                    else:
                        start_offset=-1
                    if predicted_span[1]!=-1:
                        end_offset = offsets[predicted_span[1]][1]
                    else:
                        end_offset=-1
                    if end_offset!=-1 and start_offset!=-1:
                        best_span_string = passage_str[start_offset:end_offset]
                    else:
                        best_span_string=""
                    output_dict['best_span_str'].append(best_span_string)
                    answer_texts = metadata[i].get('answer_texts', [])
                    if answer_texts:
                        self._squad_metrics(best_span_string, answer_texts)
                except Exception as e:
                    print(str(e))    
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
                'start_acc': self._span_start_accuracy.get_metric(reset),
                'end_acc': self._span_end_accuracy.get_metric(reset),
                'answer_impossible_accuracy':self._answer_impossible_accuracy.get_metric(reset),
#                 'span_acc': self._span_accuracy.get_metric(reset),
                'em': exact_match,
#                 'f1': f1_score,
                }

    @staticmethod
    def get_best_span(span_start_probs: torch.Tensor, span_end_probs: torch.Tensor) -> torch.Tensor:
        if span_start_probs.dim() != 2 or span_end_probs.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_probs.size()

        max_span_prob = [.25] * batch_size
        batch_size, passage_length = span_start_probs.size()
        span_start_argmax = [0] * batch_size
        best_word_span = span_start_probs.new_zeros((batch_size, 2), dtype=torch.long)-1
        
        span_start_probs = span_start_probs.detach().cpu().numpy()
        span_end_probs = span_end_probs.detach().cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_probs[b, span_start_argmax[b]]
                if val1 < span_start_probs[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_probs[b, j]

                val2 = span_end_probs[b, j]

                if val1 * val2 > max_span_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_prob[b] = val1 + val2
        return best_word_span
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BidirectionalAttentionFlow':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        num_highway_layers = params.pop_int("num_highway_layers")
        phrase_layer = Seq2SeqEncoder.from_params(params.pop("phrase_layer"))
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        modeling_layer = Seq2SeqEncoder.from_params(params.pop("modeling_layer"))
        span_end_encoder = Seq2SeqEncoder.from_params(params.pop("span_end_encoder"))
        dropout = params.pop_float('dropout', 0.2)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        mask_lstms = params.pop_bool('mask_lstms', True)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   num_highway_layers=num_highway_layers,
                   phrase_layer=phrase_layer,
                   attention_similarity_function=similarity_function,
                   modeling_layer=modeling_layer,
                   span_end_encoder=span_end_encoder,
                   dropout=dropout,
                   mask_lstms=mask_lstms,
                   initializer=initializer,
                   regularizer=regularizer)