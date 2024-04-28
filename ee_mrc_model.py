from transformers import BertForQuestionAnswering, BertPreTrainedModel, BertModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

class EventExtractionModelOutput(QuestionAnsweringModelOutput):
    def __init__(self, loss=None, start_logits=None, end_logits=None, logits=None, hidden_states=None, attentions=None, event_loss=None, MRC_loss=None):
        self.loss=loss
        self.start_logits=start_logits
        self.end_logits=end_logits
        self.logits=logits
        self.hidden_states=hidden_states
        self.attentions=attentions
        self.event_loss=event_loss
        self.MRC_loss=MRC_loss




class BertForEventExtractionModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_event = config.num_event

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_event)

        self.loss_weight = config.loss_weight
        self.attention = config.attention
        if self.attention:
            self.attention_weight1 = nn.Linear(config.num_event, config.attention_size)
            self.attention_weight2 = nn.Linear(config.hidden_size + config.attention_size, config.hidden_size)

        

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        labels = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]

        # print("sequence_output size",sequence_output.size())
        

        ## classification logits
        pooled_output,_ = torch.max(sequence_output, dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        ## attention from logits
        if self.attention:
            ## logits [B, C] -> logits_attention [B, H] ; B:batch_size, C: class_number, H: hidden_size
            logits_attention = self.attention_weight1(logits)
            # print("logits_attention ", logits_attention.size())
            ## logits_attention [B, H] -> [B, 1, H] -> [B, L, H] ; L: sequence_length
            logits_attention = logits_attention.unsqueeze(1).repeat([1,sequence_output.size(1),1])
            # print("logits_attention ", logits_attention.size())
            ## logits_attention [B, L, H] -> [B, L, 2H] 
            logits_attention = torch.cat([sequence_output, logits_attention], dim=-1)
            # print("logits_attention ", logits_attention.size())
            ## logits_attention [B, L, 2H] -> [B, L, H]  substitute original sequence_output
            sequence_output = self.attention_weight2(logits_attention)
            # print("sequence_output ", sequence_output.size())

        ## reading comprehension logits
        logits2 = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits2.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)


        total_loss = None
        event_loss = None
        MRC_loss = None

        device = 'cuda'
        ## classification loss
        if labels is not None:
            # print("labels", labels)
            # print("start", start_positions)
            # print("end", end_positions)
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                event_loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # original 
                # loss_fct = CrossEntropyLoss()
                # event_loss = loss_fct(logits.view(-1, self.num_event), labels.view(-1))
                
                # new
                # if there is no event, change the event loss to zero
                loss_fct = CrossEntropyLoss(reduction='none')
                event_loss = loss_fct(logits.view(-1, self.num_event), labels.view(-1))
                mask = (start_positions>0).float()
                # print("event loss", event_loss)
                # print("mask", mask)
                # mask = mask
                # print("mask", mask)
                event_loss = torch.sum(event_loss*mask)
            total_loss = event_loss


        ## MRC loss
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # lower down the weight of event classification
            weight = torch.ones(ignored_index)
            weight[0] = weight[0] / self.loss_weight
            # weight.to(start_logits)
            # print(weight)
            weight = weight.to(device)

            loss_fct = CrossEntropyLoss(weight=weight, ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            MRC_loss = (start_loss + end_loss) / 2
            if total_loss is None:
                total_loss = MRC_loss
            else:
                total_loss = total_loss + MRC_loss


        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        # print("MRC loss",MRC_loss)
        # print("event_loss", event_loss)
        # print("total loss", total_loss)
        
        return EventExtractionModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            event_loss=event_loss,
            MRC_loss=MRC_loss,
        )