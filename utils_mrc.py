import logging
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm
from example_freature import MRCFeatures, is_whitespace, is_chinese_char

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


def eliminate_whitespace_in_chinese(text):
    new_text = ""
    for i, char in enumerate(text):
        if char == " ":
            if is_chinese_char(text[max(0,i-1)]) or is_chinese_char(text[min(i+1,len(text)-1)]):
                continue
            new_text += char
        else:
            new_text += char
    return new_text

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)
    # logger.info("answer not improved origin=%s \t answer=%s \t old=%s", orig_answer_text, tok_answer_text, " ".join(doc_tokens[input_start: (input_end + 1)]))
    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def mrc_convert_example_to_features(example, tokenizer, max_seq_length, doc_stride, max_query_length, is_training):
    features = []
    labels = None
    if is_training:
        labels = example.labels

    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

       
        # If the answer cannot be found in the text, then skip this example.
        actual_text = "".join(example.doc_tokens[start_position: (end_position + 1)])#.replace(" ","")
        cleaned_answer_text = "".join(example.answer_text.strip().split())
        if actual_text.find(cleaned_answer_text) == -1:
            print(example.doc_tokens[start_position-2 : (end_position + 1)])
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    
    # print(all_doc_tokens)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1


        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )
        # if tok_start_position != old_start or tok_end_position != old_end:
        #     print("haha")
        #     print(old_start, old_end, tok_start_position, tok_end_position)
        #     print(example.answer_text)
        #     print(all_doc_tokens[old_start:old_end+1])
        #     print(all_doc_tokens[tok_start_position:tok_end_position+1])
        #     print()
        # span_answer = "".join(all_doc_tokens[tok_start_position:tok_end_position+1]).replace("##","")
        # if example.answer_text.lower() != span_answer:
        #     print(example.answer_text.lower())
        #     print(span_answer)
        #     print()
            
    spans = []

    truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length, truncation=True)
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer))
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    # print("sequence_added_tokens",sequence_added_tokens)
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
    # print("sequence_pair_added_tokens",sequence_pair_added_tokens)

    # span_doc_tokens = all_doc_tokens

    token_spans = []
    rest_length = max_seq_length - len(truncated_query) - sequence_pair_added_tokens
    for i in range(0, len(all_doc_tokens), doc_stride):
        token_spans.append(all_doc_tokens[i:i+rest_length])
        if i+rest_length >= len(all_doc_tokens):
            break

    # while len(spans) * doc_stride < len(all_doc_tokens):
    for span_doc_tokens in token_spans:
        # print(len(spans), doc_stride, len(all_doc_tokens))
        # print(truncated_query, span_doc_tokens)
        if len(span_doc_tokens) == 0:
            break
        encoded_dict = tokenizer.encode_plus(
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            padding='max_length',
            # stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            # stride=doc_stride,
            # truncation_strategy=
            truncation="only_second" if tokenizer.padding_side == "right" else "only_first",
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )
        # print(encoded_dict["input_ids"])
        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            # print("haha")
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict:
            break
        # print()
        # print(span_doc_tokens)
        # print(len(span_doc_tokens))
        # span_doc_tokens = encoded_dict["overflowing_tokens"]
        # print(span_doc_tokens)
        # print()
        
    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(span["token_type_ids"])

        p_mask = np.minimum(p_mask, 1)

        if tokenizer.padding_side == "right":
            # Limit positive values to one
            p_mask = 1 - p_mask

        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

        # Set the CLS index to '0'
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False
            # print(doc_start, doc_end)

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
                assert span["tokens"][doc_offset-1] == '[SEP]'
                # print(tok_start_position, start_position, doc_start, doc_offset)
                # if all_doc_tokens[tok_start_position] != span["tokens"][start_position]:
                    
                #     print(all_doc_tokens[tok_start_position])
                #     print(span["tokens"][start_position])
                #     print(span["tokens"][start_position-20:start_position+20])
        # if is_training and start_position != 0:
        #     actual_text = example.answer_text.replace(" ", "")
        #     actual_text22 = "".join(all_doc_tokens[tok_start_position: (tok_end_position + 1)]).lower()
        #     span_answer = span["tokens"][start_position:end_position+1]
        #     span_answer = "".join(span_answer).replace("##", "")
        #     if actual_text.lower() != span_answer:
        #         print(actual_text)
        #         print(actual_text22)
        #         print(span_answer)
        #         print()

        features.append(
            MRCFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                labels=labels,
            )
        )
    return features


def mrc_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_features(features, is_training):
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in features:
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

    if not is_training:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask
        )
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids, all_attention_masks, all_token_type_ids, all_start_positions, all_end_positions, all_cls_index, all_p_mask, all_is_impossible, all_labels
        )
    return features, dataset


def mrc_convert_examples_to_features_multiprocessing(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, threads=1):
    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=mrc_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            mrc_convert_example_to_features,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="Convert Features",
            )
        )
    
    features, dataset = convert_features(features, is_training)
    return features, dataset
    

def mrc_convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, threads=1):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~mrcExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`mrcFeatures`
    """
    if threads > 1:
        return mrc_convert_examples_to_features_multiprocessing(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, threads)
    features = []
    for example in tqdm(examples, desc="Convert Features"):
        feature = mrc_convert_example_to_features(example, tokenizer, max_seq_length, doc_stride, max_query_length, is_training)
        features.append(feature)

    features, dataset = convert_features(features, is_training)

    return features, dataset
