import os
import warnings
from shutil import copyfile
from typing import List, Optional, Tuple

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import is_sentencepiece_available, logging

from transformers import T5Tokenizer, T5TokenizerFast, PreTrainedTokenizerFast, AutoTokenizer

logger = logging.get_logger(__name__)

class T5TokenizerForQA(T5Tokenizer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.cls_token_id = self.additional_special_tokens_ids[0]
    
    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
        cls = [self.cls_token_id]
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return cls + token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return cls + token_ids_0 + token_ids_1
    
    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + eos) * [0]
        return len(cls + token_ids_0 + eos) * [0] + len(token_ids_1 + eos) * [1]

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
        ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

from transformers.convert_slow_tokenizer import SpmConverter
from tokenizers import processors
from tokenizers import Tokenizer as TokenizerFast

class T5QAConverter(SpmConverter):
    def vocab(self, proto):
        num_extra_ids = self.original_tokenizer._extra_ids
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        vocab += [(f"<extra_id_{i}>", 0.0) for i in range(num_extra_ids - 1, -1, -1)]
        return vocab

    def post_processor(self):
        cls = str(self.original_tokenizer.cls_token)
        eos = str(self.original_tokenizer.eos_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        eos_token_id = self.original_tokenizer.eos_token_id

        return processors.TemplateProcessing(
            single=f"{cls} $A {eos}",
            pair=f"{cls} $A {eos} $B:1 {eos}:1",
            special_tokens=[
                (cls, cls_token_id),
                (eos, eos_token_id),
            ],
        )


class T5TokenizerFastForQA(T5TokenizerFast):
    def __init__(self, *args, **kwargs) -> None:

        slow_tokenizer = kwargs.pop("__slow_tokenizer", None)
        fast_tokenizer_file = kwargs.pop("tokenizer_file", None)

        fast_tokenizer = None

        if fast_tokenizer_file is not None:
            fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)

            cls = "<extra_id_0>"
            eos = "</s>"
            fast_tokenizer.cls_token_id = fast_tokenizer.token_to_id(cls)
            fast_tokenizer.eos_token_id = fast_tokenizer.token_to_id(eos)
            
            fast_tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{cls} $A {eos}",
                pair=f"{cls} $A {eos} $B:1 {eos}:1",
                special_tokens=[
                    (cls, fast_tokenizer.cls_token_id),
                    (eos, fast_tokenizer.eos_token_id),
                ],
            )
        elif self.slow_tokenizer_class is not None and (slow_tokenizer is None and fast_tokenizer is None):
            # We need to create and convert a slow tokenizer to build the backend
            slow_tokenizer = self.slow_tokenizer_class(*args, **kwargs)

        if slow_tokenizer is not None:
            slow_tokenizer.cls_token_id = slow_tokenizer.additional_special_tokens_ids[0]
            slow_tokenizer.cls_token = "<extra_id_0>"
            fast_tokenizer = T5QAConverter(slow_tokenizer).converted()
        
        assert fast_tokenizer is not None, "fast_tokenizer is None..."
        super().__init__(tokenizer_object=fast_tokenizer)

        self.cls_token = "<extra_id_0>"
        self.cls_token_id = fast_tokenizer.token_to_id(self.cls_token)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
        cls = [self.cls_token_id]

        token_ids_0 = cls + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0
        else:
            token_ids_1 = token_ids_1 + [self.eos_token_id]
            return self.prefix_tokens + token_ids_0 + token_ids_1
    
    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + eos) * [0]
        return len(cls + token_ids_0 + eos) * [0] + len(token_ids_1 + eos) * [1]

if __name__=="__main__":
    # tokenizer_name = "KETI-AIR/ke-t5-base"
    tokenizer_name = "../../small_model"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(type(tokenizer))
    print(tokenizer.cls_token_id)

    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name)
    print(type(tokenizer))
    print(tokenizer.cls_token_id)

    tokenizer = T5TokenizerForQA.from_pretrained(tokenizer_name)
    print(type(tokenizer))
    print(tokenizer(
            "Whether or not the token list is already formatted with special",
            "You are instantiating a new tokenizer from scratch. This is not supported by",
        ))
    print(tokenizer.cls_token_id)

    tokenizer = T5TokenizerFastForQA.from_pretrained(tokenizer_name)
    print(type(tokenizer))
    print(tokenizer(
            "Whether or not the token list is already formatted with special",
            "You are instantiating a new tokenizer from scratch. This is not supported by",
        ))
    tokenized_examples = tokenizer(
            "Whether or not the token list is already formatted with special",
            "You are instantiating a new tokenizer from scratch. This is not supported by",
        )
    print(tokenizer.cls_token_id)

    sequence_ids = tokenized_examples.sequence_ids(0)
    print(sequence_ids)

    tokenizer.save_pretrained("tok")


    # from transformers import BertTokenizerFast

    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    # tokenizer.save_pretrained("tok_bert")

    