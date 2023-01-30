# Copyright 2022 san kim
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import json

import datasets

_VERSION = datasets.Version("1.0.0", "")

_URL = ""

_CITATION = """\
There is no citation information
"""

_DESCRIPTION = """\
(batched datasets)
principle sentence generation loading script
"""

TRAIN_FILE_PATTERN = "train.jsonl"
VALIDATION_FILE_PATTERN = "validation.jsonl"

def generator(fpath_list):
    for fpath in fpath_list:
        with open(fpath, "r") as f:
            for line in f:
                item = json.loads(line)
                yield item

        
class PrincipleSentenceGenerationBatchedDataset(datasets.GeneratorBasedBuilder):
    """Principle Sentence Generation Dataset"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="base",
            version=_VERSION,
            description="Principle Sentence Generation Batched Dataset",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "input_ids": datasets.Sequence(datasets.Sequence(datasets.Value("int32"))),
                    "attention_mask": datasets.Sequence(datasets.Sequence(datasets.Value("int32"))),
                    "labels": datasets.Sequence(datasets.Sequence(datasets.Value("int32"))),
                    "decoder_attention_mask": datasets.Sequence(datasets.Sequence(datasets.Value("int32"))),
                }
            ),
            supervised_keys=None,  # Probably needs to be fixed.
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):

        path_kv = {
            datasets.Split.TRAIN: glob.glob(os.path.join(dl_manager.manual_dir, TRAIN_FILE_PATTERN)),
            datasets.Split.VALIDATION: glob.glob(os.path.join(dl_manager.manual_dir, VALIDATION_FILE_PATTERN)),
        }

        return [
                datasets.SplitGenerator(name=k, gen_kwargs={'fpath': v}) for k, v in path_kv.items()
        ]

    def _generate_examples(self, fpath):
        """Yields examples."""
        for idx, item in enumerate(generator(fpath)):
            yield idx, item