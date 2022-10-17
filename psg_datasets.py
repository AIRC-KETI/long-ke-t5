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
principle sentence generation loading script
"""

FILE_PATTERN = "train*.jsonl*"


def generator(fpath_list):
    for fpath in fpath_list:
        with open(fpath, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                yield {
                    "text": item["text"],
                    "content-length": item["content-length"] if isinstance(item["content-length"], int) else len(" ".join(item["content-length"])),
                    "content-type": item["content-type"],
                    "timestamp": item["timestamp"],
                    "url": item["url"],
                    "order": item["order"],
                    "lead": item["lead"],
                    "random": item["random"],
                    "ind-orig": item["ind-orig"],
                    "ind-uniq": item["ind-uniq"],
                    "seq-orig": item["seq-orig"],
                    "seq-uniq": item["seq-uniq"],
                }

        
class PrincipleSentenceGenerationDataset(datasets.GeneratorBasedBuilder):
    """Principle Sentence Generation Dataset"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="base",
            version=_VERSION,
            description="Principle Sentence Generation Dataset",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Sequence(datasets.Value("string")),
                    "content-length": datasets.Value("int32"),
                    "content-type": datasets.Value("string"),
                    "timestamp": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "order": datasets.Value("int32"),
                    "lead": datasets.Sequence(datasets.Value("int32")),
                    "random": datasets.Sequence(datasets.Value("int32")),
                    "ind-orig": datasets.Sequence(datasets.Value("int32")),
                    "ind-uniq": datasets.Sequence(datasets.Value("int32")),
                    "seq-orig": datasets.Sequence(datasets.Value("int32")),
                    "seq-uniq": datasets.Sequence(datasets.Value("int32")),
                }
            ),
            supervised_keys=None,  # Probably needs to be fixed.
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):

        path_kv = {
            datasets.Split.TRAIN: glob.glob(os.path.join(dl_manager.manual_dir, FILE_PATTERN))
        }

        return [
                datasets.SplitGenerator(name=k, gen_kwargs={'fpath': v}) for k, v in path_kv.items()
        ]

    def _generate_examples(self, fpath):
        """Yields examples."""
        for idx, item in enumerate(generator(fpath)):
            yield idx, item