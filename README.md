# long-ke-t5

**long-ke-t5**는 한국어와 영어 비정형 데이터를 이용하여 학습시킨 사전학습모델입니다.
[Pegasus](https://arxiv.org/abs/1912.08777)에서 사용된 PSG (Principle Sentence Generation)을 이용하여 모델을 학습시켰으며, prinsiple sentence 선정 방식은 **ind-uniq**를 사용하였고, 한 sample document에 N개의 문장이 존재할때 N * 0.2 만큼의 문장을 principle sentence로 선정하였습니다.
**long-ke-t5**는 학습간에 encoder의 최대 입력 길이 4K 토큰으로, decoder의 최대 입력 길이 1K 토큰으로 제한을 걸어 학습되었습니다.
따라서 **long-ke-t5**에서 지원하는 최대 입력 길이는 encoder 입력: 4096, decoder 입력: 1024입니다.
자세한 내용은 [Long-T5](https://arxiv.org/abs/2112.07916) 논문을 참고하시길 바랍니다.



## 학습에 사용된 데이터양

### 언어 별 문서 수

| language | # of docs |
| --- | --- |
| ko | 36,052,644 |
| en | 38,168,031 |

학습에 사용된 총 토큰 수는 35,105,474,195 tokens 입니다.

<table>
<tr>
<td>한국어 문서 분포</td>
<td>영어 문서 분포</td>
</tr>

<tr>
<td>
<img height="400" width="400"
            src="https://github.com/AIRC-KETI/long-ke-t5/blob/main/images/ko_doc_dist.png?raw=true"
            alt="score: 183.50">
</td>
<td>
<img height="400" width="400"
            src="https://github.com/AIRC-KETI/long-ke-t5/blob/main/images/en_doc_dist.png?raw=true"
            alt="score: 183.50">
</td>
</tr>


</table>

## 사전학습 모델

| model name | Huggingface |
| --- | --- |
| long-ke-t5-small | `KETI-AIR/long-ke-t5-small` |
| long-ke-t5-base | `KETI-AIR/long-ke-t5-base` |


## Downstream 모델

| model name | Huggingface | prefix | description |
| --- | --- | --- | --- |
| translation | `KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-bidirection` | `translate_en2ko: ` or `translate_ko2en` | AI hub translation dataset을 활용한 모델 (1 line) |
| translation | `KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-ko2en` | `translate_ko2en` | Korean2English (1 line) |
| translation | `KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-en2ko` | `translate_en2ko: ` | English2Korean (1 line) |
| translation | `KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-bidirection_e1` | `translate_en2ko: ` or `translate_ko2en` | AI hub translation dataset의 여러 병렬 코퍼스를 이어붙여 학습한 모델 (N line) |
| summarization | `KETI-AIR-Downstream/long-ke-t5-base-summarization` | `summarization-num_lines-{N}: `| AI hub summarization dataset을 학습한 모델 |
| summarization | `KETI-AIR-Downstream/long-ke-t5-base-summarization_e10` | `summarization-num_lines-{N}: `| AI hub summarization dataset을 학습한 모델 |


### Translation 모델 사용 예시

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load the model and tokenizer
model_path = "KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-bidirection"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# korean to english
source = """translate_ko2en: IBM 왓슨X는 AI 및 데이터 플랫폼이다. 신뢰할 수 있는 데이터, 속도, 거버넌스를 갖고 파운데이션 모델 및 머신 러닝 기능을 포함한 AI 모델을 학습시키고, 조정해, 조직 전체에서 활용하기 위한 전 과정을 아우르는 기술과 서비스를 제공한다."""

input_ids = tokenizer(source, return_tensors="pt").input_ids
gen_seq = model.generate(
    input_ids,
    num_beams=4,
    max_length=1024
)
print(tokenizer.decode(gen_seq[0], skip_special_tokens=True))
# IBM WatsonX is an AI and data platform. With reliable data, speed, and governance, it learns, adjusts, and provides technology and services that cover the entire process for use throughout the organization by training and coordinating AI models, including foundation models and machine learning capabilities.

# english to korean
source = """translate_en2ko: The Seoul Metropolitan Government said Wednesday that it would develop an AI-based congestion monitoring system to provide better information to passengers about crowd density at each subway station."""

input_ids = tokenizer(source, return_tensors="pt").input_ids
gen_seq = model.generate(
    input_ids,
    num_beams=4,
    max_length=1024
)
print(tokenizer.decode(gen_seq[0], skip_special_tokens=True))
# 서울시는 지하철 역별 군중 밀집도에 대해 승객들에게 더 나은 정보를 제공하기 위해 AI 기반 혼잡 모니터링 시스템을 개발할 것이라고 수요일 밝혔다.
```


### Summarization 모델 사용 예시

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load the model and tokenizer
model_path = "KETI-AIR-Downstream/long-ke-t5-base-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

prefix = "summarization-num_lines-{}: "
article = """
'역사 속에 잠들어 있던 포니 쿠페가 화려하게 부활하다'

현대자동차는 18일(현지 시간) 이탈리아 레이크 코모에서 개최된 '현대 리유니온' 행사에서 '포니 쿠페 콘셉트' 복원 모델을 세계에 첫 공개했습니다. 이 프로젝트는 현대차의 창업자인 정주영 선대 회장의 수출보국(輸出報國) 정신과 포니 쿠페를 통한 글로벌 브랜드 정립에 대한 끊임없는 열정과 도전 정신을 재조명하기 위한 것입니다. 현대차에 따르면, 이번 현대 리유니온 행사는 회사의 역사를 다시 돌아보며 변하지 않는 미래 지향적인 비전과 방향성을 공유하는 브랜드 유산 행사입니다.
현대차의 주요 전현직 임직원들이 참석한 가운데 진행된 이번 행사에서 정의선 회장은 "1970년대 기술적 한계에도 불구하고 우리나라가 완벽하게 자동차를 생산할 수 있음을 증명하고 심지어 항공기까지 생산할 수 있음을 믿는 정주영 선대 회장의 비전이 실현되었다"며 "현재의 역사적 성과는 정주영 선대 회장, 정세영 회장, 그리고 정몽구 명예 회장을 비롯한 모든 현대차 직원들의 노력으로 이루어진 결과"라고 강조했습니다.

정주영 선대 회장은 정 회장의 할아버지로, 자동차를 국가의 주요 수출품으로 발전시키려는 비전을 가지고 있었습니다.
'포니 쿠페 콘셉트' 복원 프로젝트를 처음으로 공개한 이번 프로젝트는 이탈리아의 유명 디자이너 조르제토 주지아로와 그의 아들 파브리지오 주지아로와의 협업을 통해 진행됐습니다. 조르제토 주지아로는 1970년대 포니를 처음 디자인한 디자이너입니다.

포니는 1975년부터 1990년까지 현대차가 생산한 소형차로, 국내 최초의 독자적인 자동차 모델입니다. 1974년에 현대차가 이탈리아 토리노 모터쇼에서 첫번째 독자 모델인 포니와 함께 선보인 포니 쿠페 콘셉트는 그 당시 세계적인 주목을 받았습니다. 특히 쐐기 모양의 노즈, 원형 헤드램프, 그리고 종이 접기를 연상시키는 기하학적 선들이 독특한 디자인 요소로 평가 받았습니다.

당시 양산 단계까지 거의 진행됐지만, 1979년 석유 파동과 세계 경제 침체 등으로 양산에는 이르지 못했습니다. 이후 자연 재해로 도면과 차량이 손실되면서 역사 속으로 사라졌습니다.
포니를 통해 해외 시장을 개척한 현대차는 1985년에 세계 최대 자동차 시장인 미국에 진출했습니다, 이후 스텔라, 포니 엑셀, 프레스토 등 다양한 모델을 전 세계로 수출하며 글로벌 브랜드로서의 입지를 확고히 했습니다.

현대차는 포니 쿠페 콘셉트에 담긴 혁신적인 도전 정신이 현대차의 선진국 진출과 스포츠카 분야 도전에 큰 도움이 되었으며, 이 모델이 계속해서 창의적인 영감을 제공하고 있다고 설명했습니다.

포니 쿠페 콘셉트 복원 모델이 공개되기 전 외부에 전시된 `N 비전 74`도 관심을 끌었습니다. N 비전 74는 포니 쿠페 콘셉트에서 영감을 받아 만들어진 고성능 N 브랜드의 스포츠카로, 배터리 모터와 수소연료전지를 결합한 하이브리드 시스템으로 개발됐습니다.
"""
# article 출처
#   - press: YTM
#   - author: 김재형 기자

input_text = prefix.format(1) + article
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
gen_seq = model.generate(
    input_ids,
    num_beams=4,
    max_length=1024
)
print(tokenizer.decode(gen_seq[0], skip_special_tokens=True))
# 현대자동차는 정 선대 회장의 수출보국 정신과 포니 쿠페를 통한 글로벌 브랜드 정립에 대한 끊임없는 열정과 도전 정신을 재조명하기 위해 현대 리유니온 행사에서 포니 쿠페 콘셉트 복원 모델을 세계에 첫 공개했다.

input_text = prefix.format(4) + article
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
gen_seq = model.generate(
    input_ids,
    max_length=1024
)
print(tokenizer.decode(gen_seq[0], skip_special_tokens=True))
# 현대자동차는 현대 리유니온 행사에서 포니 쿠페 콘셉트 복원 모델을 세계에 첫 공개했다. 이 프로젝트는 정 선대 회장의 수출보국 정신과 포니 쿠페를 통한 글로벌 브랜드 정립에 대한 끊임없는 열정과 도전 정신을 재조명하기 위한 것이다. 이번 행사는 회사의 역사를 다시 돌아보며 변하지 않는 미래 지향적인 비전과 방향성을 공유하는 브랜드 유산 행사다. 포니쿠프 콘셉트 복원 모델이 공개되기 전 외부에 전시된 N 비전 74도 관심을 끌었다.
```

## 정량적 성능

### Summarization

| Dataset | Category | Model | Rouge 1 | Rouge 2 | Rouge L(LCS) | Rouge L summary |
| --- | --- | --- | --- | --- | --- | --- |
| cnn dailymail |  | long-ke-t5-small | 40.2415 | 18.4701 | 28.0805 | 37.1094 |
|  |  | ke-t5-small | 34.0791 | 14.8677 | 25.6886 | 31.3533 |
|  |  | long-ke-t5-small-b | 41.5118 | 19.3257 | 28.8733 | 38.3584 |
|  |  | long-ke-t5-base | 43.0111 | 20.5549 | 30.1687 | 39.8946 |
| arXiv |  | long-ke-t5-small | 36.7894 | 14.7789 | 22.6219 | 32.5735 |
|  |  | ke-t5-small | 20.3037 | 4.2265 | 14.9102 | 18.822 |
|  |  | long-ke-t5-small-b | 26.0701 | 7.5479 | 17.6685 | 22.4715 |
|  |  | long-ke-t5-base |  |  |  |  |
| multi news |  | long-ke-t5-small | 38.9107 | 14.6994 | 21.778 | 35.2766 |
|  |  | ke-t5-small | 17.0172 | 2.3512 | 10.9336 | 15.7156 |
|  |  | long-ke-t5-small-b | 39.1856 | 14.1602 | 21.4293 | 35.5396 |
|  |  | long-ke-t5-base | 42.5761 | 15.9075 | 22.6332 | 38.8568 |
| pubmed |  | long-ke-t5-small | 39.2269 | 17.8739 | 24.6764 | 35.6237 |
|  |  | ke-t5-small | 17.2858 | 2.3915 | 12.2287 | 16.0601 |
|  |  | long-ke-t5-small-b | 34.5738 | 13.2232 | 21.3909 | 31.1162 |
|  |  | long-ke-t5-base |  |  |  |  |
| bigpatent |  | long-ke-t5-small | 32.4983 | 12.5052 | 23.4889 | 26.5726 |
|  |  | ke-t5-small | 18.6584 | 5.5504 | 15.2256 | 14.8946 |
|  |  | long-ke-t5-small-b | 24.238 | 8.5326 | 18.3082 | 19.6711 |
|  |  | long-ke-t5-base | 30.7679 | 11.9528 | 22.4974 | 25.7071 |
| WCEP-10 |  | long-ke-t5-small | 42.3726 | 21.4949 | 34.0206 | 35.0072 |
|  |  | ke-t5-small | 0.0506 | 0.0 | 0.0499 | 0.0489 |
|  |  | long-ke-t5-small-b | 39.8325 | 19.3215 | 31.5279 | 32.5918 |
|  |  | long-ke-t5-base | 40.3259 | 19.4846 | 32.1004 | 33.1556 |
| media sum |  | long-ke-t5-small | 31.4069 | 15.5984 | 28.3295 | 28.8727 |
|  |  | ke-t5-small | 25.0165 | 11.4217 | 22.8546 | 23.2541 |
|  |  | long-ke-t5-small-b | 29.2367 | 14.3669 | 26.3199 | 26.8766 |
|  |  | long-ke-t5-base | 31.3612 | 15.9249 | 28.3042 | 28.8722 |
| aihub 리포트 요약 |  | long-ke-t5-small | 15.0943 | 3.2761 | 14.9415 | 14.977 |
|  |  | ke-t5-small | 10.7222 | 1.7525 | 10.6337 | 10.6303 |
|  |  | long-ke-t5-small-b | 15.3224 | 3.3282 | 15.1299 | 15.113 |
|  |  | long-ke-t5-base | 15.5752 | 3.3971 | 15.3948 | 15.4057 |
| aihub 도서 요약 |  | long-ke-t5-small | 16.7562 | 5.281 | 16.5748 | 16.6182 |
|  |  | ke-t5-small | 14.2405 | 4.1656 | 14.0975 | 14.0956 |
|  |  | long-ke-t5-small-b | 16.9341 | 5.4442 | 16.7344 | 16.7436 |
|  |  | long-ke-t5-base | 17.1789 | 5.5753 | 16.9486 | 17.0192 |
| aihub 대화 요약 |  | long-ke-t5-small | 7.2883 | 1.3608 | 7.2556 | 7.2436 |
|  |  | ke-t5-small | 5.7903 | 0.992 | 5.7529 | 5.7645 |
|  |  | long-ke-t5-small-b | 7.3371 | 1.4168 | 7.2879 | 7.286 |
|  |  | long-ke-t5-base | 7.4415 | 1.4991 | 7.4073 | 7.4037 |
| aihub 문서 요약 | Law | long-ke-t5-small | 7.2883 | 1.3608 | 7.2556 | 7.2436 |
|  |  | ke-t5-small | 1.7607 | 0.4086 | 1.6902 | 1.6845 |
|  |  | long-ke-t5-small-b | 29.3078 | 18.3298 | 28.9446 | 28.9207 |
|  |  | long-ke-t5-base | 29.1202 | 17.626 | 28.8872 | 28.8756 |
|  | magazine | long-ke-t5-small | 25.2596 | 8.8668 | 24.9621 | 24.9561 |
|  |  | ke-t5-small | 7.969 | 1.428 | 7.9237 | 7.9373 |
|  |  | long-ke-t5-small-b | 25.1503 | 8.9734 | 24.8403 | 24.7956 |
|  |  | long-ke-t5-base | 25.6775 | 8.7183 | 25.3353 | 25.3608 |
|  | news | long-ke-t5-small | 48.5417 | 22.4796 | 45.8339 | 45.8638 |
|  |  | ke-t5-small | 45.848 | 20.2444 | 43.9049 | 43.9259 |
|  |  | long-ke-t5-small-b | 48.9797 | 22.8113 | 46.2126 | 46.2326 |
|  |  | long-ke-t5-base | 50.0466 | 23.3671 | 47.2297 | 47.2576 |
| aihub 논문 요약 | paper entire | long-ke-t5-small | 21.4589 | 10.2949 | 21.2773 | 21.2379 |
|  |  | ke-t5-small | 19.9052 | 9.2992 | 19.7425 | 19.705 |
|  |  | long-ke-t5-small-b | 21.4008 | 10.2528 | 21.2608 | 21.1763 |
|  |  | long-ke-t5-base |  |  |  |  |
|  | paper section | long-ke-t5-small | 19.7416 | 9.159 | 19.5661 | 19.57 |
|  |  | ke-t5-small | 18.4416 | 8.4036 | 18.3012 | 18.2885 |
|  |  | long-ke-t5-small-b | 19.7998 | 9.303 | 19.6518 | 19.6355 |
|  |  | long-ke-t5-base |  |  |  |  |
|  | patent entire | long-ke-t5-small | 41.0209 | 34.428 | 40.6636 | 40.6225 |
|  |  | ke-t5-small | 12.9245 | 8.0218 | 12.7234 | 12.6897 |
|  |  | long-ke-t5-small-b | 13.665 | 9.2875 | 13.5268 | 13.4781 |
|  |  | long-ke-t5-base | 13.7759 | 9.2902 | 13.6306 | 13.5864 |
|  | patent section | long-ke-t5-small | 3.7274 | 1.788 | 3.6996 | 3.6942 |
|  |  | ke-t5-small | 1.0789 | 0.3615 | 1.0819 | 1.0827 |
|  |  | long-ke-t5-small-b | 2.7939 | 1.2206 | 2.7638 | 2.7556 |
|  |  | long-ke-t5-base |  |  |  |  |
| 모두의 말뭉치 요약 |  | long-ke-t5-small | 40.1879 | 19.3039 | 38.1355 | 39.4839 |
|  |  | ke-t5-small |  |  |  |  |
|  |  | long-ke-t5-small-b | 40.4486 | 20.1517 | 38.7216 | 39.7924 |
|  |  | long-ke-t5-base | 42.9854 | 21.2934 | 41.1163 | 42.3471 |


### Translation

- En2Ko: long-ke-t5-small(4096), long-ke-t5-small(512), ke-t5-small(512)
    - learning_rate: 0.001
    - 3 epochs
    - linear lr scheduler
    - per_device_train, eval batch size: 32
    - gradient_accumulation_steps: 2
    
    | Dataset | Model | BLEU (sacre) | Rouge 1 | Rouge 2 | Rouge L(LCS) | Rouge L summary |
    | --- | --- | --- | --- | --- | --- | --- |
    | aihub 전문분야 (식품) | long-ke-t5-small | 21.1552 | 47.9535 | 25.2124 | 47.4610 | 47.4541 |
    |  | ke-t5-small | 23.9372 | 50.7227 | 27.6673 | 50.1315 | 50.1351 |
    |  | long-ke-t5-small-b | 21.3214 | 47.9787 | 25.2593 | 47.4883 | 47.4961 |
    |  | long-ke-t5-base | 23.2660 | 48.2490 | 25.6514 | 47.8017 | 47.8034 |
    | aihub 기술과학 분야 | long-ke-t5-small | 11.8753 | 40.9999 | 16.8922 | 40.4692 | 40.4665 |
    |  | ke-t5-small | 14.4392 | 43.6611 | 19.0312 | 43.0210 | 43.0296 |
    |  | long-ke-t5-small-b | 11.9339 | 41.0901 | 16.9801 | 40.5691 | 40.5683 |
    |  | long-ke-t5-base | 13.3433 | 41.1310 | 17.2228 | 40.6583 | 40.6685 |
    | aihub 번역 (기술과학) | long-ke-t5-small | 29.3809 | 56.0565 | 31.2868 | 55.3868 | 55.3915 |
    |  | ke-t5-small | 29.9757 | 56.7740 | 31.8781 | 56.0861 | 56.0871 |
    |  | long-ke-t5-small-b | 29.5043 | 56.1805 | 31.4341 | 55.5118 | 55.5104 |
    |  | long-ke-t5-base | 31.6584 | 56.3259 | 31.7108 | 55.7085 | 55.7112 |
    | aihub 번역 (사회과학) | long-ke-t5-small | 11.5187 | 19.7873 | 7.5594 | 19.6404 | 19.6432 |
    |  | ke-t5-small | 12.8128 | 20.1936 | 7.8825 | 20.0312 | 20.0346 |
    |  | long-ke-t5-small-b | 11.6431 | 19.8517 | 7.6123 | 19.6960 | 19.6956 |
    |  | long-ke-t5-base | 13.7315 | 20.0635 | 7.8219 | 19.9223 | 19.9237 |
    | aihub 구어체 번역 | long-ke-t5-small | 29.2361 | 16.2141 | 5.7002 | 16.1348 | 16.1324 |
    |  | ke-t5-small | 28.3456 | 16.1321 | 5.7008 | 16.0532 | 16.0556 |
    |  | long-ke-t5-small-b | 29.3987 | 16.2242 | 5.7132 | 16.1476 | 16.1549 |
    |  | long-ke-t5-base |  |  |  |  |  |

- Ko2En: long-ke-t5-small(4096), long-ke-t5-small(512), ke-t5-small(512)
    - learning_rate: 0.001
    - 3 epochs
    - linear lr scheduler
    - per_device_train, eval batch size: 32
    - gradient_accumulation_steps: 2
    
    | Dataset | Model | BLEU (sacre) | Rouge 1 | Rouge 2 | Rouge L(LCS) | Rouge L summary |
    | --- | --- | --- | --- | --- | --- | --- |
    | aihub 전문분야 (식품) | long-ke-t5-small | 25.2258 | 61.8541 | 45.6312 | 58.6194 | 58.6162 |
    |  | ke-t5-small | 19.3967 | 55.8553 | 38.3890 | 52.4390 | 52.4392 |
    |  | long-ke-t5-small-b | 25.3589 | 61.9443 | 45.8026 | 58.7508 | 58.7521 |
    |  | long-ke-t5-base | 27.3901 | 63.6928 | 48.8768 | 60.7670 | 60.7704 |
    | aihub 기술과학 분야 | long-ke-t5-small | 23.0891 | 58.3203 | 42.9309 | 53.8890 | 53.8919 |
    |  | ke-t5-small | 19.0635 | 53.9964 | 37.2124 | 49.3091 | 49.3110 |
    |  | long-ke-t5-small-b | 23.3431 | 58.5863 | 43.3384 | 54.2245 | 54.2290 |
    |  | long-ke-t5-base | 25.2226 | 60.3302 | 46.4616 | 56.3676 | 56.3701 |
    | aihub 번역 (기술과학) | long-ke-t5-small | 43.9547 | 73.5404 | 58.9721 | 70.2830 | 70.2845 |
    |  | ke-t5-small | 35.4884 | 66.8445 | 50.7745 | 63.5526 | 63.5513 |
    |  | long-ke-t5-small-b | 44.1539 | 73.6785 | 59.1921 | 70.4340 | 70.4340 |
    |  | long-ke-t5-base | 46.6629 | 75.3339 | 62.1307 | 72.3458 | 72.3442 |
    | aihub 번역 (사회과학) | long-ke-t5-small | 34.6086 | 66.7367 | 48.5440 | 62.1330 | 62.1380 |
    |  | ke-t5-small | 27.6142 | 60.4710 | 40.2997 | 55.7643 | 55.7616 |
    |  | long-ke-t5-small-b | 35.0204 | 67.0560 | 49.0532 | 62.5399 | 62.5394 |
    |  | long-ke-t5-base | 38.3283 | 69.4958 | 53.1121 | 65.3983 | 65.3952 |
    | aihub 구어체 번역 | long-ke-t5-small | 47.1456 | 72.3959 | 53.9094 | 69.7503 | 69.7548 |
    |  | ke-t5-small | 41.9165 | 68.9184 | 49.3018 | 66.2888 | 66.2892 |
    |  | long-ke-t5-small-b | 47.4668 | 72.5553 | 54.1523 | 69.9538 | 69.9496 |
    |  | long-ke-t5-base | 51.6471 | 75.1553 | 58.0273 | 72.7531 | 72.7469 |


### Extractive QA

- KE-T5-small Encoder, Long-KE-T5-small Encoder, Long-KE-T5-base Encoder
    - learning_rate: 1e-3
    - 3 epochs
    - linear lr scheduler
    - per_device_train, eval batch size: 16
    - gradient_accumulation_steps: 1
    
    | Dataset | Model | Max seq. len. | Doc. stride | EM(std.) | F1(std.) |
    | --- | --- | --- | --- | --- | --- |
    | korquad v1.1 | long-ke-t5-small | 384 | 128 | 74.7315 | 81.2290 |
    |  |  | 512 | 128 | 74.3851 | 80.7247 |
    |  |  | 1024 | 256 | 74.0561 | 80.7784 |
    |  |  | 2048 | 512 | 73.9695 | 80.5849 |
    |  |  | 4096 | 1024 |  |  |
    |  | long-ke-t5-small-b | 384 | 128 | 75.9438 | 82.3362 |
    |  |  | 512 | 128 | 75.9612 | 82.3290 |
    |  |  | 1024 | 256 | 76.4807 | 82.9690 |
    |  |  | 2048 | 512 | 76.0304 | 82.2674 |
    |  |  | 4096 | 1024 |  |  |
    |  | ke-t5-small | 384 | 128 | 63.0585 | 71.9313 |
    |  |  | 512 | 128 | 63.1104 | 71.9144 |
    |  | long-ke-t5-base | 384 | 128 | 78.9054 | 84.9327 |
    |  |  | 512 | 128 | 79.3210 | 85.1142 |
    |  |  | 1024 | 256 | 78.6456 | 84.8061 |
    |  |  | 2048 | 512 | 79.4076 | 85.4435 |
    |  |  |  |  |  |  |
    | squad | long-ke-t5-small | 384 | 128 | 71.3150 | 80.1650 |
    |  |  | 512 | 128 | 71.3150 | 80.2506 |
    |  |  | 1024 | 256 | 71.5137 | 80.6000 |
    |  |  | 2048 | 512 | 70.8041 | 79.7309 |
    |  |  | 4096 | 1024 | 72.0813 | 80.7688 |
    |  | long-ke-t5-small-b | 384 | 128 | 75.2223 | 83.6019 |
    |  |  | 512 | 128 | 74.2289 | 82.7523 |
    |  |  | 1024 | 256 | 74.3803 | 82.9887 |
    |  |  | 2048 | 512 | 75.0236 | 83.2219 |
    |  |  | 4096 | 1024 | 73.9924 | 82.5684 |
    |  | ke-t5-small | 384 | 128 | 75.3642 | 83.9345 |
    |  |  | 512 | 128 | 74.5884 | 83.7374 |
    |  | long-ke-t5-base | 384 | 128 | 78.2119 | 86.2028 |
    |  |  | 512 | 128 | 77.7861 | 86.3788 |
    |  |  | 1024 | 256 | 77.8713 | 86.1739 |
    |  |  | 2048 | 512 | 78.2497 | 86.2862 |
    |  |  | 4096 | 1024 | 78.4295 | 86.5528 |
    | squad v2 | long-ke-t5-small | 384 | 128 | 63.6570 | 66.8638 |
    |  |  | 512 | 128 | 62.5789 | 66.1508 |
    |  |  | 1024 | 256 | 63.1264 | 66.6315 |
    |  |  | 2048 | 512 | 62.7810 | 66.0335 |
    |  |  | 4096 | 1024 | 62.7052 | 65.9422 |
    |  | long-ke-t5-small-b | 384 | 128 | 66.4532 | 69.6286 |
    |  |  | 512 | 128 | 66.3353 | 69.7081 |
    |  |  | 1024 | 256 | 66.3859 | 69.7294 |
    |  |  | 2048 | 512 | 66.1669 | 69.4706 |
    |  |  | 4096 | 1024 | 65.7794 | 69.0530 |
    |  | ke-t5-small | 384 | 128 | 68.8958 | 72.6531 |
    |  |  | 512 | 128 | 67.9777 | 71.5013 |
    |  | long-ke-t5-base | 384 | 128 | 71.3804 | 74.8221 |
    |  |  | 512 | 128 | 70.7487 | 74.4550 |
    |  |  | 1024 | 256 | 70.0412 | 73.3430 |
    |  |  | 2048 | 512 | 70.6645 | 74.1008 |
    |  |  | 4096 | 1024 | 70.4623 | 73.9362 |
    | AIhub 문서 v1 | long-ke-t5-small | 384 | 128 | 72.1344 | 83.8586 |
    |  |  | 512 | 128 | 71.9950 | 83.9625 |
    |  |  | 1024 | 256 | 71.7627 | 83.6288 |
    |  |  | 2048 | 512 | 71.8917 | 83.7626 |
    |  |  | 4096 | 1024 | 71.8143 | 83.8760 |
    |  | long-ke-t5-small-b | 384 | 128 | 72.9760 | 84.7126 |
    |  |  | 512 | 128 | 72.9605 | 84.7563 |
    |  |  | 1024 | 256 | 72.9863 | 84.8015 |
    |  |  | 2048 | 512 | 73.2187 | 84.9925 |
    |  |  | 4096 | 1024 | 72.8159 | 84.7260 |
    |  | ke-t5-small | 384 | 128 | 62.8769 | 78.9064 |
    |  |  | 512 | 128 | 63.7701 | 79.5062 |
    |  | long-ke-t5-base | 384 | 128 | 78.6503 | 88.4627 |
    |  |  | 512 | 128 | 77.9429 | 88.1608 |
    |  |  | 1024 | 256 | 78.5212 | 88.5138 |
    |  |  | 2048 | 512 | 78.1495 | 88.4320 |
    |  |  | 4096 | 1024 | 77.9997 | 88.2615 |
    | AIhub 문서 v2 | long-ke-t5-small | 384 | 128 | 73.7837 | 84.2847 |
    |  |  | 512 | 128 | 73.2841 | 84.0630 |
    |  |  | 1024 | 256 | 73.4335 | 84.2797 |
    |  |  | 2048 | 512 | 73.6296 | 84.2559 |
    |  |  | 4096 | 1024 | 73.2934 | 83.9189 |
    |  | long-ke-t5-small-b | 384 | 128 | 74.7035 | 85.2114 |
    |  |  | 512 | 128 | 74.3813 | 85.0149 |
    |  |  | 1024 | 256 | 74.3766 | 85.0537 |
    |  |  | 2048 | 512 | 74.3393 | 85.0309 |
    |  |  | 4096 | 1024 | 74.5634 | 85.2222 |
    |  | ke-t5-small | 384 | 128 | 65.3282 | 79.6755 |
    |  |  | 512 | 128 | 65.2628 | 79.8523 |
    |  | long-ke-t5-base | 384 | 128 | 79.5732 | 88.4408 |
    |  |  | 512 | 128 | 79.3398 | 88.4503 |
    |  |  | 1024 | 256 | 79.6292 | 88.7518 |
    |  |  | 2048 | 512 | 79.5452 | 88.6065 |
    |  |  | 4096 | 1024 | 79.2230 | 88.5651 |
    | AIhub 뉴스 v1 | long-ke-t5-small | 384 | 128 | 61.6810 | 70.2850 |
    |  |  | 512 | 128 | 62.0847 | 70.5243 |
    |  |  | 1024 | 256 | 62.5312 | 70.9533 |
    |  |  | 2048 | 512 | 62.9456 | 71.4680 |
    |  |  | 4096 | 1024 | 62.4062 | 71.0108 |
    |  | long-ke-t5-small-b | 384 | 128 | 64.1708 | 72.7946 |
    |  |  | 512 | 128 | 64.5066 | 73.0617 |
    |  |  | 1024 | 256 | 64.6995 | 73.2705 |
    |  |  | 2048 | 512 | 65.3282 | 73.9562 |
    |  |  | 4096 | 1024 | 64.7031 | 73.2146 |
    |  | ke-t5-small | 384 | 128 | 48.4032 | 60.5978 |
    |  |  | 512 | 128 | 48.8783 | 61.2285 |
    |  | long-ke-t5-base | 384 | 128 | 66.3070 | 74.8959 |
    |  |  | 512 | 128 | 67.2822 | 75.7887 |
    |  |  | 1024 | 256 | 66.7285 | 75.5530 |
    |  |  | 2048 | 512 | 67.3715 | 76.3660 |
    |  |  | 4096 | 1024 | 67.7966 | 76.3266 |
    | AIhub 뉴스 v2 | long-ke-t5-small | 384 | 128 | 62.7558 | 70.0984 |
    |  |  | 512 | 128 | 63.5060 | 70.9493 |
    |  |  | 1024 | 256 | 64.0594 | 71.9637 |
    |  |  | 2048 | 512 | 64.7629 | 72.3940 |
    |  |  | 4096 | 1024 | 64.2428 | 72.1601 |
    |  | long-ke-t5-small-b | 384 | 128 | 64.0261 | 71.4518 |
    |  |  | 512 | 128 | 65.6531 | 73.4838 |
    |  |  | 1024 | 256 | 66.4332 | 74.3676 |
    |  |  | 2048 | 512 | 66.7000 | 74.7448 |
    |  |  | 4096 | 1024 | 66.7500 | 74.7664 |
    |  | ke-t5-small | 384 | 128 | 50.9768 | 61.6998 |
    |  |  | 512 | 128 | 51.4536 | 62.6124 |
    |  | long-ke-t5-base | 384 | 128 | 65.9865 | 73.6660 |
    |  |  | 512 | 128 | 67.1634 | 74.8282 |
    |  |  | 1024 | 256 | 68.3069 | 76.3238 |
    |  |  | 2048 | 512 | 68.5303 | 76.5576 |
    |  |  | 4096 | 1024 | 68.6870 | 76.6632 |


## bibtex

```
@misc{long_ke_t5,
    author       = {KETI AIRC},
    title        = {Long-KE-T5: Long Korean English T5},
    month        = may,
    year         = 2023,
    url          = {https://github.com/AIRC-KETI/long-ke-t5}
}
```

