# SurGE 셋업 & 구현 체크리스트

SurGE 벤치마크로 survey 생성 시스템을 평가하기 위해 필요한 것들을 정리한 문서.
README는 데이터셋 설명 위주라, 여기서는 **실제로 돌리기 위해 필요한 것**과 **생성기가 지켜야 하는 출력 계약**에 집중한다.

---

## 0. 한눈에 보기

| 단계 | 결과물 | 완료 조건 |
|---|---|---|
| 1. 환경 | `surge` conda env | `python -c "import torch, FlagEmbedding, sentence_transformers"` 통과 |
| 2. 데이터 | `data/{corpus,surveys,queries}.json` | 3개 파일 존재 |
| 3. 생성 | `<out>/<survey_id>/*.md` | 아래 §4 출력 계약 충족 |
| 4. 평가 | `log.json` | `test_final.py` 가 지표 dict 출력 |

---

## 1. 사전 요구사항

- [ ] **GPU** — SH-Recall(BGE) / Relevance(NLI Cross-Encoder) 가 GPU를 쓴다. `--device` 는 `CUDA_VISIBLE_DEVICES` 로 들어가므로 단일 GPU 인덱스를 준다.
- [ ] **RAM** — `corpus.json`(1,086,992편)을 통째로 파싱해 dict 2개(`corpus_map`, `title2docid`)로 올린다. 수 GB 단위 여유 필요.
- [ ] **디스크** — 데이터 + HF 모델 캐시로 넉넉히 잡을 것.
- [ ] **OpenAI API 키** — `Structure_Quality`, `Logic` 두 지표가 `gpt-4o` 를 호출한다. 없으면 이 두 지표는 평가 불가.
- [ ] **네트워크** — HF 모델(`BAAI/bge-large-en-v1.5`, `cross-encoder/nli-deberta-v3-base`)을 첫 실행 시 자동 다운로드한다.

### 이 서버(연구실 장비) 기준 확정값

```
GPU     : NVIDIA L40S x8 (sm_89), driver 550.120 / CUDA 12.4
conda   : /data2/chanjoong/miniforge3  (conda 26.3.2 / mamba 2.5.0)
```

`conda` 는 PATH에 없을 수 있다. `~/.bashrc` 에 아래가 들어있는지 확인:

```bash
export MAMBA_ROOT_PREFIX="$HOME/miniforge3"
if [ -f "$MAMBA_ROOT_PREFIX/etc/profile.d/conda.sh" ]; then
    . "$MAMBA_ROOT_PREFIX/etc/profile.d/conda.sh"
    . "$MAMBA_ROOT_PREFIX/etc/profile.d/mamba.sh"
fi
```

---

## 2. 환경 구축

```bash
conda create -n surge python=3.10 -y
conda activate surge
pip install -r requirements.txt

# ★ 필수 후처리 — 아래 §2.1 참조
conda install -c conda-forge pyarrow pandas "numpy=1.26.4" -y
```

- [ ] `requirements.txt` 첫 줄의 `--extra-index-url https://download.pytorch.org/whl/cu121` 유지할 것. `torch==2.4.1` 을 CUDA 12.1 빌드로 받기 위한 것이다.
- [ ] 원본 README의 `torch==1.13.1+cu117` 은 CUDA 11.8(= Ada sm_89 지원 도입) 이전 버전이라 L40S용 cuBLAS/cuDNN 커널이 없다. 되돌리지 말 것.
- [ ] `FlagEmbedding==1.3.2` 핀을 풀지 말 것. §2.2 참조.

검증 (env 변수 없이 통과해야 정상):

```bash
cd src && python -c "
import markdownParser, rougeBleuFuncs, structureFuncs, informationFuncs
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagModel
import sqlite3, torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.4.1+cu121 True NVIDIA L40S
```

### 2.1 `pip` pyarrow/pandas → `libstdc++` 충돌 ★ 반드시 처리

`pip` 로 설치한 `pyarrow`(및 `pandas`)는 **시스템** `/lib/x86_64-linux-gnu/libstdc++.so.6`(6.0.30)을 먼저 로드한다.
그 뒤 conda의 `libicui18n.so.78`(→ `_sqlite3` → `nltk` → `rouge_score`)이 `CXXABI_1.3.15` 를 요구하는데,
같은 SONAME 이 이미 올라와 있어 conda의 최신 `libstdc++`(6.0.35, CXXABI_1.3.17)이 쓰이지 않는다.

```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
             (required by .../envs/surge/lib/libicui18n.so.78)
```

`import sqlite3` 단독은 성공하고 `import pandas, sqlite3` 는 실패하는 **로드 순서 의존 버그**라 진단이 까다롭다.
`ROUGE-BLEU` 지표가 통째로 막힌다.

**해결** — conda-forge 빌드로 교체하면 conda의 `libstdc++` 이 먼저 로드되어 근본적으로 해소된다:

```bash
conda install -c conda-forge pyarrow pandas "numpy=1.26.4" -y
```

`numpy=1.26.4` 고정은 필수다. 빼면 numpy 2.2 로 올라가 torch/transformers ABI가 깨진다.

> `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` 로도 해결되지만 **쓰지 말 것.** conda의 OpenSSL을 시스템 `ssh` 가 잡아
> `OpenSSL version mismatch` 로 SSH push 가 깨진다. (`LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` 은
> 부작용이 확인되지 않았으나, 위 conda-forge 방식이면 env 변수 자체가 불필요하다.)

### 2.2 `FlagEmbedding` 은 1.3.2 로 고정

최신 1.4.0 은 `AutoModel.from_pretrained(dtype=...)` 를 호출하는데, 이 kwarg 는 훨씬 최신 `transformers` 에만 있다.
핀된 `transformers==4.44.2` 와 조합하면 `FlagModel` 생성 시점에 죽는다:

```
TypeError: BertModel.__init__() got an unexpected keyword argument 'dtype'
```

`SH-Recall` 이 통째로 막힌다. **1.3.2 는 `transformers==4.44.2` 를 정확히 핀하는 마지막 릴리스**라
레포의 다른 핀들과 정확히 맞물린다 (`datasets==2.19.0` 도 함께 내려간다).

`requirements.txt` 에 명시되지 않았지만 **전이 의존성으로 반드시 설치되는** 것들 (누락 시 import 에러):

| 패키지 | 필요한 이유 | 출처 |
|---|---|---|
| `sentence-transformers` | `CrossEncoder` (NLI 지표) | FlagEmbedding |
| `sentencepiece`, `protobuf` | DeBERTa-v3 토크나이저 | FlagEmbedding |
| `pandas`, `scipy`, `numpy` | ROUGE/BLEU, Relevance | 각 패키지 |

> `bertopic` 은 평가 코드에서 **import되지 않는다**. 데이터셋의 `Bertopic_CD` 필드를 만들 때 쓰인 것이라 평가에는 불필요하지만, requirements에 남아 있으므로 그냥 둔다.

---

## 3. 데이터 준비

```bash
python src/download.py     # 반드시 레포 루트에서
```

- [ ] `data/corpus.json` — 1,086,992편 논문 (Title/Authors/Year/Date/Abstract/Category/doc_id)
- [ ] `data/surveys.json` — 205편 ground truth survey (structure/all_cites 포함)
- [ ] `data/queries.json` — **`download.py` 가 받지 않는다.** [Google Drive 폴더](https://drive.google.com/drive/folders/1ZZPeZvjexFcCmgFqxftKeCPn1vYeBR0Q)에서 수동 다운로드. 리트리버 학습용이라 평가만 할 거면 없어도 된다.

주의점:

- `src/download.py` 는 맨 위에서 `from evaluator import SurGEvaluator` 를 import한다. 다운로드만 하려 해도 **torch/FlagEmbedding 설치가 끝나 있어야** 한다.
- `data/` 경로가 CWD 기준이라 **레포 루트에서 실행**해야 한다.
- 이미 파일이 있으면 건너뛴다(`os.path.exists` 체크).

---

## 4. 생성기 출력 계약 ★ 구현 시 핵심

평가기는 마크다운을 정규식으로 파싱한다. 아래를 어기면 **에러 없이 점수만 0으로 떨어진다.**

### 4.1 디렉토리 구조

```
<passage_dir>/
  0/    <파일 1개>            ← 폴더명 = survey_id (정수)
  1/    <파일 1개>
  ...
```

- [ ] 폴더명은 `surveys.json` 의 `survey_id` 와 일치하는 **정수 문자열**. `int()` 로 캐스팅되므로 `survey_0` 같은 건 안 된다.
- [ ] 폴더 안에 파일이 **1개면 확장자 무관**하게 그 파일을 읽는다. (`baselines/ID/output/0/` 는 확장자 없는 파일 하나)
- [ ] 파일이 **2개 이상이면 `.md` 로 끝나는 첫 파일**만 읽는다. (`baselines/Autosurvey/output/0/` 는 `.json` + `.md`)
- [ ] 테스트셋은 `survey_id` **0–40 (41편)**. 세 베이스라인(`ID`, `Autosurvey`, `Naive`) 출력이 `baselines/` 에 들어 있다.

### 4.2 본문 (마크다운)

- [ ] 제목은 `^#+\s+제목` 형식. `#` 개수가 곧 계층 레벨.
- [ ] `root` 나 `Abstract:` 를 제목에 포함한 노드는 구조 비교에서 **제외**된다.
- [ ] **제목이 5개 미만이면 `Structure_Quality` 가 무조건 0**을 반환한다 (`structureFuncs.py:186`).
- [ ] 본문 블록이 **100자 미만이면 ROUGE/BLEU, Logic 평가에서 무시**된다 (`get_content_list`).
- [ ] 인용은 본문에 `[숫자]` 로 표기. 정규식 `\[(\d+)\]` 로 뽑고, 앞뒤 마침표 기준으로 문장을 잘라 NLI 쌍을 만든다 → **문장 끝에 마침표를 반드시 넣을 것.**

### 4.3 참고문헌 목록

문서 말미, 각 줄이 `[` 로 시작해야 한다. 두 형식 지원 (`markdownParser.py:62-87`):

```
[1] Zhi Zhou,Xu Chen,En Li. (n.d.). *Edge Intelligence  Paving the Last Mile of AI*
[2] Edge Intelligence  Paving the Last Mile of AI
```

- [ ] 제목 매칭 정규식이 `[\w\s:,.-]` 만 허용한다. **괄호 `()`, 슬래시 `/`, 물음표 등이 제목에 있으면 파싱이 잘리거나 실패**한다. corpus의 원 제목을 그대로 쓰되 이 문자셋을 벗어나면 대체할 것.
- [ ] 제목→`doc_id` 매칭은 **알파벳만 남기고 소문자화**한 뒤 비교한다(`normalize_string`). 공백/구두점 차이는 무시되므로 걱정 없지만, 단어 자체가 다르면 매칭 실패 → 그 인용은 `Coverage` 에서 미스 처리된다.
- [ ] 본문의 `[n]` 번호와 목록의 `[n]` 번호가 **일치**해야 한다.

---

## 5. 평가 실행

```bash
conda activate surge
cd /data2/chanjoong/survey-agent/SurGE   # 반드시 레포 루트

python src/test_final.py \
  --passage_dir ./baselines/ID/output \
  --save_path   ./baselines/ID/output/log.json \
  --device 0 \
  --api_key sk-xxx
```

- [ ] **레포 루트에서 실행**. `src/test_final.py` 내부의 `import evaluator` 가 `sys.path[0]=src` 에 의존한다.
- [ ] `--eval_list` 로 일부 지표만 돌릴 수 있다 (공백 구분). 기본값 `ALL`.

### 지표별 요구사항

| `--eval_list` 값 | 논문 표기 | 모델/API | 필요 데이터 | 비고 |
|---|---|---|---|---|
| `Coverage` | **Recall** (§4.1) | 없음 | surveys + corpus | GT `all_cites` 중 맞힌 비율 |
| `Relevance-Paper` | **Doc-Acc** (§4.2) | `cross-encoder/nli-deberta-v3-base` (GPU) | surveys + corpus | GT 인용에 있으면 무조건 1점 |
| `Relevance-Section` | **Sec-Acc** (§4.2) | 동일 NLI 모델 | corpus.json | 섹션 제목 ↔ 인용 논문 |
| `Relevance-Sentence` | **Sent-Acc** (§4.2) | 동일 NLI 모델 | corpus.json | 문장 ↔ 인용 논문 |
| `Structure_Quality` | **SQS** (§4.3) | **OpenAI gpt-4o** | surveys.json | 0–5 정수, 제목 5개 미만이면 0 |
| `SH-Recall` | **SHR** (§4.3) | `BAAI/bge-large-en-v1.5` (GPU, fp16) | surveys.json | soft cardinality 기반 |
| `Logic` | **CQS** (§4.4) | **OpenAI gpt-4o** | 없음 | 블록 최대 20개 샘플(seed 42), 1000자 절단 |
| `ROUGE-BLEU` | R-L / BLEU — **정식 지표 아님** | 없음 (CPU) | surveys.json | §9.2 참조. 100자 이상 블록만 |
| `ALL` | 위 전부 | 전부 | 전부 | |

### OpenAI 이외의 엔드포인트를 쓸 때

`src/evaluator.py:31` 의 클라이언트 생성부를 고친다. 바로 위에 DeepSeek 예시가 주석으로 있다:

```python
self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
```

단, `structureFuncs.py:159` 와 `informationFuncs.py:288` 에 `model="gpt-4o"` 가 하드코딩되어 있으므로 모델명도 같이 바꿔야 한다.

---

## 6. 트러블슈팅

| 증상 | 원인 / 대응 |
|---|---|
| `ModuleNotFoundError: evaluator` | 레포 루트가 아닌 곳에서 실행함 |
| `FileNotFoundError: data/surveys.json` | CWD 문제 또는 `download.py` 미실행 |
| `Structure_Quality` 가 계속 0 | 생성 문서의 `#` 제목이 5개 미만 |
| `Coverage` 가 비정상적으로 낮음 | 참고문헌 제목에 `()` `/` 등 정규식 미허용 문자 → `parse_refs` 실패 |
| API 오류 시 프로세스가 멈춘 듯 반복 | `chat_openai` 가 예외마다 자기 자신을 재귀 호출하는데 **종료 조건이 없다**. try 5에서 경고만 찍고 계속 돈다 → 키/쿼터부터 확인 (`structureFuncs.py:152`, `informationFuncs.py:278`) |
| CUDA arch 경고 / 커널 에러 | `torch` 가 cu117로 되돌아갔는지 확인 |

---

## 7. Git

```
origin   https://github.com/oneal2000/SurGE.git        (업스트림, 읽기용)
myfork   git@github.com:brian-223134/SurGE.git         (SSH, push 대상 / main 추적)
```

- SSH 키 `~/.ssh/id_ed25519` 는 passphrase가 걸려 있어 push 시 입력을 요구한다. `~/.ssh/config` 의 `AddKeysToAgent yes` 덕분에 세션당 1회만 입력하면 된다.
- 업스트림 동기화: `git fetch origin && git merge origin/main`

---

## 8. 데이터 스키마 요약

`surveys.json` — 205편. 평가에서 쓰는 필드:

| 필드 | 용도 |
|---|---|
| `survey_id` | 출력 폴더명과 매칭되는 키 |
| `structure` | `{id, parent_id, title, content}` 리스트. 제목 비교(Structure_Quality, SH-Recall)와 ROUGE/BLEU 정답 텍스트 |
| `all_cites` | `doc_id` 리스트. Coverage / Relevance-Paper 정답 |
| `abstract`, `year`, `category`, `authors` | 메타데이터 |

`corpus.json` — 1,086,992편. `doc_id`, `Title`, `Abstract` 를 평가에 쓴다 (`Title`→`doc_id` 역인덱스, `Abstract`는 NLI 전제).

`queries.json` — 리트리버 학습용. `prefix_titles_query`(질의) ↔ `cites`(정답 doc_id). train/dev 분할은 직접 해야 한다.

---

## 9. 논문(SurGE.pdf) 대조

레포 루트의 `SurGE.pdf` (SIGIR '26) 기준. 코드만 봐서는 알 수 없는 설계 의도와, **코드와 논문이 어긋나는 지점**을 정리한다.

### 9.1 목표 수치 (논문 Table 4)

Qwen2.5-14B-Instruct 로 통일한 베이스라인 결과. 구현 시 참고할 기준선:

| Baseline | Recall | Doc-Acc | Sec-Acc | Sent-Acc | SQS | SHR | CQS |
|---|---|---|---|---|---|---|---|
| RAG-Qwen | 0.0214 | 0.2857 | 0.2502 | 0.2500 | 0.6829 | **0.7900** | 4.6723 |
| RAG-GPT | 0.0419 | 0.4525 | 0.3166 | 0.3226 | 0.7561 | 0.6659 | **4.9250** |
| RAG-Gemini | **0.0784** | **0.4680** | **0.4056** | **0.4124** | **1.2927** | 0.7591 | 4.5619 |
| AutoSurvey | 0.0351 | 0.3617 | **0.4935** | **0.4870** | **1.3902** | 0.9697 | 4.7390 |
| StepSurvey | 0.0630 | 0.4576 | 0.4571 | 0.4636 | 1.1951 | **0.9763** | **4.8451** |
| SurveyForge | **0.0868** | **0.4719** | 0.4651 | 0.4772 | 1.0537 | 0.9493 | 4.8070 |

**Recall 이 8% 대에 그친다.** 반면 리트리버 자체는 R@1000 = 0.6805 (BM25 는 0.2715) 로 상한이 68%다 (§6.1).
즉 병목은 검색이 아니라 **검색된 근거를 생성 단계에서 실제로 인용에 반영하는 능력**이다. 논문이 지목한 핵심 개선 포인트.

### 9.2 ROUGE / BLEU 는 공식 지표가 아니다

논문 §4.4: *"both ROUGE and BLEU are provided strictly for reference and do not constitute part of the metric suite in our SurGE benchmark."*
n-gram 매칭이 개방형 과학 텍스트의 의미적 정확성을 못 잡는다는 이유. 성능 보고 시 주 지표로 쓰지 말 것.

### 9.3 코드 ↔ 논문 불일치 ★ 실측 확인함

**(a) 부분점수 0.5 조건이 코드에서 더 엄격하다**

논문 식 (2):

```
y(r) = 1     if p_ent > max(p_neu, p_con)
       0.5   if p_neu > max(p_ent, p_con)
       0     otherwise
```

코드 (`informationFuncs.py` 의 3개 함수 모두 동일):

```python
if e > c and e > n:      hit += 1
elif n > c and n > e:
    if e > c:            hit += 0.5      # ← 논문에 없는 추가 조건
```

`p_neu` 가 최대이면서 `p_con > p_ent` 인 경우, 논문은 0.5 를 주지만 코드는 0 을 준다.

**(b) NLI 입력 쌍 순서가 레벨마다 다르다**

논문 §4.2 는 premise = 인용 논문 내용, hypothesis = 관련성 주장으로 고정한다. 코드는:

| 레벨 | 실제 전달 순서 | 논문과 일치? |
|---|---|---|
| `Relevance-Paper` | `(paper_content, claim)` | O |
| `Relevance-Section` | `(claim, paper_content)` | **X — 뒤집힘** |
| `Relevance-Sentence` | `(claim, paper_content)` | **X — 뒤집힘** |

`evaluator.py:205-206` 이 `(sen_1, sen_section)` = (premise, hypothesis) 로 만들지만,
`eval_relevance_section` 이 `for citation, subtitle in ...` 로 받아 `(subtitle, citation)` 으로 되돌려 넣으면서 순서가 반전된다.

**영향 실측** — 동일한 논문/섹션 쌍에 대해:

```
논문 순서 (premise, hypothesis) : [con, ent, neu] = [-4.123, 0.435,  3.129]  -> 0.5
코드 순서 (hypothesis, premise) : [con, ent, neu] = [-6.852, 3.588,  2.137]  -> 1.0
```

같은 내용이 0.5 와 1.0 으로 갈린다. **평가기를 고치지 말 것** — 논문 Table 4 수치가 이 코드로 산출된 값이므로,
비교 가능성을 유지하려면 있는 그대로 쓰고 이 사실만 인지하면 된다.

### 9.4 리트리버 학습 프로토콜 (§5.1, §5.3)

`queries.json` 으로 학습할 때의 논문 설정:

- 구조: RoBERTa-base **dual encoder**, 쿼리/문서 인코더 **파라미터 공유**. 입력은 `[CLS] q [SEP]`, `[CLS] d [SEP]`, 최종층 `[CLS]` 벡터 사용.
- 점수: `s(q,d) = h_q · h_d` (dot product)
- 손실: positive 1개 + corpus 에서 무작위 샘플한 negative N개에 대한 softmax cross-entropy
- 분할: **train:test = 4:1 무작위**
- 최적화: AdamW, lr **5e-6**, **10 epochs**, fp16
- 추론: 토픽당 **top-100** 검색 → 생성 단계 입력

성능 (§6.1, Table 3):

| Model | R@20 | R@100 | R@1000 |
|---|---|---|---|
| BM25 | 0.0548 | 0.1193 | 0.2715 |
| PR (논문 리트리버) | 0.1706 | 0.3665 | 0.6805 |

### 9.5 기타 확정 사항

- **테스트셋**: 논문은 GT 205편 중 테스트 인스턴스 수를 명시하지 않는다. 레포의 `baselines/*/output` 은 `survey_id` **0–40 (41편)** 이며, 리트리버 학습 분할이 4:1 이므로 205 × 1/5 ≈ 41 과 일치한다.
- **베이스라인 base LLM**: 전부 **Qwen2.5-14B-Instruct** 로 통일. RAG 만 GPT-4o / Gemini-2.5-Pro-Thinking 변형 추가.
- **실험 환경**: 8× A100 40GB, 1TB RAM.
- **SHR 임베딩 모델**: `bge-large-en-v1.5` (코드 기본값과 일치).
- **메타평가**: SQS 의 인간 순위 상관 Kendall τ = 0.805 (GT 포함) / 0.610 (GT 제외), CQS 는 0.797 / 0.594.
- **미해결**: 레포의 `baselines/ID` 가 무엇인지 논문으로 확정되지 않는다. 논문 베이스라인은 RAG / AutoSurvey / StepSurvey / SurveyForge 인데 레포에는 `ID` / `Autosurvey` / `Naive` 만 있다. `Naive` 는 RAG 로 보이나 `ID` 는 대응이 불분명하다.
