# Oracle 프레임 crop QA 실험 계획 (WorldMM, EgoLifeQA A1_JAKE)

> **질문.** 검색 실패를 없앤 상태(정답 클립 프레임 직접 제공)에서, 프레임을 **잘라서**(gaze / center crop) 주거나
> **전체 프레임에 crop을 덧붙여** 주면 QA 정확도가 오르는가? 오른다면 그것은 **시선 위치** 덕인가, **확대(자르기) 자체** 덕인가?
>
> 이 문서는 결과를 보기 전에 설계를 고정하기 위한 것이다. 결과를 본 뒤에 바꾼 것은 7장에 "사후"로 따로 적는다.

관련: [visual_routing_qa/ORACLE_REPORT.md](../visual_routing_qa/ORACLE_REPORT.md)(oracle + gaze prompting),
[visual_routing_qa/REPORT.md](../visual_routing_qa/REPORT.md)(검색 Top-3에서 full / gaze crop / dual),
[gaze_crop/README.md](../gaze_crop/README.md)(검색 임베딩에서 full / center / gazef crop).

---

## 1. 왜 하나

지금까지 확인한 것과 비어 있는 칸:

| | full | gaze crop | center crop | full + gaze crop | full + center crop |
|---|---|---|---|---|---|
| 검색 임베딩 (recall) | 9.2% | **13.4%** | 10.8% | – | – |
| 검색 Top-3 QA | 44.1% | 44.8% | – | 45.0% (`dual_gaze`) | – |
| **Oracle 프레임 QA** | 48.3% (`oracle_full`) | **없음** | **없음** | **없음** | **없음** |

- 검색 Top-3의 QA는 정답 클립이 거의 없어서(hit@3 10.8~15.8%) crop의 효과가 가려졌을 수 있다.
- 검색에서는 gaze crop이 center crop보다 나았지만 차이가 작았다(+2.6 %p, p=0.079). QA 입력에서도 같은 순서가 나오는지는 모른다.
- 그래서 정답 프레임을 고정해 놓고, 같은 프레임을 **어떤 모양으로 보여주는지만** 바꾼다.

## 2. 고정하는 것 (기존 oracle 실험과 동일)

| 항목 | 내용 |
|---|---|
| 대상 문항 | 정답 클립(`tol=0`)이 있는 문항 전체, **487개** (`visual_routing_qa/run_data/oracle_manifest.json`의 `question_ids`와 같은 목록) |
| 정답 클립 | 같은 manifest의 `target_keys` (중앙값 1개, 최대 6개) |
| 텍스트 메모리 | `visual_routing_qa/run_data/text_context/{qid}.json`를 **읽기만** 한다. `items_hash`를 결과에 기록해 모든 조건이 같은지 검증 |
| visual 검색 | **하지 않는다.** `WorldMemory`와 임베딩 모델을 올리지 않고 QA 템플릿(`qa_egolife`)만 쓴다 |
| QA 모델 | `qwen3vl-8b`, greedy, `max_new_tokens=32`, 채점 `evaluate_prediction` |
| 프롬프트 | `run.py`의 `qa_prompt` 그대로. **추가 지시문(note)은 넣지 않는다** (검색 Top-3의 `dual_gaze`와 같은 방식) |
| gaze 좌표 | `gaze_crop/gaze_points.json`, 프레임 시각의 가장 가까운 sample(±0.5 s, 없으면 클립 median) = `gaze_at(tol=0.5)` |
| crop | `gaze_common.crop_box`, **ratio 0.5** (짧은 변의 절반인 정사각형, 프레임 밖으로 나가면 안쪽으로 밀어 넣음). 검색 실험의 `gazef_05`, `center_05`와 같다 |

## 3. 조건

모든 조건은 **같은 타임스탬프**를 쓴다. 정답 클립의 1 fps 프레임을 클립별 비례 샘플링으로 **최대 32개 타임스탬프**까지 고른다
(`oracle.proportional_cap`, 상한만 64 → 32). dual 조건은 타임스탬프마다 이미지가 2장이 되므로 이미지 수는 최대 64장이고,
이 상한은 기존 oracle 실험과 같다.

| 조건 | 타임스탬프마다 넣는 이미지 | 이미지 수 | 역할 |
|---|---|---|---|
| `full` | 전체 프레임 | T | 기준선 (이 파이프라인에서 다시 실행) |
| `gaze` | gaze crop | T | crop만 줄 때, 시선 위치 |
| `center` | 중앙 crop (0.5, 0.5) | T | crop만 줄 때, 위치 정보 없는 대조군 |
| `dual_center` | 전체 → 중앙 crop | 2T | 전체 + 확대, 위치 정보 없음 |
| `dual_gaze` | 전체 → gaze crop | 2T | 전체 + 시선 확대 |
| `dual_offcenter` | 전체, 그리고 **시선이 중심에서 멀 때만** → gaze crop | T + n_off | 시선이 중앙 crop 밖을 볼 때만 덧붙임 |

(T = 문항의 타임스탬프 수. 정답 클립이 1개인 문항은 30~31개로 상한에 걸리지 않는다.)

- `full`을 다시 돌리는 이유: 정답 클립이 여러 개인 문항은 상한이 64에서 32로 바뀌어 기존 `oracle_full`과 입력이 다르다.
  정답 클립이 1개인 문항은 입력이 기존 `oracle_full`과 같아야 하므로, **응답이 같은지 비교해서 재현성을 확인**한다(5장).
- 이미지 순서는 시간 순이고, dual은 `[t1 전체, t1 crop, t2 전체, t2 crop, ...]`처럼 타임스탬프마다 짝을 지어 넣는다.
  `dual_offcenter`는 시선이 멀지 않은 타임스탬프에서는 전체 프레임 1장만 넣는다.

### 3.1 off-center 기준 — B로 확정

"중심 좌표에서 gaze가 30% 이상 떨어져 있을 때"를 정규화 좌표로 어떻게 재느냐에 따라 해당 프레임 비율이 크게 다르다.
gpu3의 `gaze_points.json`으로 487문항의 정답 클립 447개(1 fps 프레임 13,816장)를 미리 세어 봤다. 결과를 보기 전에 센 것이다.

| 정의 (d = gaze와 (0.5, 0.5)의 유클리드 거리, 정규화 좌표) | 해당 프레임 | 1장 이상인 문항 | 5장 이상 | 문항당 평균 |
|---|---:|---:|---:|---:|
| A. d / 0.5 ≥ 0.3 (중심→가장자리 거리의 30%, 즉 d ≥ 0.15) | 26.7% | 468 | 389 | 9.0장 |
| **B. d / 0.707 ≥ 0.3 (중심→모서리 거리의 30%, 즉 d ≥ 0.212)** ← 채택 | 7.6% | 374 | 98 | 2.6장 |
| C. d ≥ 0.3 (정규화 좌표 그대로) | 0.7% | 87 | 2 | 0.2장 |
| (참고) max(\|dx\|, \|dy\|) / 0.5 ≥ 0.3 | 22.3% | 456 | 328 | 7.5장 |

- 이 데이터의 시선은 중앙에 몰려 있다. x는 0.517 ± 0.085, y는 0.565 ± 0.073이고, 거리 d의 중앙값은 0.106, 상위 10%는 0.20이다.
  좌표는 캘리브레이션 없이 단일 focal로 투영한 근사값이라 실제보다 중앙으로 몰려 있을 수 있다([gaze_crop README](../gaze_crop/README.md)).
- **C를 쓰면 이 조건은 사실상 `full`과 같다.** 400문항에서 crop이 한 장도 붙지 않으므로 비교할 수가 없다.
- **채택: B (d ≥ 0.3 × √0.5 ≈ 0.212), 결과를 보기 전에 정했다.** 시선이 이만큼 벗어나면 gaze crop과 center crop의 겹침(IoU)이
  약 41% 이하(한 축으로 벗어날 때, 대각선이면 약 32%)로 줄어서, 중앙 crop이 이미 덮는 곳은 덧붙이지 않는다는 이 조건의 취지에 맞는다.
- 대신 crop이 붙는 경우가 적다. 117문항은 한 장도 붙지 않아 `dual_offcenter`가 `full`과 입력이 같고, 문항당 평균 2.6장뿐이다.
  그래서 `dual_offcenter − full` 차이는 작게 나올 가능성이 높고, crop이 실제로 붙은 문항(분석 3)과 5장 이상 붙은 문항(기술 분석)을 따로 본다.
- preflight로 실제 쓰는 타임스탬프(1 fps, 상한 32)를 세면 off-center 8.3%(1,217/14,651), 1장 이상 **370문항**, 5장 이상 76문항,
  문항당 평균 2.5장이다(위 표는 gaze sample을 초 단위로 센 사전 추정치). `dual_offcenter` 이미지 수는 중앙값 32, 최대 53장이다.
- 코드 상수 `OFFCENTER_DIST = 0.3 * 0.5 ** 0.5`로 고정하고 manifest에 기록한다. 결과를 본 뒤에는 바꾸지 않는다.

### 3.2 선택 조건 (포함)

| 조건 | 내용 | 가리는 것 |
|---|---|---|
| `dual_offcenter_center` | `dual_offcenter`와 **같은 타임스탬프**에 gaze crop 대신 중앙 crop을 붙임 | `dual_offcenter`의 이득이 시선 위치 덕인지, 이미지가 더 들어간 덕인지 |

`dual_offcenter`는 문항마다 이미지 수가 달라서 `full`과 비교하면 위치 효과와 이미지 수 효과가 섞인다.
이 조건은 이미지 수와 위치를 정확히 맞춘다. 비용은 487회 추가다.

## 4. 분석 (사전 고정)

**표본**

| 분석 | 문항 | 비고 |
|---|---|---|
| 1 (주) | 487 전체 | |
| 2 | `need_audio=False` 287 | 시각 근거 문항 |
| 3 | off-center 프레임이 1장 이상인 문항 (370) | `dual_offcenter`, `dual_offcenter_center` 비교에만 쓴다. gaze만으로 정해지므로 결과와 무관한 사전 정의다 |

**주 비교 3개** (짝지은 비교, Holm 보정)

| 비교 | 질문 |
|---|---|
| `gaze − center` | crop만 줄 때 시선 위치가 중앙보다 나은가 |
| `dual_gaze − dual_center` | 전체 프레임이 함께 있을 때 시선 위치가 중앙보다 나은가 |
| `dual_offcenter − full` | 시선이 벗어날 때만 crop을 덧붙이면 나아지는가 |

**부 비교** (기술 분석, 보정하지 않음): 모든 조건 − `full`, `dual_gaze − gaze`, `dual_center − center`,
`dual_offcenter − dual_gaze`, (선택 조건을 돌렸다면) `dual_offcenter − dual_offcenter_center`.

**방법**: `analyze_oracle.py`와 같은 방법을 쓴다. 짝지은 부트스트랩 95% CI(10,000회), 정확 McNemar 검정, 개선/악화 문항 수.
유형별(EntityLog 등) 분석은 사후·기술 분석으로만 표기한다.

**해석 기준** (미리 정함)

| 결과 | 해석 |
|---|---|
| `gaze` ≈ `center`, 둘 다 > `full` | 이득은 확대 자체에서 온다. 시선 위치는 기여하지 않는다 |
| `gaze` > `center` | 시선 위치가 QA 입력에서도 기여한다 |
| `gaze`, `center` < `full`, dual ≈ `full` | 자르면 맥락을 잃는다. 덧붙여도 손해는 없다 |
| `dual_*` > `full`이지만 `dual_gaze` ≈ `dual_center` | 확대한 이미지를 추가하는 것 자체의 효과 |
| `dual_offcenter` ≈ `dual_gaze` | 시선이 중앙에 있을 때 붙인 crop은 정보가 없었다 (같은 결과를 더 적은 이미지로 얻는다) |

**검정력**: N=487에서 이전 실험의 짝지은 차이 기준으로 약 ±2.5 %p 이내의 효과는 구분하기 어렵다.
검색 실험의 `gazef − center` 차이(+2.6 %p)와 비슷한 크기라면 이 실험으로 유의하게 나오지 않을 수 있다.

## 5. 검증 (본 실행 전후)

1. **preflight (CPU)**: 487문항 모두 영상, gaze, 텍스트 컨텍스트가 있는지 확인한다. 조건별 타임스탬프 수와 이미지 수 분포,
   off-center 프레임 수 분포(3.1의 표와 같은지)를 출력한다.
2. **입력 예시 그림 (CPU)**: 시선이 중심에서 먼 문항 1개를 골라 6개 조건의 입력을 한 장에 그려(`input_example.png`) crop 위치를 눈으로 확인한다.
3. **smoke (`--limit 5`)**: 오류가 없는지, 이미지 해시가 조건마다 다른지, `text_hash`가 기존 oracle 결과와 같은지 확인한다.
4. **재현성**: 정답 클립이 1개인 문항에서 새 `full`과 기존 `oracle_full`의 응답이 같은지 본다. 다르면 같은 입력에서도
   응답이 바뀐다는 뜻이므로 노이즈 크기로 보고에 적는다.

## 6. 구현

새 폴더 `experiments/oracle_crop_qa/`에 둔다. 기존 코드는 import해서 쓰고 수정하지 않는다.

| 파일 | 내용 |
|---|---|
| `oracle_crop.py` | `--stage preflight / example / qa`. `visual_routing_qa/oracle.py`의 `load_questions`, `proportional_cap`과 `run.py`의 `qa_prompt`, `text_record`(텍스트 경로만 `--text-dir`로 지정)를 재사용. 조건별 입력 생성, 이어서 실행 가능, `--shard i/n`으로 문항 분할 |
| `analyze_crop.py` | 4장의 표를 `run_data/analysis/summary.md`로, 문항별 정오를 `per_question.csv`로 저장 |
| `run_data/manifest.json` | 문항 목록, 정답 클립, 타임스탬프 상한(32), crop ratio(0.5), `OFFCENTER_DIST`, 모델, 입력 파일 해시 |
| `run_data/results/{cond}/{qid}.json` | 응답, 정오, 이미지 id(`full:key:idx` / `crop:key:idx` / `center:key:idx`), 이미지 해시, gaze 좌표, 중심 거리, `text_hash` |
| `README.md` | 결과 보고 (실행 후) |

**비용**: 487문항 × 7조건 = 3,409회. gpu3(RTX 3090 24 GB × 2, `device_map="auto"`로 모델을 두 장에 나눠 올림)에서
기존 oracle 실행의 호출 시간은 이미지 약 30장일 때 중앙값 7.0초, 약 60장일 때 14.4초였다(파일 수정 시각 간격으로 잼).
문항당 단일 3조건 약 22초 + dual 2조건 약 29초 + `dual_offcenter`·`dual_offcenter_center`(평균 약 34장) 각 약 7.5초 ≈ 66초이므로
**약 9시간**(7조건, 3,409회)으로 예상한다.

- 메모리: 이미지 상한이 64장으로 기존과 같고, 기존 실행에서 60장 이상 입력(152회)도 OOM 없이 돌았다. 다만 이번에는 60장 이상 입력이
  약 970회로 늘어나므로, smoke에 **텍스트 컨텍스트가 가장 긴 문항 + 정답 클립이 여러 개인 문항**을 넣어 최대 메모리를 먼저 확인한다.
- 시간 단축: 두 장은 모델을 나눠 담을 뿐 병렬로 처리하지 않는다. gpu3의 나머지 두 장(2, 3번)이 비어 있으면 `--shard 0/2`, `--shard 1/2`로
  문항을 나눠 두 프로세스를 돌려 약 4~5시간으로 줄일 수 있다.

gpu3의 venv(`/var/tmp/minyoy/worldmm-venv`)에서 실행한다.

```bash
python experiments/oracle_crop_qa/oracle_crop.py --stage preflight
python experiments/oracle_crop_qa/oracle_crop.py --stage example
CUDA_VISIBLE_DEVICES=2,3 python -u experiments/oracle_crop_qa/oracle_crop.py --stage qa --out experiments/oracle_crop_qa/run_data_smoke --qids <긴 텍스트> <클립 많은 문항>   # smoke
CUDA_VISIBLE_DEVICES=2,3 python -u experiments/oracle_crop_qa/oracle_crop.py --stage qa
python experiments/oracle_crop_qa/analyze_crop.py
```

## 7. 한계 (미리 알고 있는 것)

- 모델 1개(Qwen3-VL-8B), greedy 1회, 피험자 1명.
- 입력 이미지는 QA 모델 안에서 최대 512×512로 줄어든다. 전체 프레임(1408 px)은 약 2.75배 줄고, crop(704 px)은 약 1.4배 줄어서
  crop은 같은 영역을 약 2배 해상도로 보여준다. "확대 효과"는 이 해상도 차이를 뜻한다.
- 프롬프트에 crop이라는 설명을 넣지 않으므로, 모델은 crop을 전체 프레임과 다른 장면으로 착각할 수 있다. 이 문제는 모든 crop 조건에 똑같이 적용된다.
- gaze 좌표는 근사값이고 y가 아래쪽으로 치우쳐 있다(평균 0.565). 중앙 crop은 (0.5, 0.5)에 고정이라 시선 평균 위치와 약간 다르다.
- `dual_offcenter`는 문항마다 이미지 수가 달라서, 선택 조건 없이 `full`과 비교하면 위치 효과와 이미지 수 효과를 구분하지 못한다.
- crop ratio는 0.5 하나다. 더 작은 ratio(0.3 등)는 이 실험 결과를 보고 따로 정한다.

## 8. 결과를 본 뒤 바꾼 것

(없음)
