# Gaze-crop 검색 실험: 착용자가 본 곳만 보면 visual retrieval이 나아지는가

## 왜 이 실험인가

A1_JAKE 120문항에 대한 visual-bottleneck 실험 결과:

| 관찰 | 값 |
|---|---|
| 텍스트 검색이 실패하는 82/120 문항에서 oracle visual 프레임의 기여 | **+12.2 %p** |
| 실제 visual 검색이 target 30초 클립을 top-3에 넣는 비율 (조건 E′) | **5%** (6/120) |
| 오디오가 필요한 문항, oracle 프레임도 못 올린 구간 (C−A) | 50/120, **−2.0 %p** |
| visual 단독 vs 텍스트 단독 (C−B) | **−4.2 %p** |

쓸 만한 시각 정보는 존재하는데(oracle로 +12.2) 검색이 거의 찾지 못합니다(5%). 이 실험은 그 검색 실패의
일부가 **프레이밍 실패**인지 묻습니다. 30초 클립 하나를 전체 화면 16프레임으로 임베딩 하나로 압축하니,
질문과 무관한 배경이 임베딩을 지배하고 있을 수 있습니다.

알고 싶은 것은 결국 둘입니다: **gaze 중심으로 자른 것이 (1) 전체 프레임 대비 retrieval을 올리는가,
(2) 그래서 LLM 답변 정확도를 올리는가.** 둘 다 재지만 순서가 있습니다.

| 단계 | 재는 것 | 비교 | 비용 |
|---|---|---|---|
| **1단계** (`run_stage1.sh`) | visual retrieval recall@k. LLM 없음 | `gazef@R` vs `full` | 풀 623클립, ~40분 |
| **2단계** (`run_stage2.sh`, 1단계 통과 시) | EgoLifeQA 정확도, 조건 E′ | 같음 | 전체 6,223클립 + LLM 평가, 3~4시간 |

먼저 recall을 보는 이유는 crop이 직접 움직이는 양이 recall이고, 정확도는 그 recall이 나중에 훨씬 비싸게
사 오는 것이기 때문입니다. 대략적인 환산: recall 5% → 40%면 텍스트 실패 82문항에서 약 +5 %p, 전체로는
+3~4 %p. 1단계에서 recall이 안 움직이면 2단계에 GPU를 쓸 이유가 없습니다.

## Arm 구성

| arm | 픽셀 | 역할 |
|---|---|---|
| `full` | 전체 화면 | 베이스라인 (기존 임베딩이 본 것과 같은 픽셀) |
| `center@R` | 정사각 crop, 한 변 = R × 짧은 변, 화면 중앙 | **핵심 대조군** |
| `gaze@R` | 같은 박스를 **클립 median gaze** 지점 중심으로 | 실험군 (약함, 아래 참고) |
| `gazef@R` | 같은 박스가 **프레임별 gaze**를 따라감 | 실험군 (본진) |
| `gaze@R` + `--gaze gaze_random.json` | 같은 박스를 무작위 위치에 | crop-anywhere 대조군 |
| `pkl` | 저자의 `visual_embeddings.pkl` 행 | 참조용 (아래 주의사항 참고) |

**핵심 비교는 `gazef@R` vs `full`입니다** — 프레임마다 시선 지점을 따라가며 자른 것이 전체 프레임보다
나은가. 그게 1단계와 2단계의 헤드라인 숫자입니다.

`gaze@R`(클립 median 1개로 박스 고정)은 헤드라인이 아닙니다. 실제 데이터에서 클립별 median gaze는
x p10–p90 **0.48–0.55**, y **0.51–0.62** — 거의 화면 중앙에 뭉쳐 있습니다. 30초간의 saccade를 median으로
평균 내면 중앙으로 수렴하기 때문입니다. 그래서 `gaze@R`은 구조적으로 `center@R`에 가깝고, 시선의 위치
정보는 프레임별로 따라갈 때만 살아남습니다.

그럼에도 두 arm을 같이 돌리는 이유는 **`gazef@R`이 지는 두 가지 경로를 갈라내기 위해서**입니다.
박스가 프레임마다 튀면 16프레임이 연속된 영상처럼 보이지 않고, VLM2Vec은 연속 영상으로 학습된 모델이라
위치를 정확히 맞춰도 그 흔들림 때문에 임베딩이 나빠질 수 있습니다. `gaze@R`은 박스가 고정이라 이 손실이
없는 쪽이고, 게다가 median y가 0.561(거의 모든 클립이 중앙보다 아래 — 손과 책상을 보니까)이므로
"유용한 영역이 중앙보다 체계적으로 아래인가"라는 값싼 가설도 함께 테스트합니다.

`center@R`은 **crop 자체의 효과**를 떼어내는 대조군입니다. 1인칭 영상은 중심 편향이 강해서 crop만으로도
대상이 커지고 배경이 빠집니다. `center@R`이 없으면 `gazef@R > full`이 나와도 gaze 덕인지 crop 덕인지 말할
수 없습니다. arm 하나 추가 비용은 1단계에서 약 8분이니 뺄 이유가 없습니다.

픽셀 외의 모든 것은 고정입니다: 같은 모델(VLM2Vec-V2.0), 같은 16프레임(`../visual_bottleneck/
extract_oracle_frames.py`의 `frame_indices_uniform` 규칙), 같은 `max_pixels = 360*420`, 같은
`encode_video` 호출, 같은 쿼리, 같은 candidate set. 프레임 리스트를 `{"video": [PIL 프레임], ...}` 형태로
`encode_video`에 넘기면 `qwen_vl_utils.fetch_video`가 이를 비디오로 처리하므로, 저자들의 임베딩 절차를
다시 구현하는 부분은 없습니다.

## 평가 풀

arm마다 6,223 클립 전부를 재임베딩할 필요는 없습니다. 풀은 120문항 subset의 target 클립 전부(123개)에
**같은 날에서 뽑은** 무작위 distractor 500개를 더한 것입니다. 같은 집, 같은 사람들이라 배경만으로
갈라지지 않습니다. 문항별 후보는 `ts_end <= query_time`인 클립으로 제한되므로 미래 클립은 절대 랭킹에
들어가지 않고, 중간값 기준 한 문항이 약 320개 클립과 경쟁합니다.

1-in-~320에서 `full`을 못 이기는 arm은 1-in-3000에서도 못 이깁니다. 작은 풀을 쓰는 이유가 그것입니다 —
값싼 kill switch.

2단계(정확도)는 사정이 다릅니다. 실제 E′는 6,223클립 전부에서 top-3을 뽑으므로, 623개 인덱스로 낸 정확도는
검색기에게 비현실적으로 쉬운 문제를 준 것이고 기존 E′ 45.8%와 비교할 수 없습니다. 그래서 2단계 전에
`build_pool.py --all-clips`로 풀을 전체로 바꿉니다.

| | 1단계 (recall) | 2단계 (정확도) |
|---|---|---|
| 풀 | 623클립 (target 123 + distractor 500) | 6,223클립 전부 |
| 문항당 경쟁 클립(중앙값) | 323 | 2,561 |
| 비용 | arm당 ~10분 | arm당 ~45~90분 |

**전체 재임베딩은 처음부터 다시가 아닙니다.** `embed_arms.py`는 클립 키로 이어받으므로 1단계의 623개는
그대로 쓰이고 나머지 5,600개만 추가됩니다. 그리고 전체 풀 임베딩이 끝나면 같은 파일로 `recall_eval.py`를
다시 돌려 **실제 인덱스 크기에서의 recall**도 얻습니다 — 그 숫자가 기존 E′의 recall@3 = 5%와 직접 비교되는
값입니다.

## 실행

### 데이터 준비

gaze는 영상과 별도 HF 데이터셋입니다. 클립 mp4와 **같은 이름의 CSV**가 `DAY1..DAY7` 폴더에 들어 있습니다
(30초당 300행, 10 Hz):

```bash
huggingface-cli download Wangtwohappy/EgoLife_EyeTracking_EyeGaze --repo-type dataset \
    --include "EyeGaze/A1_JAKE/*" --local-dir data/EgoLife
#   -> data/EgoLife/EyeGaze/A1_JAKE/DAY{1..7}/<클립이름>.csv   (스크립트 기본 경로)
```

CSV는 **픽셀 좌표가 아니라 CPF 기준 yaw/pitch 라디안**입니다:

```
tracking_timestamp_us, left_yaw_rads_cpf, right_yaw_rads_cpf, pitch_rads_cpf, depth_m, ...
```

`prepare_gaze.py`가 처리하는 것:

- **좌/우 눈 yaw 평균** → cyclopean 시선 방향 (이 데이터의 좌우 시차 중앙값은 6.2°)
- **시각 재기준화**: `tracking_timestamp_us`는 기기 클럭(여기서는 ~21042초)입니다. 파일이 클립 하나를
  덮으므로 첫 샘플을 t=0으로 옮깁니다(검증: 스팬 29.9초, 간격 100 ms).
- **투영**: 기본 equidistant(`r = f·θ`, Aria RGB는 ~110° 어안이라 pinhole은 주변부를 과하게 밀어냄),
  focal 기본값은 Aria RGB 611 px @1408을 프레임 폭에 맞춰 스케일. `--projection pinhole`, `--focal-px`,
  `--hfov`로 바꿀 수 있습니다.
- 프레임 밖 시선은 버리지 않고 **가장자리로 clamp** — 시선이 간 방향으로 crop 박스가 따라가야 하므로.
  clamp 비율을 출력하니, 이 값이 크면 투영이나 transform이 틀린 것입니다.

### 두 스크립트

두 스크립트로 나뉘어 있습니다. 1단계는 ~40분, 2단계는 3~4시간이고, **2단계는 1단계 결과를 보고 돌립니다.**

```bash
# 1단계 1차 (~10분): transform 후보를 그려놓고 멈춤. 내가 보고 고른다
bash experiments/gaze_crop/run_stage1.sh
#   -> variants/*_variants.jpg 를 열어 박스가 맞는 타일의 라벨 확인

# 1단계 2차 (~30분): 고른 transform 으로 실제 측정
nohup GAZE_TRANSFORM=rot90cw bash experiments/gaze_crop/run_stage1.sh > /dev/null 2>&1 &
cat experiments/gaze_crop/analysis/DIGEST.md

# 2단계 (3~4시간): 전체 풀 확장 + E′ 정확도. 1단계에서 이긴 arm을 자동으로 가져감
nohup bash experiments/gaze_crop/run_stage2.sh > /dev/null 2>&1 &
cat experiments/gaze_crop/analysis/DIGEST.md
```

`run_stage1.sh`는 **`GAZE_TRANSFORM` 없이 부르면 transform 후보만 그려놓고 멈춥니다**(`compare_transforms.py`
→ 점수표 + `variants/*.jpg`). 라벨을 골라 다시 부르면: preflight → 풀(623) → gaze 준비(+overlay 렌더) →
arm 임베딩(`full`, `center@R`, `gaze@R`, `gazef@R`, `pkl`, 무작위 대조군) → recall ×2 → **판정 출력**.
마지막에 `gate.py`가 gaze 계열이 `full`을 이겼는지 찍고 다음 명령을 알려줍니다. 2단계는 직접 돌리지 않습니다.

`run_stage2.sh`: preflight(1단계 결과, 조건 B의 `text_context` 120개) → 게이트 재확인 → 풀(6,223) →
**이긴 arm만** 확장(623개는 재사용, arm당 ~5,600개 추가) → 실제 인덱스 크기 recall → `export_pkl` →
`eval_egolife --condition E_prime` ×2 arm → 재채점 → 다이제스트.

주요 환경 변수:

| 변수 | 기본 | 뜻 |
|---|---|---|
| `VIDEO_ROOT` | `/datasets/EgoLife` | mp4 루트 |
| `GAZE_ROOT` | `data/EgoLife/EyeGaze/A1_JAKE` | gaze CSV 루트 (`WORLDMM_GAZE_ROOT` 로도 지정 가능) |
| `RATIO` | `0.5` | crop 비율 (1단계) |
| `GAZE_TRANSFORM` | (없으면 후보만 그리고 정지) | 내가 고른 transform. 실제 측정에는 필수 |
| `ARM` | (1단계 승자) | 2단계로 가져갈 arm |
| `FORCE=1` | – | 1단계가 실패해도 2단계 강행 |
| `MIN_GAIN_PP` | `0.0` | 2단계로 넘어가는 데 필요한 recall 이득(%p) |
| `SKIP_QA=1` | – | 2단계에서 임베딩·recall만, LLM 없이 |
| `SKIP_CONTROLS=1` | – | 1단계에서 `pkl`·무작위 대조군 생략 |

판정만 따로 보려면 `python experiments/gaze_crop/gate.py` (종료 코드 0=통과, 1=실패, 2=판정 불가).

전 단계가 이어받기 가능합니다. **transform을 바꿔서 다시 돌리면** 해당 gaze arm은 이어받지 않고
자동으로 처음부터 다시 만듭니다(경고 출력). 이어받으면 한 npz 안에 좌표계가 다른 임베딩이 섞이고, 그건
나중에 알 방법이 없기 때문입니다.

### transform 고르기 (`compare_transforms.py`)

gaze CSV에는 픽셀이 아니라 **머리 기준 각도**(yaw = 좌우, pitch = 상하)가 들어 있습니다. 이걸 화면 좌표로
옮기려면 "yaw가 음수면 화면 왼쪽인가 오른쪽인가", "영상이 센서 기준으로 돌아가 있나"를 알아야 합니다.
Aria 문서가 좌표계를 정의하지만 EgoLife mp4는 재인코딩된 영상이라 그 과정의 회전·반전을 문서로 알 수
없습니다. 틀리면 착용자가 화면 왼쪽 아래의 프라이팬을 봤을 때 박스가 천장에 올라가고, 결과는 "gaze는
도움이 안 된다"로 나옵니다 — 사실은 버그인데.

그래서 후보 6개(`none, flipx, flipy, flipx+flipy, rot90cw, rot90ccw`) 중 **사람이 보고 고릅니다.**
`compare_transforms.py`는 고르지 않고, 판단 재료만 만듭니다:

1. **그림** — `variants/<clip>_variants.jpg`: 같은 프레임에 후보별로 시선 점과 crop 박스를 그려 라벨과 함께
   격자로 배치합니다. 프레임은 **그 클립에서 시선이 중앙에서 가장 멀었던 순간**을 씁니다(후보 간 차이가
   가장 큰 지점). 박스가 "그 사람이 보고 있던 것"에 올라간 타일을 고르면 됩니다.
2. **점수표(참고)** — 후보별로 crop해서 임베딩하고 **그 클립 자신의 30초 캡션**과의 코사인 평균을 냅니다.
   캡션이 그 30초에 무엇을 했는지 서술하니, 박스가 제대로 올라간 후보가 자기 캡션에 더 가까워야 합니다.
   클립별 1위 투표 수와, 1·2위 격차의 부트스트랩 95% CI도 함께 출력합니다.

점수를 **선택 기준이 아니라 참고로만** 두는 이유가 둘입니다. 하나, 이 데이터에서 실제로 구분에 실패했습니다
— 6개 후보가 0.3036~0.3165로 붙고 1·2위 격차 CI가 0을 포함했습니다(tie). 둘, 실험의 지표(recall)로 고르는
것은 애초에 금지입니다 — 이기는 후보를 골라놓고 이겼다고 보고하는 셈이니까요. 그래서 점수도 QA 질문·정답
라벨·검색 랭킹을 일절 쓰지 않고 캡션만 씁니다.

판별력을 위한 장치 둘: (1) 프레임별로 각 프레임의 시선 지점에서 crop합니다 — 클립 median으로 하면 박스가
거의 안 움직여 후보가 다 같아집니다. (2) 시선이 중앙을 거의 벗어나지 않는 클립은 제외합니다
(`--min-offset`, 기본 0.05). 후보 변환은 모두 화면 중심에 대한 등거리 변환이라 이 필터는 후보마다 동일한
값이고 특정 후보를 편들 수 없습니다.

```bash
# 후보 비교만 (run_stage1.sh 를 GAZE_TRANSFORM 없이 불러도 같은 일을 한다)
python experiments/gaze_crop/compare_transforms.py

# 어느 타일도 맞지 않으면 투영 자체가 틀렸을 수 있다
python experiments/gaze_crop/compare_transforms.py \
    --projection pinhole                      # 또는 --focal-px 로 조정
```

고른 값은 `GAZE_TRANSFORM`으로 넘깁니다. 측정에 쓰인 값은 `gaze_points.json`과 각 arm의
`emb/<arm>.meta.json`에 기록되고, `overlay/*.jpg`가 그 선택의 증거로 남습니다. 나중에 다른 값으로 바꿔
돌리면 해당 gaze arm은 이어받지 않고 자동으로 다시 만듭니다.

### 스크립트 없이 단계별로

```bash
# 1단계
python experiments/gaze_crop/check_gaze.py
python experiments/gaze_crop/build_pool.py --n-distractors 500
python experiments/gaze_crop/compare_transforms.py  # GPU, ~10분
#   -> variants/*.jpg 를 보고 transform 을 고른 뒤
python experiments/gaze_crop/prepare_gaze.py --transform <고른 값> --dump-overlay 6
python experiments/gaze_crop/embed_arms.py \
    --arms full center@0.5 gaze@0.5 gazef@0.5                                       # GPU, ~25분
python experiments/gaze_crop/recall_eval.py --query-source question
python experiments/gaze_crop/gate.py            # 0=통과 1=실패 2=판정불가, BEST=<arm> 출력

# 2단계 (위 gate 가 통과했을 때)
python experiments/gaze_crop/build_pool.py --all-clips --out experiments/gaze_crop/pool_all.json
python experiments/gaze_crop/prepare_gaze.py --pool experiments/gaze_crop/pool_all.json \
    --transform <같은 값>
python experiments/gaze_crop/embed_arms.py --pool experiments/gaze_crop/pool_all.json \
    --arms full gazef@0.5                                                           # GPU, ~2시간
python experiments/gaze_crop/recall_eval.py --pool experiments/gaze_crop/pool_all.json \
    --arms full gazef@0.5 --out experiments/gaze_crop/results/recall_allclips.json \
    --markdown experiments/gaze_crop/analysis/recall_allclips.md
python experiments/gaze_crop/export_pkl.py --arm gazef@0.5 \
    --pool experiments/gaze_crop/pool_all.json
python experiments/gaze_crop/export_pkl.py --arm full --pool experiments/gaze_crop/pool_all.json
for ARM in full gazef_05; do                                                      # arm당 실측 34분
  python experiments/visual_bottleneck/eval_egolife.py --condition E_prime \
      --visual-path experiments/gaze_crop/emb/$ARM.pkl \
      --results-dir experiments/gaze_crop/results_qa/$ARM \
      --video-root /datasets/EgoLife --resume
done
python experiments/gaze_crop/rescore.py experiments/gaze_crop/results_qa/*/E_prime.json
python experiments/gaze_crop/digest.py
```

비용: 1단계는 623 클립 × 16 프레임 × 4 arm ≈ 2,500회 forward, 디코딩 포함 GPU 1장에서 20~30분.
`embed_arms.py`는 50클립마다 체크포인트하고 `clip/s`·ETA를 출력합니다.

GPU 없이 도는 것: `check_gaze.py`, `build_pool.py`, `prepare_gaze.py`, `export_pkl.py`, `gate.py`,
`rescore.py`, `digest.py`, `embed_arms.py --arms pkl`, `embed_arms.py --dry-run`,
`recall_eval.py --self-test`.

### Aria 공식 가이드와의 대조

Meta의 MPS Eye Gaze 가이드(`mps.read_eyegaze` → `compute_depth_and_combined_gaze_direction` →
`get_gaze_vector_reprojection`)와 이 구현의 관계:

| 공식 단계 | 이 구현 | 상태 |
|---|---|---|
| `compute_depth_and_combined_gaze_direction(left_yaw, right_yaw, pitch)` | 같은 기하학을 직접 계산: 두 눈이 CPF x축 위 `[±0.0315,0,0]`에 있으므로 결합 방향 = `atan((tan yaw_L + tan yaw_R)/2)`, vergence depth = `0.063/(tan yaw_R − tan yaw_L)`, pitch는 공통이라 그대로 | **일치** |
| `get_eyegaze_point_at_depth` / depth 사용 | `--cpf-offset`을 줬을 때만 사용(3D 점 → 카메라 프레임 → 투영). 기본값 0,0,0에서는 depth가 투영을 바꿀 수 없으므로 안 씀 | **자기정합적** |
| `get_gaze_vector_reprojection` (fisheye624 + `get_transform_cpf_sensor`) | 단일 focal + equidistant 근사. VRS가 없어 실제 캘리브레이션 불가 | **근사** |
| VRS가 있을 때 | `--aria-vrs <file>`로 `projectaria_tools`의 공식 경로를 그대로 사용 | **구현됨**(EgoLife에 VRS가 없어 미검증) |

**부호 규약은 데이터로 검증했습니다.** vergence로 역산한 depth를 CSV의 `depth_m` 컬럼과 비교하니 중앙
오차 **0.011 m** — 좌/우 컬럼과 부호를 문서대로 읽고 있다는 뜻입니다. `prepare_gaze.py`가 매 실행 이
수치를 출력하고 `gaze_points.json`에 기록하니, 다른 피험자·다른 날짜에서도 확인하세요. 이 값이 크면 투영이
아니라 컬럼 해석이 틀린 것입니다.

**남는 오차는 캘리브레이션 부재에서 옵니다.** CPF→RGB 평행이동(~3 cm 가정)을 무시하면 시차 오차가
depth 0.59 m에서 2.9° = 31 px, 0.3 m에서 5.7° = 61 px입니다(f=611 기준). crop 박스가 704 px이니 박스가
다른 물체로 옮겨갈 정도는 아니지만 공짜도 아닙니다. 실제 캘리브레이션 값을 얻으면
`--cpf-offset x,y,z`로 넣을 수 있고, 그때만 depth가 쓰입니다(`--default-depth`, 문서와 같이 기본 1.0 m).
참고로 제가 각도 평균으로 결합했던 초기 구현은 공식 tangent 평균과 최대 1.64° = 20 px 차이가 났고, 위 표의
계산으로 교체했습니다.

## 산출물

| 경로 | 내용 |
|---|---|
| `analysis/DIGEST.md` | **먼저 볼 것.** 1·2단계 숫자와 넘어야 할 기준선 한 화면 |
| `results/recall_question.json`, `recall_keywords.json` | 1단계 recall (623클립 풀), 문항별 순위 포함 |
| `results/recall_allclips.json` | 2단계 recall (6,223클립 = 실제 인덱스 크기) |
| `results_qa/<arm>/E_prime.json` | 2단계 정확도. 문항별 응답까지 |
| `analysis/transform_choice.json` | transform 후보별 점수, 승자, tie 여부 |
| `variants/*.jpg`, `overlay/*.jpg` | gaze 박스를 실제 프레임에 그린 것. transform 검증 증거 |
| `emb/<arm>.npz`, `<arm>.meta.json` | arm별 임베딩과 그것을 만든 설정(transform 포함) |
| `logs/stage{1,2}_*.log` | 전체 로그 |

## 결과 읽는 법

`k=3`(WorldMM의 visual top-k, E′가 실제로 소비한 값)에서, 모든 arm이 채점 가능한 문항만 대상으로:

| 결과 | 해석 | 다음 |
|---|---|---|
| `gazef@R` > `gaze@R` ≈ `center@R` | 프레임별 추종이 이득. 시선의 순간 위치가 정보다 | ratio 스윕 → 2단계 |
| `gazef@R` < `gaze@R` ≈ `center@R` | 박스 흔들림 손해가 위치 이득보다 큼 | 고정 박스 계열로 2단계 |
| `gaze@R` > `center@R`, 둘 다 `gazef@R`보다 높음 | 시선 추적이 아니라 "아래로 치우친 고정 crop"이 답 | 오프셋 crop을 별도 arm으로 |
| 셋 다 ≈ `full` | 프레이밍이 병목이 아님 | 중단. `../visual_bottleneck/dpr_recall.py`의 oracle 쿼리 점검으로 |

`gazef@R` 하나만 돌리면 첫 두 줄이 갈라지지 않습니다 — 졌을 때 "시선이 쓸모없다"인지 "흔들림 때문에
졌다"인지 알 수 없습니다. 무작위 위치 대조군(`gaze_random.json`)은 그 아래를 받칩니다: `gazef@R`이 무작위
crop도 못 이기면 위치 정보가 전혀 쓰이지 않고 있다는 뜻입니다.

차이는 paired(같은 문항, 같은 쿼리 임베딩, 같은 candidate set)로 재고, 95% 부트스트랩 CI와 불일치 개수
(`a only` / `b only`)를 함께 출력합니다. 이 표본 크기에서는 몇 문항이 뒤집힌 것도 추세처럼 보이기 때문입니다.

아무리 crop을 잘해도 움직이지 않는 천장이 둘 있습니다:

- **50/120 문항이 오디오를 필요로 하고**, 그 문항들에서 oracle 프레임은 −2.0 %p였습니다. 그래서
  `recall_eval.py`가 recall을 `need_audio`로 분해합니다. 실제 여유분은 `False` 행으로 읽으세요.
- **visual 단독은 텍스트 단독보다 4.2 %p 낮습니다.** gaze로 visual이 텍스트를 대체하게 만들 수는 없고,
  싸움은 보완재로서의 +12.2 %p 구간을 얼마나 회수하느냐입니다.

## 2단계: LLM 답변 정확도

1단계에서 gaze 계열 arm이 `full`을 이겼을 때만 진행합니다(`gate.py`가 판정). 평가기를 새로 쓰지 않고, 기존
`visual_bottleneck/eval_egolife.py`의 조건 **E′**를 그대로 씁니다. E′ = B가 캐시해 둔 텍스트 컨텍스트 +
질문 텍스트로 뽑은 visual top-3. 여기서 **visual 인덱스만** arm의 임베딩으로 바꿔 끼웁니다
(`export_pkl.py` → `--visual-path`). `run_stage2.sh`가 이 전부를 순서대로 하고, 손으로 하려면 위
「스크립트 없이 단계별로」의 2단계 블록입니다.

바뀌지 않는 것 셋:

- **풀은 6,223클립 전부**여야 합니다. 623개 인덱스로 낸 정확도는 검색이 비현실적으로 쉬워 기존 E′
  45.8%와 비교할 수 없습니다.
- **`--robust-reasoning`은 켜지 않습니다.** 기존 E′와 플래그를 맞추기 위해서이고, E′에서는 채점만 바꾸는데
  그 효과가 0pp로 측정됐습니다(아래 표).
- **LLM에게 가는 프레임은 잘리지 않은 전체 프레임**입니다(검색된 클립의 1 fps 프레임, E′ 원래 동작).

**파서**: 답 채점은 사후 처리이므로 arm 비교가 파서 선택에 흔들리는지는 다시 돌리지 않고 확인합니다.
기존 실행을 두 파서로 재채점한 결과:

| 조건 | strict | lenient | 차이 |
|---|---|---|---|
| E′ | 45.8% | 45.8% | 0pp (0/120 문항) |
| B | 40.8% | 41.7% | +0.8pp (1문항) |
| E (`--robust-reasoning` 실행) | 44.2% | 45.0% | +0.8pp (1문항) |

E′의 45.8%는 파서와 무관한 숫자라 기준선으로 그대로 쓸 수 있습니다. E가 41.7% → 45.0%로 오른 것은 파서가
아니라 `--robust-reasoning`이 **라운드 결정 JSON 파싱**까지 느슨하게 해서 검색 루프 자체가 달라진
결과입니다(다른 실행, +2.5pp) — E′는 루프를 안 타므로 이 영향이 없습니다.

### 왜 E가 아니라 E′인가

E(원래 에이전트 루프)에서는 8B가 visual을 **120문항 중 2문항에서만** 호출했습니다(호출 8회).
`--robust-reasoning`으로 파서를 느슨하게 해도 2문항 그대로이므로 파싱 실패가 아니라 라우팅 판단 자체의
문제입니다. 그 상태로는 검색 품질을 아무리 올려도 E의 end-to-end 숫자는 움직이지 않습니다.

| 조건 | visual 사용 문항 | 문항당 프레임(중앙값) | 정확도 |
|---|---|---|---|
| B (텍스트만) | 0 | 0 | 40.8% |
| E (에이전트가 스스로 결정) | 2 / 120 | 0 | 41.7% |
| E′ (visual 호출 강제) | 120 / 120 | 63 | 45.8% |

즉 병목이 둘이고 서로 독립입니다: **(1) 검색 품질**(불러와도 엉뚱한 클립, recall 5%)과
**(2) 호출 라우팅**(부를 생각을 안 함, 2/120). 이 실험은 (1)만 다루고, E′가 (2)를 제거한 harness입니다.
gaze arm의 주장은 "visual을 호출했을 때 그 호출이 더 쓸모 있어지는가"(45.8% → ?)이며, E 기준 end-to-end
향상이 안 나오는 것은 gaze의 실패가 아니라 라우팅의 실패입니다. (2)는 프롬프트/라우팅 수정으로 별도로
고칠 문제입니다 — 그 극단적 형태가 바로 E′입니다.

비교 기준선(A1_JAKE 120문항, 기존 실행):

| 조건 | 정확도 |
|---|---|
| B (텍스트만) | 40.8% |
| E′ (텍스트 + visual top-3, 저자 임베딩) | 45.8% |

즉 recall 5%인 현재 상태로도 visual top-3은 이미 +5.0 %p를 내고 있습니다. gaze arm이 넘어야 할 선은
40.8%가 아니라 **45.8%**입니다. 그리고 `--all-clips` 없이 샘플 풀로 낸 정확도는 이 45.8%와 비교할 수
없습니다(검색 난이도가 다름). 같은 풀에서 내보낸 `full` arm과만 비교하세요 — `export_pkl.py`가 그 경고를
출력합니다.

## 주의사항

- **`pkl`은 arm이 아니라 참조점입니다.** 그 행들은 `fetch_video`의 mp4 경로를 타서 프레임 샘플링을 스스로
  하고, `full`은 프레임 리스트 경로를 탑니다. 둘 사이의 작은 차이는 예상된 것이며 비교 대상이 아닙니다.
  비교는 항상 `full` 기준으로 하세요.
- **gaze 기하는 근사입니다.** 결합 방향과 vergence depth는 Aria 문서의 기하학과 일치하지만(검증: depth
  오차 0.011 m), 픽셀 투영은 캘리브레이션 없이 단일 focal + equidistant입니다. CPF→카메라 오프셋 무시로
  median depth 0.59 m에서 ~2.9° = 31 px. transform은 `compare_transforms.py`가 제시한 그림을 보고 **사람이** 고릅니다.
  **고르지 않은 채 측정한 gaze arm 숫자는 숫자가 아닙니다** — 잘못 회전된 crop을 40분
  동안 측정한 것일 뿐입니다. `overlay/*.jpg`는 결과와 함께 보관하세요.
- **gaze가 없는 클립은 gaze arm에서 빠집니다.** 중앙 crop으로 조용히 대체하지 않습니다.
  `recall_eval.py`가 누락 개수를 보고하고, paired 비교는 모든 arm이 채점 가능한 문항에서만 합니다.
- 피험자 1명(A1_JAKE), 임베딩 모델 1개. 120문항 subset과 그 target 클립은
  `../visual_bottleneck/subset.json`에서 오므로 두 실험은 같은 클립을 이야기합니다.

## 파일

| 파일 | GPU | 내용 |
|---|---|---|
| `check_gaze.py` | 불필요 | 0단계: gaze 루트를 훑어 클립과 이름이 맞는 CSV 수·헤더 출력 |
| `build_pool.py` | 불필요 | `pool.json`: target 123 + 같은 날 distractor 500 + 120문항 |
| `prepare_gaze.py` | 불필요 | 임의 gaze 포맷 → 정규화된 `gaze_points.json`; `--pseudo`, `--dump-overlay` |
| `embed_arms.py` | 필요 | 1회 디코딩 → arm별 crop → `encode_video` → `emb/<arm>.npz` (이어받기 가능) |
| `recall_eval.py` | 필요¹ | arm별 recall@k, paired CI, 분해표 → `results/`, `analysis/` |
| `export_pkl.py` | 불필요 | arm 임베딩 → `visual_embeddings.pkl` 형식. 2단계 정확도 실행용 |
| `rescore.py` | 불필요 | 저장된 응답을 strict/lenient 두 파서로 재채점. "파서 덕에 이겼나" 확인용 |
| `compare_transforms.py` | 필요 | transform 후보를 그림 + 점수표로 제시. **고르는 건 사람** |
| `digest.py` | 불필요 | 1·2단계 결과를 한 화면으로 → `analysis/DIGEST.md` |
| `gaze_common.py` | 불필요 | 경로·클립 키·crop 박스·gaze 조회. `../visual_bottleneck/common.py` 재사용 |
| `gate.py` | 불필요 | 1단계 판정: gaze 계열이 `full`을 이겼나 (종료 코드로 분기) |
| `run_stage1.sh` | 필요 | 1단계 전 과정 (~40분) |
| `run_stage2.sh` | 필요 | 2단계 전 과정 (3~4시간), 1단계 승자를 자동으로 가져감 |

¹ 쿼리 인코딩에만 필요하고 `.cache_queries.npz`에 캐시됩니다. `--self-test`는 이 단계를 건너뜁니다.
