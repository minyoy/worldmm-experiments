# fixation frame selection

`gaze_crop` 은 프레임마다 **가장 가까운 gaze 샘플 하나**만 쓰고 10 Hz 의 나머지를 버린다
(`gaze_common.gaze_at()`). 그 샘플이 saccade 한가운데면 crop 박스는 아무 의미 없는 지점을 잡는다.

여기서는 10 Hz 전체를 써서 **fixation** — 같은 위치에 머물고(dispersion), 충분히 오래
지속되는(duration) 구간 — 을 검출하고, 그게 crop 대상으로 쓸 만한지 판단한다.

## 파일

| | |
|---|---|
| `fixation_sweep.py` | 검출 임계값 sweep. 클립당 fixation 개수 / duration / **프레임 커버리지** |
| `plot_long_fixations.py` | 긴 fixation 을 눈으로 확인. 진짜 fixation 인가, smooth pursuit 인가 |
| `streamgaze/` | 외부 참고 코드(EGTEA / Ego4D / HoloAssist). `preprocess/gaze_processing.py` 의 `extract_fixation_segments` 를 이 실험이 쓴다 |
| `embed_fix_arms.py` | fixation 프레임만 골라 임베딩. `fixsel` / `fixsel_gazef@R` / `fixsel_ctl` arm |
| `run_recall.sh` | 위 arm 들을 `full` / `gazef@R` 와 recall 로 비교 |

`gaze_points.json`, `pool.json`, `gaze_common.py` 는 **`../gaze_crop` 에 그대로 두고 읽기만 한다.**
두 실험이 같은 클립·같은 gaze 를 말하게 하려는 것이고, 새 파일을 만들지 않는다.

## `extract_fixation_segments` 에 가한 수정

streamgaze 원본에는 균일 샘플링 데이터에서 **fixation 이 절대 끝나지 않는** 버그가 있었다.

```python
if dist > radius_thresh:
    gap_duration = timestamps[i] - timestamps[i-1]   # 연속 샘플 간 dt
    if gap_duration <= gap_thresh:                   # 10 Hz 에서 항상 0.1 <= 0.2 -> 항상 참
        i += 1; continue                             # -> 클립 전체가 fixation 1 개
```

30 초 동안 두 지점을 1 초씩 번갈아 보는 합성 신호로 확인: 원본은 `n_fix = 1` (29.9 초),
수정본은 `n_fix = 30`. streamgaze 쪽에서 이게 드러나지 않은 건 invalid 샘플을 버려서 실제
시간 구멍이 생기는 데이터였기 때문이다.

수정 내용:

- **`gap_thresh` 의 의미** — 이탈이 시작된 지점부터 시간을 재고, 반경 안으로 돌아오면 리셋.
  이제 진짜로 "반경 밖에 머문 시간" 이다.
- **`dropout_thresh` 신설 (기본 0.4 초)** — 샘플 결손(눈깜빡임·트래킹 실패)은 반경 이탈과
  성격이 다르다. 눈깜빡임은 flick 보다 길어서 임계도 따로 둔다. 결손 구간은 centroid 평균에서 빠진다.
- **거리 기준을 첫 샘플 → running centroid 로** — 첫 샘플 노이즈에 구간 전체가 끌려가지 않는다.
  용서된 이탈 샘플은 centroid 계산에서 제외한다.

## 결과: 임계값

A1_JAKE 6222 클립, `radius=0.05  min_dur=0.3s  dropout=0.4s`:

| gap | fix/clip | dur_med | dur_p90 | coverage | >3s |
|---|---|---|---|---|---|
| 0.1 | 20.37 | 0.70s | 2.20s | 73.7% | 5.4% |
| 0.2 | 18.54 | 0.80s | 2.50s | 74.5% | 6.7% |
| 0.3 | 16.56 | 0.90s | 2.80s | 75.0% | 8.5% |

**무릎(knee)이 없다.** 세 지표가 전부 매끄럽게 단조 변화한다. 그리고 결정적으로 **coverage 가
거의 안 움직인다**(총 1.3 pp). 0.1 이 fixation 을 쪼개고 있었다면 gap 을 늘려 붙일 때 그 사이
프레임들이 새로 들어오면서 coverage 가 올라야 하는데, 안 올랐다. 즉 gap 을 늘려 사라진 fixation 들은
원래 프레임을 커버하지 않던 짧은 조각이고, 늘어난 건 병합이다(`>3s` 가 처음부터 계속 오르는 것과 일치).

**0.1 에서 이미 쪼개짐 구간을 벗어나 있다. gap 은 이 실험에서 중요한 파라미터가 아니다.**
`0.2` 를 쓴다 — 0.3 은 얻는 것 없이 병합만 늘고, 0.1 은 눈깜빡임 전후 튐을 못 넘겨 조각을 만든다.

## 결과: coverage 74%

원래 알고 싶었던 숫자다. `embed_arms.py` 가 crop 하는 **16 개 프레임 중 3/4 가 fixation 안에 있다.**
fixation-aware arm 을 만들면 프레임의 74 % 에서 "nearest sample" 대신 "fixation centroid" 를 쓰고,
26 % 만 폴백한다. **arm 을 만들 가치가 있다.**

정합성도 맞는다: 30 초 클립당 18.5 개 × 평균 ~1.1 초 ≈ 20 초 ≈ 67 %, coverage 74 % 와 일관된다.
1.6 초마다 시선이 옮겨가는 셈인데 요리·대화 중심 활동으로 타당하다. `no-fix clips = 0.0%` 라
폴백만 타는 죽은 클립도 없다. `dur_med 0.8s` 가 `min_dur=0.3s` 바닥에서 충분히 떨어져 있어
검출 결과가 임계값에 눌린 인공물도 아니다.

## 알려진 한계

- **I-DT 는 smooth pursuit 을 fixation 과 구별하지 못한다.** 천천히 움직이는 대상을 따라가도
  연속 샘플은 서로 가깝다. `plot_long_fixations.py` 가 이걸 검사한다 (아래).
- **10 Hz 로는 saccade 를 못 잡는다.** saccade 는 20–80 ms 라 샘플 한 개 안에서 끝난다.
  streamgaze 의 `detect_saccade_segments_ivt_dispersion` (I-VT) 은 이 데이터에서 의미가 희박하다.
  fixation 검출만 10 Hz 로 성립한다(fixation 은 보통 200 ms 이상 = 샘플 2 개 이상).
- **여기서 말하는 fixation 은 "이미지 상에서 안정된 응시" 다.** EgoLife gaze 는 eye-in-head
  yaw/pitch 를 카메라 평면에 투영한 이미지 좌표라, 머리가 돌아가는 동안 세계의 한 점을 계속 보면
  (VOR) 이미지 좌표상 gaze 는 움직인다. crop 박스 안정화가 목적이라면 오히려 이 정의가 맞다.
- **`radius` 는 각도가 아니라 이미지 거리다.** `px`/`py` 가 각각 width/height 로 정규화되어 있어
  프레임이 정사각형이 아니면 `hypot` 이 x/y 를 다른 척도로 섞는다. Aria RGB native 1408×1408
  기준 `radius=0.05` ≈ 70 px ≈ 6–7°.

## 사용법

```bash
cd experiments/fixation_frame_selection

# 임계값 sweep (GPU 불필요)
python fixation_sweep.py                                  # gap 0.1/0.2/0.3
# radius 는 한 번에 하나씩
python fixation_sweep.py --gap 0.2 --radius 0.03

# 긴 fixation 눈으로 확인 -> analysis/long_fixations.jpg
python plot_long_fixations.py                             # 가장 긴 6 개
python plot_long_fixations.py --sort straightness         # pursuit 의심 순
python plot_long_fixations.py --min-dur-shown 5 --rows 10
```

### `plot_long_fixations.py` 가 보여주는 것

sweep 의 `>3s` 가 진짜 fixation 인지 잘못 묶인 pursuit 인지 가른다. 판별 지표는 gaze 경로의 **모양**:

```
straightness = |마지막 - 처음| / (스텝 길이의 합)     # 0.3초 평활한 경로 위에서
```

**반드시 평활한 경로에서 재야 한다.** 원시 샘플로 재면 pursuit 을 전혀 못 잡는다:
분모(path)는 샘플마다의 tracker 노이즈를 전부 더하는데 분자(net)는 검출기의 `radius` 에
막혀 있어서, drift 가 있든 없든 긴 구간은 전부 0 근처로 눌린다. 순수 pursuit 을 검출기에
통과시키면 원시 경로에서 0.00–0.25 가 나온다 — "배회" 구간 안이다.

검출기에 대고 시뮬레이션으로 보정한 임계 (평활 후):

| | jitter 0.002 | 0.008 | 0.015 |
|---|---|---|---|
| **drift 0 (진짜 fixation)** | 0.05 | 0.05 | 0.05 |
| pursuit 0.005/s | 0.59 | 0.15 | 0.07 |
| pursuit 0.010/s | 0.84 | 0.32 | 0.16 |
| pursuit 0.020/s | 0.96 | 0.63 | 0.33 |

```
  <= 0.15   진짜 fixation   (drift 0 대조군이 노이즈와 무관하게 0.05)
  >= 0.30   pursuit
```

노이즈가 큰 상태의 느린 drift(0.005/s)는 여전히 숨는다. 다만 그건 6 초에 화면의 0.03 이라
crop 박스를 움직이지 못하므로 이 실험에서는 문제가 아니다.

그림에서는 노란 점이 현재 gaze, 파란 점이 이 fixation 의 이전 샘플들이다. **pursuit 이면 파란
점이 패널을 가로지르는 선을 그리고, 진짜 fixation 이면 한 덩어리로 뭉친다.** 흰 원은 검출기가
허용한 `radius`, 파란 박스는 fixation-aware arm 이 그 구간 내내 고정할 crop 이다.
그림을 그리기 전에 전체 분포(straightness p10/median/p90, pursuit 비율)를 먼저 출력한다.

## 프레임 선택 arm: `embed_fix_arms.py`

`gaze_crop` 은 16 프레임을 **어디를 crop 할지** 만 바꿨다. 여기서는 **어떤 프레임을 임베딩할지**
를 바꾼다. fixation 밖 프레임은 saccade 한가운데라 모션 블러가 끼고 주체가 보지 않던 장면이므로,
crop 하는 게 아니라 버린다. **fixation 프레임이 하나도 없는 클립은 임베딩하지 않는다** — 그 arm 에서
아예 빠지고, `recall_eval.py` 가 그 클립이 정답인 문항을 `unscorable` 로 빼준다(다른 걸 슬쩍
끼워넣지 않는다).

| arm | 프레임 | crop |
|---|---|---|
| `full` | 균일 16 | 없음 (baseline) |
| `gazef@R` | 균일 16 | 프레임별 gaze |
| `fixsel` | fixation 안에 든 것만 (~12/16) | 없음 |
| `fixsel_ctl` | `fixsel` 과 **같은 개수**, 클립 전체에 균일 | 없음 |
| `fixsel_gazef@R` | fixation 안에 든 것만 | 그 fixation 의 **centroid** |

`fixsel_ctl` 은 빼면 안 된다. 프레임을 26 % 버리는 것 자체가 인코더 입력의 변화이고(개수가
줄고 시간적으로 촘촘해진다) VLM2Vec 은 프레임을 pooling 한다. 이게 없으면 "fixsel 이 full 을
이겼다" 를 "16 프레임보다 12 프레임이 나았다" 와 분리할 수 없다. **`gazef@R` 에 `center@R` 이
했던 역할이다.** 읽는 법:

```
fixsel - fixsel_ctl   프레임 선택의 효과          <- 이게 이 실험의 결과다
fixsel - full         선택 효과 + "프레임이 줄었다" 가 섞인 값
```

crop 중심은 프레임의 최근접 gaze 샘플이 아니라 **fixation centroid** 다. fixation 안에서 샘플은
`radius` 만큼 흩어지므로 그 평균이 "보고 있던 한 점" 의 더 좋은 추정이고, 구간 내내 박스를
고정시킨다 — `gazef@R` 이 못 하는 일이 정확히 이것이다.

`--select` 는 두 가지다. `subset`(기본)은 baseline 이 쓰는 **같은** 16 개 시각 중 fixation 밖을
버리므로 `full` 과 삭제 하나만큼 다르다(개수가 클립마다 변한다). `refill` 은 fixation 시간의
합집합 위에서 16 개를 균일하게 다시 뽑아 개수를 16 으로 고정한다 — 개수 교란이 없어지지만
baseline 이 본 적 없는 시각을 보게 된다. **두 모드를 한 표에 섞지 말 것.**

```bash
# GPU 불필요: 프레임 개수 / 버려진 클립만 센다 (기본 pool.json, 623 클립)
python embed_fix_arms.py --dry-run

# 임베딩 + recall 비교. 기본이 stage 2 의 전체 풀(6,223 클립 / 500 문항)이라 ~6시간이다
nohup bash run_recall.sh > /dev/null 2>&1 &
tail -f logs/recall_*.log

POOL=../gaze_crop/pool.json bash run_recall.sh   # stage 1 의 623 클립 (~35분, 120 문항)
SELECT=refill bash run_recall.sh                 # 개수 고정 모드
RADIUS=0.03 bash run_recall.sh
SKIP_EMBED=1 bash run_recall.sh                  # emb/ 에 있는 걸로 채점만 다시
```

끊겨도 안전하다 — 50 클립마다 체크포인트하고 다시 실행하면 이어받는다. 안전하지 않은 건
중간에 `--radius`/`--select` 를 바꾸는 것이고, 그건 arm 을 버리고 처음부터 다시 만든다(의도된
동작이다).

실측 속도 (`gaze_crop/logs/`, gpu2): 클립당 ≈ 디코드 1.05s + arm 당 0.76s.

| | 623 클립 | 6,223 클립 |
|---|---|---|
| fixation arm 3개 | ~35분 | **~5.8시간** |
| `full` + `gazef@0.5` (먼저 필요) | ~27분 | ~2.7시간 |

`run_recall.sh` 는 `full`/`gazef@R` 를 **절대 다시 만들지 않는다.** 재실행이 기준선을 조용히
바꾸면 안 되기 때문이다. 대신 preflight 에서 **행 개수**까지 본다 — npz 경로에는 풀 크기가
안 적혀 있어서, stage 1 의 623클립 `full.npz` 가 6,223클립짜리가 있어야 할 자리에 그대로 앉는다.
`recall_eval.py` 는 모든 arm 이 채점 가능한 문항만 쓰므로, 그러면 표가 stage 1 클립으로
쪼그라든 채 맨 위에는 500 문항이라고 찍힌다. 그래서 풀의 90 % 미만이면 중단한다.

임베딩은 `../gaze_crop/emb/` 에 떨어진다. 같은 디렉터리에 두어야 `recall_eval.py` 가 다섯 arm 을
**같은 문항·같은 후보 집합**으로 한 표에서 짝지어 비교한다. `gaze_common.py` 는 건드리지 않았다 —
`load_arms` 가 npz 파일명을 arm 이름으로 읽기 때문에 새 arm 을 등록할 곳이 없다.

저장된 행이 다른 `transform` 이나 다른 검출 임계값으로 만들어졌으면 이어받지 않고 그 arm 을
처음부터 다시 만든다. `radius=0.03` 실행이 `radius=0.05` 행 위에 얹히면 npz 안에서 영영 안 보인다.

## 다음

`gap=0.2` 고정. `radius` 를 0.03 / 0.05 / 0.08 로 훑어 coverage 와 `>3s` 의 trade-off 를 본다
(이쪽이 지배적인 파라미터다).
