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
straightness = |마지막 - 처음| / (스텝 길이의 합)

  ~0.0-0.3   한 점 주위를 배회 ......... 진짜 fixation
  ~0.7-1.0   한 방향으로 꾸준히 행진 ... smooth pursuit, 잘못 병합
```

그림에서는 노란 점이 현재 gaze, 파란 점이 이 fixation 의 이전 샘플들이다. **pursuit 이면 파란
점이 패널을 가로지르는 선을 그리고, 진짜 fixation 이면 한 덩어리로 뭉친다.** 흰 원은 검출기가
허용한 `radius`, 파란 박스는 fixation-aware arm 이 그 구간 내내 고정할 crop 이다.
그림을 그리기 전에 전체 분포(straightness p10/median/p90, pursuit 비율)를 먼저 출력한다.

## 다음

`gap=0.2` 고정. `radius` 를 0.03 / 0.05 / 0.08 로 훑어 coverage 와 `>3s` 의 trade-off 를 본다
(이쪽이 지배적인 파라미터다). 그 다음 `gaze_crop` 에 `gazefix@R` arm 을 추가해 `gazef@R` 와
recall 로 직접 비교한다.
