# GazeQwen 수정분 (StreamGaze 재현과 placebo 대조)

[`experiments/visual_routing_qa/GAZE_PROMPTING_REPORT.md`](../../experiments/visual_routing_qa/GAZE_PROMPTING_REPORT.md)의 GazeQwen 실험(3장 EgoLife, 4장 StreamGaze 검증)을 재현하는 데 필요한 것만 모았습니다.
GazeQwen 저장소 전체는 넣지 않았습니다. 원본을 clone한 뒤 아래 패치와 스크립트를 얹어 씁니다.

- 원본: [phamtrongthang123/gazeqwen](https://github.com/phamtrongthang123/gazeqwen), 커밋 `c3195b137e612026d6e0481b6c04b173a8cec13e`
- 라이선스: Apache 2.0 (원본의 [`LICENSE`](LICENSE))
- 체크포인트(`best_model.pt`, V-JEPA 2.1), 영상, StreamGaze QA는 포함하지 않습니다.

## 내용

| 경로 | 내용 |
|---|---|
| `patches/transformers457_compat_and_placebo.patch` | 원본 `gazeqwen/` 패키지 수정 3개 (아래) |
| `scripts/make_fixations.py` | StreamGaze step 1 출력 → GazeQwen fixation CSV (EGTEA/EgoExoLearn 컬럼 차이, `duration` 컬럼 처리) |
| `scripts/run_eval.sh` | `eval.py` + 38개 영상 split + 체크포인트 + `--resume` 래퍼 (실행 당시 서버 경로가 박혀 있음) |
| `scripts/plot_gaze_bias_grid.py`, `plot_score_decomposition.py`, `plot_gaze_vs_placebo.py` | 보고서 4.5의 그림 (`--index N`, test38 결과 순서 기준 샘플 번호) |
| `splits/test_egtea_egoexo.json` | 원본 split의 test 44개 영상 중 HoloAssist 6개를 뺀 38개 |
| `results/{gazeqwen,placebo,lora_only}_test38.json` | StreamGaze test 2,140샘플의 조건별 결과 (세 파일의 샘플 순서가 같음, 순서로 짝지어 비교) |

## 패치 내용

| 파일 | 수정 |
|---|---|
| `gazeqwen/hooks.py` | transformers ≥ 4.52에서 LLM 레이어가 `model.model.language_model.layers`로 옮겨져 hook 설치가 실패하던 문제 |
| `gazeqwen/lora.py` | LoRA 가중치 키(`model.layers.N…`)가 파라미터 이름과 맞지 않아 **LoRA가 한 번도 적용되지 않던** 문제. 키 매핑을 고치고, 불일치하면 오류를 냄 |
| `gazeqwen/eval.py` | `--placebo_seed` 추가: fixation 시각·지속시간은 유지하고 중심만 실제에서 0.25 이상 떨어진 무작위 위치로 옮김 (샘플별 고정 시드) |

## 사용

```bash
git clone https://github.com/phamtrongthang123/gazeqwen.git && cd gazeqwen
git checkout c3195b137e612026d6e0481b6c04b173a8cec13e
git apply /path/to/worldmm-experiments/external/gazeqwen/patches/transformers457_compat_and_placebo.patch
cp /path/to/worldmm-experiments/external/gazeqwen/scripts/* scripts/
cp /path/to/worldmm-experiments/external/gazeqwen/splits/test_egtea_egoexo.json splits/
```

이후 fixation 생성과 평가 명령은 보고서 8장(재현)을 따릅니다. `run_eval.sh`와 plot 스크립트에는 실행했던 서버의 경로(`/home/minyoy/...`)가
들어 있으니 환경에 맞게 바꿔 쓰세요.
