# Full / gaze / query routing QA 실행

A와 B는 각각 `full.npz` / `gazef_05.npz`로 Top-3를 검색하고, 원본 / 시선 crop 이미지를 QA VLM에 준다. C는 각 arm이 검색한 프레임을 보고 고르는 Qwen 라우터가 선택한 A 또는 B의 **검색·이미지·답변을 그대로 재사용**한다. 분석에서 A/B/C와 A OR B oracle의 Hit@3·QA accuracy를 비교한다.

## gpu2의 GPU 0에서 실행

로컬 변경 사항이 `gpu2`의 `~/WorldMM`에 보이는 상태에서 실행한다. 공유 파일시스템이 아니면 먼저 새 폴더 `experiments/visual_routing_qa/`의 파일을 서버에 복사한다. 아래 명령은 기존 full/gaze 임베딩과 gaze 좌표, 사전 구축된 episodic OpenIE 캐시가 있다고 가정한다. 두 임베딩의 **공통 클립만** 후보로 사용하며, 제외된 클립 ID는 `manifest.json`과 `preflight.json`에 남긴다. 현재 gaze 임베딩에서 한 클립이 빠져 후보는 6,222개다.

`gpu2`의 `~/WorldMM`에서 `worldmm-venv`를 활성화한 뒤 실행한다. 각 단계가 끝났는지 로그에서 확인한 다음 **다음 명령을 실행**한다. 네 명령을 한꺼번에 백그라운드로 시작하면 후속 단계가 필요한 입력 없이 실행된다.

```bash
mkdir -p experiments/visual_routing_qa/logs

# 입력·임베딩·영상·gaze 파일 검사 (GPU 불필요)
python experiments/visual_routing_qa/run.py --stage preflight

# 1. 500문항 공통 텍스트 메모리
nohup env CUDA_VISIBLE_DEVICES=0 python -u experiments/visual_routing_qa/run.py --stage text \
  > experiments/visual_routing_qa/logs/text.log 2>&1 &

# 2. A/B: 각 인덱스 Top-3 검색 + 원본/gaze 이미지 QA (text 완료 후)
nohup env CUDA_VISIBLE_DEVICES=0 python -u experiments/visual_routing_qa/run.py --stage qa \
  > experiments/visual_routing_qa/logs/qa.log 2>&1 &

# 3. C: Qwen 질문 라우팅 + 선택된 A/B 결과 조합 (qa 완료 후)
nohup env CUDA_VISIBLE_DEVICES=0 python -u experiments/visual_routing_qa/run.py --stage route \
  > experiments/visual_routing_qa/logs/route.log 2>&1 &

# 4. Hit@3, QA accuracy, oracle 집계 (route 완료 후; GPU 사용 없음)
nohup env CUDA_VISIBLE_DEVICES=0 python -u experiments/visual_routing_qa/analyze.py \
  > experiments/visual_routing_qa/logs/analyze.log 2>&1 &
```

진행 상황은 `tail -f experiments/visual_routing_qa/logs/<단계>.log`로 확인한다. 예: `tail -f experiments/visual_routing_qa/logs/qa.log`.

결과는 `experiments/visual_routing_qa/run_data/analysis/summary.md`와 `per_question.csv`에 생성된다. `run.py`는 완료된 문항 파일을 재사용하므로 중단 후 같은 명령으로 이어서 실행할 수 있다. 입력 파일·모델·문항 범위가 달라지면 `--out`으로 별도 결과 폴더를 지정한다.

2문항 배선 점검은 **별도 출력 폴더**에서 한다. 본 실험 결과와 manifest가 섞이지 않도록 한다.

```bash
python experiments/visual_routing_qa/run.py --stage preflight --limit 2 --out experiments/visual_routing_qa/smoke_data
python experiments/visual_routing_qa/run.py --stage text      --limit 2 --out experiments/visual_routing_qa/smoke_data
python experiments/visual_routing_qa/run.py --stage qa        --limit 2 --out experiments/visual_routing_qa/smoke_data
python experiments/visual_routing_qa/run.py --stage route     --limit 2 --out experiments/visual_routing_qa/smoke_data
python experiments/visual_routing_qa/analyze.py --out experiments/visual_routing_qa/smoke_data
```

`preflight.json`은 입력 결손과 정답 시점 매핑 불가 문항을 기록한다. 공통 후보 클립 중 영상이나 gaze sample이 없으면 preflight가 중단되므로 원인을 먼저 해결한다. A/B에서 프레임 디코딩 또는 QA 오류가 나면 해당 문항의 JSON에 `status=error`가 기록되고, C는 A/B가 모두 성공할 때만 실행된다. 실패 문항을 다시 시도하려면 해당 오류 결과 파일만 삭제한 뒤 `--stage qa`를 다시 실행한다.

## D, E: dual-view 조건 (`--stage qa`에 포함)

A/B의 검색 결과(`results/full`, `results/gaze`의 `top3`)를 그대로 쓰고, QA 입력의 시각 증거만 바꾼다. 텍스트 메모리와 QA 호출은 A/B와 같고 시각 증거는 항상 강제로 넣는다. 모든 조건의 이미지는 최대 64장이다(뷰마다 최대 32 타임스탬프, `DUAL_SLOTS`).

| 조건 | 검색 | 입력 이미지 | 결과 폴더 |
|---|---|---|---|
| D `dual_gaze` | gaze Top-3 | 같은 3클립의 각 타임스탬프를 **전체 프레임, gaze crop** 순으로 | `results/dual_gaze/` |
| E `dual_retrieve` | full Top-3 + gaze Top-3 | full Top-3의 **전체 프레임**, 이어서 gaze Top-3의 **gaze crop** | `results/dual_retrieve/` |

- 이미 있는 A/B 결과는 건너뛰고 D/E만 새로 돌린다: `--stage qa`를 다시 실행하면 된다(A/B 문항당 호출은 없다). A/B가 `ok`가 아닌 문항은 D/E를 건너뛴다.
- E의 `hit_at_3`는 두 Top-3의 합집합(최대 6클립)이라 A OR B와 같다. 단일 Top-3 hit과 직접 비교하지 말 것. D의 hit은 gaze와 같다.
- 프롬프트는 A/B와 같아서 이미지에 라벨을 붙이지 않는다.
- `analyze.py`는 D/E가 모두 끝난 문항에 대해 `correct (dual-view conditions)` 표와 짝지은 비교를 `summary.md`에 추가하고, `per_question.csv`에 `qa_dual_gaze`, `qa_dual_retrieve` 열을 넣는다.
