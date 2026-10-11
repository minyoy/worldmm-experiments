# gaze_cases 관찰 노트

`index.html`(A/B/C 탭)에서 문항을 직접 보며 정리한 관찰입니다. 탭 구성은 [`../build_gaze_cases.py`](../build_gaze_cases.py) 상단 설명을 보세요.
순위는 tol 60s 채점의 정답 클립 순위(작을수록 좋고 3 이하가 hit@3)이고, 숫자는 `analysis/recall_arms_tol60.md`와 같은 결과 파일에서 가져왔습니다.

## 1. gaze 를 넣었을 때만 맞춘 문항 (A 탭): 무엇이 달랐나

- **효과는 주로 오디오가 필요 없는 `RelationMap`, `EntityLog` 문항에서 나왔다.**
- "누가 이 행동을 했는가"를 묻는 문항은 crop 으로 **인물이 확대되어 보여서** 도움이 됐다.
- "마지막으로 수행한 작업", "이전 위치"를 묻는 문항은 시선이 **물체를 바라보는 장면**이 많아서 실질적으로 도움이 됐다.
- 오디오가 필요한 문항이나 `EventRecall`처럼 **질문 시각과 검색된 구간의 타임스탬프를 비교해야 하는 문항**에서는 도움이 된 부분을 찾지 못했다.

## 2. 시선이 중앙에 없는 데이터 (B 탭): 도움이 될 것 같은가

시선이 중앙 crop 밖으로 벗어나는 경우는 크게 세 가지다. 이런 구간에서는 중앙 crop 이 놓치는 것을 gaze crop 이 잡을 수 있으므로 **gaze 가 가장 달라지는 영역**이다.

1. **대화 상대가 화면 중앙에 없을 때.** 대화 중에는 상대를 쳐다보느라 시선이 중심에서 벗어난다. 전반적으로 대화하는 사람을 많이 쳐다보는 경향이 있다.
2. **HOI(손-물체 상호작용) 중 다음에 집을 물건을 볼 때.** 손이 닿기 전에 시선이 먼저 물건으로 가서 중심에서 벗어난다. (이전에 말했던 내용)
3. **머리를 돌릴 때.** 머리가 돌아가는 방향으로 시선이 먼저 향한다.

## 3. gaze crop 이 오히려 놓친 문항 (C 탭): 사용자가 집중하지 않은 것을 묻는 경우

C 탭은 `full`은 top-3 안에 넣었는데 `gazef`는 못 넣은 문항 17개(순위가 가장 많이 떨어진 순)다.

**공통 패턴.** 착용자는 자기 할 일에 집중해서 주변을 파악하지 못한다. 그런데 EgoLifeQA 에는 **사용자가 집중하지 않은 것**을 묻는 문항이 있다. gaze crop 은 시선 주변만 남기므로 이런 문항에서는 정답 근거가 crop 밖으로 사라진다.

| 문항 | 질문 | 순위 (full → gazef, center) | 놓친 이유 |
|---|---|---|---|
| Q199 (RelationMap) | Who was I with the last time I went to the supermarket to look at fruit? | 1 → 28 (center 11) | 딸기를 사는 데 집중해서 **주변에 누구랑 같이 있었는지** 담기지 않는다. |
| Q128 (RelationMap) | Who came in to find me while we were busy removing the computer's hard drive just now | 1 → 9 (center 1) | 누가 찾아오는지보다 **핸드폰을 보는 데 집중**해서 찾아온 사람이 crop 에 없다. |
| Q218 (EntityLog) | Who bought the vegetables on the table, and when were they bought? | 2 → 12 (center 8) | **채소 재고에 더 집중**해서 누가 샀는지를 파악하기 불리하다. |
| Q346 (RelationMap) | When we go out shopping and take a taxi, who usually sits next to me? | 2 → 7 (center 4) | **핸드폰을 많이 봐서** crop 에서 사람 얼굴이 잘린다. |
| Q1 (EntityLog) | Who used the screwdriver first? | 2 → 4 (center 6) | 앨리스가 드라이버를 쓰는 것에 **집중하지 않았다**. |

- 다섯 문항 모두 `RelationMap`/`EntityLog` 로, 1절에서 gaze 가 도움이 됐던 유형과 같다. **같은 유형이 시선이 대상에 있느냐 아니냐에 따라 이득이 되기도 손해가 되기도 한다.**
- Q128 은 `center` 가 맞혔는데 `gazef` 는 놓쳤다. 이 경우만 **gaze 위치 자체가 손해**이고, 나머지 네 문항은 `center` 도 놓쳐서 crop 자체의 영향도 섞여 있다.
- Q218, Q346 은 `gazef_05_randf`(프레임을 무작위 시선에 맞춘 대조군)가 맞혔다(순위 2, 1). 시선 위치가 어긋난 것이 손해였다는 해석과 맞다.
- 17개 중 12개는 `center` 도 놓쳤으므로, C 탭 전체를 "gaze 의 잘못"으로 읽으면 안 되고 crop 으로 주변 정보가 사라진 효과와 구분해서 봐야 한다.
