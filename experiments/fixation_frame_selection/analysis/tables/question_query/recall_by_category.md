# 카테고리별 recall@3 (fixation frame selection)

`type` 5개와 `need_audio`별 recall@3(%)이다. 채점된 문항만 센다(정답 클립이 풀에 없으면 제외). 차이는 %p이고 `+a/−b`는 짝지은 문항 중 앞 arm만 맞힌 수 / 뒤 arm만 맞힌 수다. 카테고리당 문항이 60~125개라 한 문항이 0.8~1.6 %p이므로 작은 차이는 노이즈다.

## tol 0s

### recall@3 (%)

| 카테고리 | n | full | fixsel_ctl | fixsel | center | gazef | fixsel_gazef |
|---|---|---|---|---|---|---|---|
| type=EntityLog | 122 | 4.1 | 1.6 | 5.7 | 4.9 | 4.1 | 8.2 |
| type=EventRecall | 122 | 3.3 | 3.3 | 3.3 | 4.9 | 4.1 | 5.7 |
| type=HabitInsight | 61 | 3.3 | 1.6 | 3.3 | 1.6 | 3.3 | 4.9 |
| type=RelationMap | 121 | 3.3 | 5.0 | 6.6 | 7.4 | 8.3 | 6.6 |
| type=TaskMaster | 61 | 0.0 | 0.0 | 0.0 | 1.6 | 4.9 | 4.9 |
| need_audio=False | 287 | 3.8 | 4.2 | 5.6 | 6.3 | 5.2 | 7.7 |
| need_audio=True | 200 | 2.0 | 0.5 | 2.5 | 2.5 | 5.0 | 4.5 |
| ALL | 487 | 3.1 | 2.7 | 4.3 | 4.7 | 5.1 | 6.4 |

### 짝지은 차이 (%p, +앞 arm만/−뒤 arm만)

| 카테고리 | fixsel − full | fixsel − fixsel_ctl | fixsel_gazef − gazef | fixsel_gazef − full |
|---|---|---|---|---|
| type=EntityLog | +1.6 (+5/−3) | +4.1 (+7/−2) | +4.1 (+5/−0) | +4.1 (+8/−3) |
| type=EventRecall | +0.0 (+3/−3) | +0.0 (+3/−3) | +1.6 (+3/−1) | +2.5 (+5/−2) |
| type=HabitInsight | +0.0 (+1/−1) | +1.6 (+2/−1) | +1.6 (+1/−0) | +1.6 (+2/−1) |
| type=RelationMap | +3.3 (+6/−2) | +1.7 (+5/−3) | -1.7 (+4/−6) | +3.3 (+5/−1) |
| type=TaskMaster | +0.0 (+0/−0) | +0.0 (+0/−0) | +0.0 (+2/−2) | +4.9 (+3/−0) |
| need_audio=False | +1.7 (+12/−7) | +1.4 (+13/−9) | +2.4 (+12/−5) | +3.8 (+17/−6) |
| need_audio=True | +0.5 (+3/−2) | +2.0 (+4/−0) | -0.5 (+3/−4) | +2.5 (+6/−1) |
| ALL | +1.2 (+15/−9) | +1.6 (+17/−9) | +1.2 (+15/−9) | +3.3 (+23/−7) |

### 정답 순위 중앙값 (낮을수록 좋음)

| 카테고리 | full | fixsel_ctl | fixsel | center | gazef | fixsel_gazef |
|---|---|---|---|---|---|---|
| type=EntityLog | 286 | 278 | 276 | 210 | 270 | 270 |
| type=EventRecall | 300 | 298 | 284 | 233 | 248 | 234 |
| type=HabitInsight | 260 | 184 | 275 | 336 | 265 | 358 |
| type=RelationMap | 274 | 215 | 228 | 184 | 152 | 214 |
| type=TaskMaster | 428 | 440 | 424 | 475 | 519 | 463 |
| need_audio=False | 241 | 206 | 208 | 179 | 181 | 173 |
| need_audio=True | 417 | 418 | 444 | 362 | 370 | 398 |
| ALL | 289 | 268 | 292 | 245 | 249 | 268 |

## tol 30s

### recall@3 (%)

| 카테고리 | n | full | fixsel_ctl | fixsel | center | gazef | fixsel_gazef |
|---|---|---|---|---|---|---|---|
| type=EntityLog | 125 | 7.2 | 4.8 | 10.4 | 8.8 | 8.8 | 13.6 |
| type=EventRecall | 126 | 4.8 | 5.6 | 7.1 | 7.1 | 6.3 | 7.9 |
| type=HabitInsight | 61 | 8.2 | 9.8 | 11.5 | 1.6 | 4.9 | 9.8 |
| type=RelationMap | 125 | 9.6 | 12.0 | 14.4 | 16.0 | 16.8 | 13.6 |
| type=TaskMaster | 63 | 1.6 | 3.2 | 1.6 | 3.2 | 9.5 | 7.9 |
| need_audio=False | 298 | 8.7 | 10.4 | 13.4 | 12.1 | 11.4 | 13.4 |
| need_audio=True | 202 | 3.5 | 2.5 | 4.0 | 3.5 | 7.4 | 7.4 |
| ALL | 500 | 6.6 | 7.2 | 9.6 | 8.6 | 9.8 | 11.0 |

### 짝지은 차이 (%p, +앞 arm만/−뒤 arm만)

| 카테고리 | fixsel − full | fixsel − fixsel_ctl | fixsel_gazef − gazef | fixsel_gazef − full |
|---|---|---|---|---|
| type=EntityLog | +3.2 (+7/−3) | +5.6 (+10/−3) | +4.8 (+7/−1) | +6.4 (+11/−3) |
| type=EventRecall | +2.4 (+4/−1) | +1.6 (+5/−3) | +1.6 (+5/−3) | +3.2 (+6/−2) |
| type=HabitInsight | +3.3 (+2/−0) | +1.6 (+2/−1) | +4.9 (+3/−0) | +1.6 (+3/−2) |
| type=RelationMap | +4.8 (+8/−2) | +2.4 (+5/−2) | -3.2 (+4/−8) | +4.0 (+13/−8) |
| type=TaskMaster | +0.0 (+1/−1) | -1.6 (+0/−1) | -1.6 (+1/−2) | +6.3 (+5/−1) |
| need_audio=False | +4.7 (+17/−3) | +3.0 (+16/−7) | +2.0 (+14/−8) | +4.7 (+27/−13) |
| need_audio=True | +0.5 (+5/−4) | +1.5 (+6/−3) | +0.0 (+6/−6) | +4.0 (+11/−3) |
| ALL | +3.0 (+22/−7) | +2.4 (+22/−10) | +1.2 (+20/−14) | +4.4 (+38/−16) |

### 정답 순위 중앙값 (낮을수록 좋음)

| 카테고리 | full | fixsel_ctl | fixsel | center | gazef | fixsel_gazef |
|---|---|---|---|---|---|---|
| type=EntityLog | 133 | 152 | 152 | 101 | 115 | 104 |
| type=EventRecall | 158 | 123 | 140 | 96 | 103 | 128 |
| type=HabitInsight | 189 | 159 | 169 | 182 | 213 | 171 |
| type=RelationMap | 95 | 108 | 76 | 57 | 56 | 59 |
| type=TaskMaster | 255 | 266 | 229 | 242 | 213 | 318 |
| need_audio=False | 101 | 108 | 104 | 70 | 78 | 74 |
| need_audio=True | 218 | 210 | 190 | 190 | 193 | 216 |
| ALL | 166 | 138 | 146 | 108 | 114 | 117 |

## tol 60s

### recall@3 (%)

| 카테고리 | n | full | fixsel_ctl | fixsel | center | gazef | fixsel_gazef |
|---|---|---|---|---|---|---|---|
| type=EntityLog | 125 | 9.6 | 5.6 | 12.8 | 12.0 | 12.8 | 16.8 |
| type=EventRecall | 126 | 5.6 | 6.3 | 10.3 | 10.3 | 10.3 | 11.1 |
| type=HabitInsight | 61 | 11.5 | 9.8 | 11.5 | 1.6 | 6.6 | 11.5 |
| type=RelationMap | 125 | 14.4 | 12.0 | 17.6 | 17.6 | 21.6 | 17.6 |
| type=TaskMaster | 63 | 3.2 | 6.3 | 4.8 | 4.8 | 11.1 | 9.5 |
| need_audio=False | 298 | 12.4 | 11.4 | 16.4 | 14.8 | 15.1 | 17.1 |
| need_audio=True | 202 | 4.5 | 3.0 | 5.9 | 5.0 | 10.9 | 9.4 |
| ALL | 500 | 9.2 | 8.0 | 12.2 | 10.8 | 13.4 | 14.0 |

### 짝지은 차이 (%p, +앞 arm만/−뒤 arm만)

| 카테고리 | fixsel − full | fixsel − fixsel_ctl | fixsel_gazef − gazef | fixsel_gazef − full |
|---|---|---|---|---|
| type=EntityLog | +3.2 (+8/−4) | +7.2 (+12/−3) | +4.0 (+6/−1) | +7.2 (+13/−4) |
| type=EventRecall | +4.8 (+6/−0) | +4.0 (+7/−2) | +0.8 (+7/−6) | +5.6 (+9/−2) |
| type=HabitInsight | +0.0 (+1/−1) | +1.6 (+2/−1) | +4.9 (+3/−0) | +0.0 (+3/−3) |
| type=RelationMap | +3.2 (+7/−3) | +5.6 (+8/−1) | -4.0 (+6/−11) | +3.2 (+15/−11) |
| type=TaskMaster | +1.6 (+2/−1) | -1.6 (+1/−2) | -1.6 (+2/−3) | +6.3 (+5/−1) |
| need_audio=False | +4.0 (+18/−6) | +5.0 (+22/−7) | +2.0 (+18/−12) | +4.7 (+32/−18) |
| need_audio=True | +1.5 (+6/−3) | +3.0 (+8/−2) | -1.5 (+6/−9) | +5.0 (+13/−3) |
| ALL | +3.0 (+24/−9) | +4.2 (+30/−9) | +0.6 (+24/−21) | +4.8 (+45/−21) |

### 정답 순위 중앙값 (낮을수록 좋음)

| 카테고리 | full | fixsel_ctl | fixsel | center | gazef | fixsel_gazef |
|---|---|---|---|---|---|---|
| type=EntityLog | 83 | 105 | 90 | 60 | 64 | 66 |
| type=EventRecall | 82 | 104 | 92 | 84 | 75 | 87 |
| type=HabitInsight | 139 | 123 | 115 | 98 | 109 | 98 |
| type=RelationMap | 65 | 81 | 57 | 50 | 26 | 35 |
| type=TaskMaster | 225 | 182 | 179 | 214 | 154 | 125 |
| need_audio=False | 70 | 80 | 64 | 52 | 44 | 46 |
| need_audio=True | 166 | 138 | 142 | 130 | 145 | 134 |
| ALL | 99 | 105 | 91 | 81 | 76 | 84 |
