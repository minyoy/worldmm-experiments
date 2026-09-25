#!/usr/bin/env python3
"""
에피소딕 메모리(HippoRAG 캐시)를 4개 스케일 전부 미리 만들어두는 스크립트.

[왜 필요한가]
에피소딕 메모리는 HippoRAG가 캡션을 LLM에 넣어 개체(NER)와 트리플을 뽑고, 그것으로
지식 그래프를 만드는 방식이다. 이 결과는 .cache/episodic_memory/ 에 저장된다.

그런데 배포된 output/metadata 에는 30초 캡션(6,223개)의 OpenIE 결과만 들어 있다.
(그건 seed_hipporag_cache.py 가 캐시 형식으로 바꿔서 심어준다.)
3min(1,073) / 10min(341) / 1h(64) 캡션 1,478개는 아무도 만들어둔 적이 없어서,
그냥 평가를 돌리면 --retriever-model 로 준 모델이 평가 도중에 만들게 된다.

그 모델이 로컬 Qwen이면 두 가지 문제가 생긴다 (실제로 겪었다).
  1) 논문 설정과 어긋난다. 논문의 WorldMM-8B는 에피소딕 메모리를 GPT-5-mini로 만든다.
  2) 실용적이지 않다. 1h 캡션(평균 9,588자)에서 같은 단어를 수백 번 반복 생성하다
     max_new_tokens 에 잘려 JSON이 깨지고, 그 캡션은 빈 개체로 저장된다(빈 비율 67%).
     속도도 캡션당 40~65초라 1,478개면 약 20시간이 걸린다.

[이 스크립트가 하는 일]
GPT-5-mini로 DAY7 끝까지 한 번에 인덱싱해서 캐시를 완성한다. 약 30분, 비용 $1~2.

[핵심 트릭]
HippoRAG는 캐시 파일 이름을 자기 기본 설정(llm_name=gpt-5-mini)으로 짓는다.
retriever 모델을 무엇으로 주든 파일 이름이 같다. 그래서 여기서 GPT-5-mini로 만들어두면
나중에 --retriever-model qwen3vl-8b 로 평가를 돌려도 이 캐시를 그대로 찾아 쓴다.
Qwen은 OpenIE를 한 번도 호출하지 않고 검색/추론/답변만 담당하게 된다.
논문의 "에피소딕 메모리는 GPT-5-mini, 추론은 Qwen" 설정이 이렇게 성립한다.

[실행] 저장소 루트에서, OPENAI_API_KEY 필요
    python experiments/visual_bottleneck/build_episodic_cache.py
한 번만 돌리면 되고, 이후 모든 조건(A~E, E′)이 같은 캐시를 공유한다.

[중간에 끊기면]
스케일 하나를 끝낼 때마다 캐시 파일이 저장된다. 그래서 다시 실행하면 이미 끝난 스케일은
건너뛰고 남은 것부터 한다. 다만 한 스케일 처리 도중에 죽으면 그 스케일은 처음부터 다시 한다
(HippoRAG가 배치를 다 끝낸 뒤에 저장하기 때문). 가장 큰 3min이 1,073개라 몇 분 규모다.
끊김을 피하려면 nohup 으로 띄우는 것이 좋다.
"""

import argparse
import gc
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE_ROOT, DATA_DIR, METADATA_DIR, SUBJECT, load_json  # noqa: E402

# WorldMM이 쓰는 4개 시간 스케일. 캡션 파일도 이 이름으로 되어 있다.
GRANULARITIES = ["30sec", "3min", "10min", "1h"]

# 타임스탬프는 int(날짜끝자리 + HHMMSSFF) 형식이다. 예: DAY1 11:09:43 -> 111094300
# 7일차 23:59:59.99 를 주면 모든 캡션이 인덱싱 대상이 된다.
END_OF_DAY7 = int("7" + "23595999")


def cache_stats(cache_root: str) -> None:
    """스케일별로 캐시에 문서가 몇 개 있고, 그중 개체가 비어 있는 게 몇 개인지 출력한다.

    개체가 빈 문서는 LLM이 NER에 실패한 것이라 그래프에 아무 노드도 기여하지 못한다.
    30초 스케일(GPT-5-mini로 만든 것)의 빈 비율이 7% 정도이므로, 다른 스케일도
    그 근처면 정상이고 크게 높으면 생성이 실패하고 있다는 신호다.
    """
    for g in GRANULARITIES:
        # 파일 이름의 gpt-5-mini 는 HippoRAG 기본 설정에서 나온 것이고,
        # 실제로 어떤 모델이 만들었는지와는 무관하다 (위 "핵심 트릭" 참조).
        p = os.path.join(cache_root, g, "openie_results_ner_gpt-5-mini.json")
        if not os.path.exists(p):
            print(f"  {g:6s} (없음)")
            continue
        docs = load_json(p)["docs"]
        empty = sum(1 for d in docs if not d["extracted_entities"])
        print(f"  {g:6s} {len(docs):5d} docs, entity 없음 {empty} ({empty / len(docs) * 100:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=SUBJECT, help="대상 인물 (EgoLifeQA가 있는 건 A1_JAKE뿐)")
    ap.add_argument("--model", default="gpt-5-mini", help="OpenIE에 쓸 LLM. 논문 설정은 gpt-5-mini")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--cache-root", default=CACHE_ROOT, help="기본값 .cache/episodic_memory")
    ap.add_argument("--until", type=int, default=END_OF_DAY7,
                    help="이 타임스탬프까지 인덱싱. 기본값은 DAY7 끝이라 전체를 만든다")
    ap.add_argument("--embedding-batch-size", type=int, default=8,
                    help="임베딩 배치 크기. HippoRAG 기본값은 128인데, 1h 캡션은 평균 9,600자(약 2,400토큰)라 "
                         "그대로 두면 64개가 한 배치로 묶여 GPU 메모리가 터진다. 기본값을 작게 잡았다")
    args = ap.parse_args()

    # 모델을 로드하기 전에 키부터 확인한다. 안 그러면 몇 분 기다린 뒤에 실패한다.
    if args.model.startswith("gpt") and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")

    print("시작 전 캐시 상태:")
    cache_stats(args.cache_root)

    # 캡션 4개 파일 경로. 이 캡션들은 caption.zip 에서 나온 저자 생성물이고,
    # output/metadata 의 메모리들이 전부 이 텍스트를 기준으로 만들어졌으므로 그대로 써야 한다.
    caption_dir = os.path.join(args.data_dir, "EgoLifeCap", args.subject)
    caption_files = {g: os.path.join(caption_dir, f"{args.subject}_{g}.json") for g in GRANULARITIES}
    semantic_path = os.path.join(METADATA_DIR, "semantic_memory", args.subject,
                                 "semantic_consolidation_results_gpt-5-mini.json")

    # 평가 때와 똑같은 방식으로 WorldMemory를 만든다.
    # respond 모델은 넘기지 않는다. 캐시 구축에는 답변 생성이 필요 없어서
    # WorldMemory가 retriever를 그대로 재사용하게 둔다.
    embedding_model = EmbeddingModel()
    llm = LLMModel(model_name=args.model)
    world_memory = WorldMemory(
        embedding_model=embedding_model,
        retriever_llm_model=llm,
        prompt_template_manager=PromptTemplateManager(),
        episodic_cache_root=args.cache_root,
    )
    world_memory.load_episodic_captions(caption_files=caption_files)
    # 시맨틱 메모리는 에피소딕 캐시를 만드는 데 필요 없지만,
    # index() 호출 경로를 평가 때와 똑같이 맞추려고 같이 로드한다.
    world_memory.load_semantic_triples(file_path=semantic_path)

    # 여기가 이 스크립트의 핵심이다.
    # 평가 코드는 질문마다 index(query_time)을 불러 그 시각까지만 조금씩 인덱싱하지만,
    # 여기서는 DAY7 끝을 줘서 캡션 전체를 대상으로 한다.
    #
    # 스케일을 한 번에 하나씩 처리한다. WorldMemory.index() 를 그냥 부르면 4개를 내부에서
    # 연달아 돌리는데, 그러면 어디까지 됐는지 로그로 알기 어렵고 중간에 끊겼을 때
    # 무엇이 저장됐는지 확인하기도 번거롭다. 한 스케일이 끝나면 HippoRAG 가 캐시를 저장하므로
    # 아래 루프가 곧 체크포인트가 된다.
    episodic = world_memory.episodic_memory
    all_granularities = list(episodic.granularities)
    for i, g in enumerate(all_granularities, 1):
        logger.info(f"[{i}/{len(all_granularities)}] {g}: indexing up to {args.until} with {args.model} ...")
        episodic.granularities = [g]     # 이번 반복에서 처리할 스케일만 남긴다
        episodic.indexed_time = 0        # index() 의 "이미 처리함" 조기 반환을 막는다

        # 임베딩 배치 크기를 낮춘다. HippoRAG 는 인스턴스를 만들 때 자기 기본 설정(128)을 쓰는데,
        # 긴 캡션에서는 그 크기로 GPU 메모리가 부족해진다. 인스턴스를 먼저 만들고 값을 바꿔 끼운다.
        episodic._get_or_create_hipporag(g).global_config.embedding_batch_size = args.embedding_batch_size

        episodic.index(args.until)

        # 다음 스케일로 넘어가기 전에 이 스케일이 쥐고 있던 GPU 메모리를 돌려준다.
        # 그래프와 임베딩이 스케일마다 쌓이면 마지막 1h 에서 메모리가 모자란다.
        episodic.hipporag.pop(g, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  [{i}/{len(all_granularities)}] {g} 완료. 현재 캐시:")
        cache_stats(args.cache_root)
    episodic.granularities = all_granularities

    print("\n완료 후 캐시 상태:")
    cache_stats(args.cache_root)
    print("\n이후 평가는 --retriever-model qwen3vl-8b 로 돌려도 이 캐시를 재사용한다.")


if __name__ == "__main__":
    main()
