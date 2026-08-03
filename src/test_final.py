import argparse
import logging
import os
import sys
from evaluator import SurGEvaluator

# 레포 루트의 .env 에서 API 키/엔드포인트/판정 모델을 읽어온다.
# python-dotenv 가 없어도 동작하도록 실패를 무시한다(직접 export 해도 됨).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='Survey Generation Evaluator')
    parser.add_argument(
        '--passage_dir',
        type=str,
        help='Directory containing generated survey passages'
    )
    parser.add_argument(
        '--eval_list',
        nargs='+',
        default=["ALL"],
        help='Evaluation metrics to compute (space-separated)'
    )
    parser.add_argument(
        '--survey_path',
        type=str,
        default=os.path.join("data", "surveys.json"),
        help='Path to surveys.json file'
    )
    parser.add_argument(
        '--corpus_path', 
        type=str,
        default=os.path.join("data", "corpus.json"),
        help='Path to corpus.json file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default="0",
        help='Device ID for computation'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='API key for evaluation services. Falls back to OPENROUTER_API_KEY, then OPENAI_API_KEY (see .env)'
    )
    parser.add_argument(
        '--base_url',
        type=str,
        default=None,
        help='OpenAI-compatible endpoint, e.g. https://openrouter.ai/api/v1. '
             'Falls back to OPENROUTER_BASE_URL, then OPENAI_BASE_URL. Omit to use OpenAI directly'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='LLM-as-Judge model for Structure_Quality (SQS) and Logic (CQS). '
             'Falls back to SURGE_JUDGE_MODEL, then gpt-4o'
    )
    parser.add_argument(
        '--max_retries',
        type=int,
        default=None,
        help='How many times to re-ask the judge when it does not answer with a bare 0-5. '
             'Falls back to SURGE_MAX_RETRIES, then 5. Guards against models that never '
             'match the expected format'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default=None,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging verbosity. Falls back to SURGE_LOG_LEVEL, then INFO'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Also write logs to this file. Falls back to SURGE_LOG_FILE. '
             'Logs always go to stderr'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Path to save evaluation results'
    )
    return parser.parse_args()

def resolve_llm_settings(args):
    """CLI 인자 > .env/환경변수 > 기본값 순으로 판정 LLM 설정을 확정한다."""
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENROUTER_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    model = args.model or os.environ.get("SURGE_JUDGE_MODEL") or "gpt-4o"
    max_retries = args.max_retries or os.environ.get("SURGE_MAX_RETRIES") or 5
    return api_key, base_url, model, int(max_retries)


def setup_logging(args):
    """stderr(+선택적으로 파일)로 타임스탬프가 붙은 로그를 내보낸다."""
    level = (args.log_level or os.environ.get("SURGE_LOG_LEVEL") or "INFO").upper()
    log_file = args.log_file or os.environ.get("SURGE_LOG_FILE")

    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # HTTP 요청 로그는 DEBUG 에서도 너무 시끄러워 억제한다.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # --log_level DEBUG 는 이 저장소 코드에만 적용되도록 서드파티는 INFO 로 묶는다.
    for lib in ("sentence_transformers", "transformers", "FlagEmbedding", "datasets"):
        logging.getLogger(lib).setLevel(logging.INFO)
    return log_file


if __name__ == '__main__':
    args = parse_args()

    log_file = setup_logging(args)
    logger = logging.getLogger("surge.eval")

    api_key, base_url, judge_model, max_retries = resolve_llm_settings(args)

    # SQS/CQS 를 돌리지 않는다면 키가 없어도 되므로, 이때만 더미 키로 클라이언트를 만든다.
    needs_llm = any(m in args.eval_list for m in ("ALL", "Structure_Quality", "Logic"))
    if needs_llm and not api_key:
        raise SystemExit(
            "Structure_Quality(SQS)/Logic(CQS) require an API key. "
            "Pass --api_key, or set OPENROUTER_API_KEY / OPENAI_API_KEY in .env"
        )

    logger.info("eval_list=%s passage_dir=%s device=%s", args.eval_list, args.passage_dir, args.device)
    if needs_llm:
        logger.info(
            "judge: model=%s base_url=%s max_retries=%d",
            judge_model, base_url or "https://api.openai.com/v1 (default)", max_retries,
        )
    else:
        logger.info("judge: 사용 안 함 (SQS/CQS 가 eval_list 에 없음)")
    if log_file:
        logger.info("로그 파일: %s", log_file)

    evaluator = SurGEvaluator(
        device=args.device,
        survey_path=args.survey_path,
        corpus_path=args.corpus_path,
        api_key=api_key or "unused-no-llm-metrics",
        base_url=base_url,
        judge_model_name=judge_model,
        max_retries=max_retries
    )

    result = evaluator.eval_all(
        passage_dir=args.passage_dir,
        eval_list=args.eval_list,
        save_path=args.save_path
    )

    print("\nEvaluation Results:")
    print(result)