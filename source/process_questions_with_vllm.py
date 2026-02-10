from argparse import ArgumentParser
import codecs
import copy
import csv
import gc
import json
import logging
import os
import random
import signal
import sys
from typing import Dict, List, Optional, Tuple, Union

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from jiwer import cer
from tqdm import tqdm
from annoy import AnnoyIndex
from rank_bm25 import BM25Okapi
from nltk import wordpunct_tokenize
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, T5EncoderModel
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel
from transformers import GenerationConfig, AutoConfig
from FlagEmbedding import FlagLLMReranker, FlagReranker
from vllm import LLM, SamplingParams
from vllm import EngineArgs, LLMEngine


QA_SYSTEM_PROMPT = ('Вы - эксперт в области науки, техники и юриспруденции. Вы безукоризненно вежливы, а ваши ответы '
                    'на сложные вопросы человека всегда логичны и точны. Вы способны разобраться даже в самой сложной '
                    'документации, чтобы помочь человеку извлечь из этой документации полезные сведения для ответа на '
                    'вопрос. Вы точно, понятно и детально отвечаете пользователю на русском языке.')
QA_USER_PROMPT = ('Внимательно проанализируйте найденные фрагменты следующих документов, представленные в формате JSON:'
                  '\n\n```json\n{jsonified_context}\n```\n\nОпираясь на сведения из этих документов, ответьте на '
                  'вопрос, заданный человеком:\n\n```text\n{user_question}\n```\n\nПрежде чем давать ответ, хорошенько '
                  'подумайте шаг за шагом, проанализируйте текстовые фрагменты документов на предмет скрытых логических '
                  'взаимосвязей, которые помогут вам наиболее точно ответить на вопрос человека. Отмечу, что не все из '
                  'найденных текстовых фрагментов одинаково полезны. Иногда некоторые из них могут быть нерелевантны '
                  'заданному вопросу, что обусловлено несовершенством алгоритма поиска. Поэтому для ответа на вопрос '
                  'используйте только такие фрагменты документов, которые содержат наиболее полезную, релевантную '
                  'заданному вопросу информацию. Если же вопрос человека вобще не связан с одним из найденных '
                  'текстовых фрагментов, то вежливо откажитесь от ответа (например, скажите, что информации для '
                  'точного ответа недостаточно). Главное - не обманывайте человека и не придумывайте то, чего '
                  'не знаете!\n')
INSUFFICIENT_INFORMATION_ANSWER: str = ('К сожалению, в базе данных недостаточно информации для '
                                        'точного ответа на ваш вопрос. Попробуйте его переформулировать.')
SEARCH_SYSTEM_PROMPT = ('Вы - агент в RAG-пайплайне, помогающий искать фрагменты (чанки) документов в базе знаний в '
                        'области науки, техники и юриспруденции. Вы хорошо умеете формулировать поисковые запросы, по '
                        'которым механизм поиска (retrieve) выдаст такие чанки, которые будут максимально полезны для '
                        'ответа на вопрос пользователя.')
SEARCH_USER_PROMPT = ('Поступил следующий вопрос от пользователя:\n\n```text\n{user_question}\n```\n\nНе отвечайте на '
                      'этот вопрос, а придумайте три эффективных поисковых запроса, которые бы позволили найти '
                      'наиболее релевантную информацию из документов в базе данных, необходимую для максимально '
                      'точного ответа на этот вопрос. Если же вопрос является неинформативным (например, "ты кто", '
                      '"как твои дела" или вообще бессмысленной фразой типа "балерина капучино"), то откажитесь от '
                      'генерации поисковых запросов и вместо них напишите фразу "не могу сгенерировать ни одного '
                      'поискового запроса".')
MEANINGLESS_REQUEST_ANSWER = 'Не могу сгенерировать ни одного поискового запроса.'
EXAMPLES_OF_SEARCH_QUERIES = (
    {
        'question': 'Где живут и чем питаются пингвины?',
        'search_queries': (
            'пингвины место обитания', # 0
            'пингвины рацион питания', # 1
            'где живут пингвины и чем питаются' # 2
        )
    },
    {
        'question': 'Какой ГОСТ или СНиП регламентирует разработку УПБС по укрупненным статьям затрат?',
        'search_queries': (
            'ГОСТ УПБС укрупненные статьи затрат', # 0
            'СНиП разработка УПБС', # 1
            'нормативные документы УПБС стоимостные статьи' # 2
        )
    },
    {
        'question': 'Какова схема прибора для лабораторного закрепления просадочных лессовых грунтов по методике '
                    'Ростовского ПромстройНИИпроекта?',
        'search_queries': (
            'лабораторное закрепление просадочных лессовых грунтов методика Ростовского ПромстройНИИпроекта '
            'схема прибора', # 0
            'Ростовский ПромстройНИИпроекта лабораторный прибор для исследования просадочных грунтов', # 1
            'методы и устройства для закрепления лессовых грунтов по методике Ростовского ПромстройНИИпроекта' # 2
        )
    },
    {
        'question': 'Как определяется «граница балансовой принадлежности» в документе МДС 41-3.2000?',
        'search_queries': (
            'граница балансовой принадлежности МДС 41-3.2000', # 0
            'определение границы балансовой принадлежности в МДС 41-3.2000', # 1
            'МДС 41-3.2000 балансовая принадлежность' # 2
        )
    },
    {
        'question': 'Как учитывается полезное ископаемое, в отношении которого в налоговом периоде завершен комплекс '
                    'технологических операций?',
        'search_queries': (
            'учет полезного ископаемого завершен комплекс технологических операций налоговый период порядок учета', # 0
            'комплекс технологических операций завершен полезное ископаемое как учитывается НДПИ налоговый период', # 1
            'учет добытого полезного ископаемого завершение технологических операций определение момента добычи' # 2
        )
    },
    {
        'question': 'Пошли выпьем кофе?',
        'search_queries': (MEANINGLESS_REQUEST_ANSWER,)  # -1
    }
)
MAX_NUMBER_OF_GENERATED_TOKENS: int = 4096
VECTOR_TOP_K: int = 100
BM25_TOP_K: int = 100
MAX_TOKENS_PER_CHUNK: int = 512
FINAL_NUM_BEST_CHUNKS: int = 7
MAX_NUMBER_OF_SEARCH_QUERIES: int = 4
RANDOM_SEED: int = 42
LEXICAL_DISTANCE_THRESHOLD: float = 0.05
THINKING_END_TOKEN = '</think>'
qa_logger = logging.getLogger(__name__)


def finalize_vllm():
    if torch.distributed.is_initialized():
        if hasattr(LLMEngine, 'shutdown'):
            LLMEngine.shutdown()
        torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()


def handle_exit(signal, frame):
    finalize_vllm()
    sys.exit(0)


def pages_to_string(page_numbers: List[int]) -> str:
    if len(page_numbers) < 1:
        return ''
    if len(page_numbers) == 1:
        return f'p. {page_numbers[0]}'
    sorted_pages_numbers = sorted(page_numbers)
    page_ranges = []
    page_range_start = sorted_pages_numbers[0]
    prev_page_number = page_range_start
    for page_number in sorted_pages_numbers[1:]:
        if page_number > (prev_page_number + 1):
            page_ranges.append((page_range_start, prev_page_number))
            page_range_start = page_number
        prev_page_number = page_number
    page_ranges.append((page_range_start, prev_page_number))
    description_of_pages = 'pp. '
    description_of_pages += ', '.join([(f'{it[0]}-{it[1]}' if (it[1] > it[0]) else f'{it[0]}') for it in page_ranges])
    return description_of_pages


def prepare_pretty_description_of_references(context: Dict[str, Dict[str, Union[List[str], List[int]]]]) -> str:
    doc_names = sorted(list(context.keys()))
    if len(doc_names) == 0:
        return ''
    if len(doc_names) == 1:
        return f"{doc_names[0]} ({pages_to_string(context[doc_names[0]]['pages'])})"
    return ', '.join([f"{cur_doc} ({pages_to_string(context[cur_doc]['pages'])})" for cur_doc in doc_names])


def similar_to_natural_language(document_name: str) -> bool:
    words = set(filter(lambda it: it.isalpha(), wordpunct_tokenize(document_name)))
    return len(words) > 1


def prepare_messages_for_answering(user_question: str, context: Dict[str, Dict[str, Union[List[str], List[int]]]],
                                   llm_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                                   add_json: Optional[bool] = False) -> Union[str, Tuple[str, Dict[str, str]]]:
    structured_context = []
    documents = sorted(list(context.keys()))
    for doc_name in documents:
        structured_context.append({
            'документ': doc_name if similar_to_natural_language(doc_name) else 'без названия',
            'найденные текстовые фрагменты документа': context[doc_name]['chunks']
        })
    system_prompt = QA_SYSTEM_PROMPT
    user_prompt = QA_USER_PROMPT.format(
        user_question=' '.join(user_question.strip().split()).strip(),
        jsonified_context=json.dumps(obj=structured_context, ensure_ascii=False, indent=4)
    )
    messages = [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': user_prompt
        }
    ]
    if THINKING_END_TOKEN in llm_tokenizer.vocab:
        text = llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
    else:
        text = llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    del structured_context, documents, messages
    if add_json:
        return text, {'system': system_prompt, 'query': user_prompt}
    return text


def generate_answer(input_prompt: str, large_language_model: LLM,
                    llm_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                    sampling_params: SamplingParams) -> str:
    outputs = large_language_model.generate([input_prompt], sampling_params, use_tqdm=False)
    output_ids = outputs[0].outputs[0].token_ids
    del outputs
    think_end_pos = 0
    if THINKING_END_TOKEN in llm_tokenizer.vocab:
        try:
            think_end_pos = len(output_ids) - output_ids[::-1].index(llm_tokenizer.vocab[THINKING_END_TOKEN])
        except ValueError:
            pass
    response = llm_tokenizer.decode(output_ids[think_end_pos:], skip_special_tokens=True).strip()
    del output_ids
    return response.strip()


def prepare_text_for_bm25(src: str, stemmer: SnowballStemmer) -> List[str]:
    return list(map(lambda it2: stemmer.stem(it2).lower(), filter(lambda it1: it1.isalnum(), wordpunct_tokenize(src))))


def prepare_bm25(chunks: List[str], stemmer: SnowballStemmer) -> BM25Okapi:
    stemmed_texts = [prepare_text_for_bm25(it, stemmer) for it in tqdm(chunks, desc='Chunk stemming')]
    return BM25Okapi(stemmed_texts)


def find_relevant_chunks_with_bm25_search(user_question: str, stemmer: SnowballStemmer, bm25_db: BM25Okapi,
                                          num_chunks: int, top_k: Optional[int]=BM25_TOP_K) -> List[int]:
    chunk_scores = bm25_db.get_scores(prepare_text_for_bm25(user_question, stemmer))
    if len(chunk_scores) != num_chunks:
        err_msg = f'The chunk scores list does not correspond to the chunks list! {len(chunk_scores)} != {num_chunks}'
        raise ValueError(err_msg)
    chunk_indices = list(range(num_chunks))
    ordered_chunks = sorted(list(zip(chunk_indices, chunk_scores)), key=lambda it: (-it[1], it[0]))
    if len(ordered_chunks) > top_k:
        ordered_chunks = ordered_chunks[:top_k]
    del chunk_indices, chunk_scores
    relevant_chunk_indices = [it[0] for it in ordered_chunks]
    del ordered_chunks
    return relevant_chunk_indices


def vectorize_question(user_question: str,
                       sent_emb_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                       sent_embedder: PreTrainedModel) -> np.ndarray:

    def pool(hidden_state, mask, pooling_method='cls'):
        if pooling_method not in {'mean', 'cls'}:
            raise ValueError(f'The pooling method {pooling_method} is not supported!')
        if pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            emb = s / d
        else:
            emb = hidden_state[:, 0]
        return emb

    tokenized_inputs = sent_emb_tokenizer(
        ['search_query: ' + user_question],
        max_length=MAX_TOKENS_PER_CHUNK, padding=True, truncation=True, return_tensors='pt'
    ).to(sent_embedder.device)
    with torch.no_grad():
        outputs = sent_embedder(**tokenized_inputs)

    embeddings = pool(
        outputs.last_hidden_state,
        tokenized_inputs['attention_mask'],
        pooling_method='cls'
    )
    embeddings = F.normalize(embeddings, p=2, dim=1)

    embeddings_np = embeddings.float().cpu().numpy().flatten()
    del tokenized_inputs, outputs, embeddings
    return embeddings_np


def find_relevant_chunks_with_vector_search(user_question: str,
                                            sent_emb_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                                            sent_embedder: PreTrainedModel,
                                            vector_db: AnnoyIndex,
                                            top_k: Optional[int]=VECTOR_TOP_K) -> List[int]:
    question_vector = vectorize_question(user_question, sent_emb_tokenizer, sent_embedder)
    chunk_indices = vector_db.get_nns_by_vector(vector=question_vector, n=top_k, search_k=-1, include_distances=False)
    del question_vector
    return chunk_indices


def extend_list_of_selected_chunks(indices_of_selected_chunks: List[int],
                                   chunks: List[Dict[str, Union[int, List[int], str]]]) -> List[int]:
    if len(indices_of_selected_chunks) == 0:
        return []
    structured_chunks = dict()
    for chunk_index, chunk_content in enumerate(chunks):
        doc_name = chunk_content['document']
        if doc_name not in structured_chunks:
            structured_chunks[doc_name] = []
        structured_chunks[doc_name].append(chunk_index)
    set_of_extended_chunks = set()
    for chunk_index in indices_of_selected_chunks:
        doc_name = chunks[chunk_index]['document']
        set_of_extended_chunks.add(chunk_index)
        found_pos = structured_chunks[doc_name].index(chunk_index)
        if found_pos > 0:
            set_of_extended_chunks.add(structured_chunks[doc_name][found_pos - 1])
        if found_pos < (len(structured_chunks[doc_name]) - 1):
            set_of_extended_chunks.add(structured_chunks[doc_name][found_pos + 1])
    del structured_chunks
    return sorted(list(set_of_extended_chunks))


def rerank_chunks(user_question: str, selected_chunks: List[int],
                  all_chunks: List[Dict[str, Union[int, List[int], str]]],
                  reranker: Union[FlagLLMReranker, FlagReranker],
                  num_best: int) -> Dict[int, float]:
    selected_chunks_ = sorted(list(set(selected_chunks)))
    pairs = [(user_question, all_chunks[idx]['chunk_text']) for idx in selected_chunks_]
    scores = reranker.compute_score(pairs)
    del pairs
    scored_indices = sorted(
        list(filter(lambda x: x[1] > 0.0, zip(selected_chunks_, scores))),
        key=lambda it: (-it[1], it[0])
    )
    del selected_chunks_
    reranked_chunks = dict(scored_indices[:min(len(scored_indices), num_best)])
    return reranked_chunks


def chunk_indices_to_context(
        selected_chunks: Dict[int, float],
        all_chunks: List[Dict[str, Union[int, List[int], str]]],
        llm_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_context_len: Optional[int] = None
) -> Dict[str, Dict[str, Union[List[str], List[int]]]]:
    ordered_chunk_indices = sorted(list(selected_chunks.keys()), key=lambda it: -selected_chunks[it])
    if max_context_len is None:
        best_chunk_indices = ordered_chunk_indices
    else:
        best_chunk_indices = [ordered_chunk_indices[0]]
        preliminary_context = all_chunks[ordered_chunk_indices[0]]['chunk_text'].strip()
        for chunk_idx in ordered_chunk_indices[1:]:
            chunk_text = '\n\n' + all_chunks[chunk_idx]['chunk_text'].strip()
            num_context_tokens = len(llm_tokenizer.tokenize(preliminary_context + chunk_text, add_special_tokens=True))
            if num_context_tokens > max_context_len:
                break
            preliminary_context += chunk_text
            best_chunk_indices.append(chunk_idx)
            del chunk_text
    selected_documents = dict()
    for chunk_idx in best_chunk_indices:
        doc_name = all_chunks[chunk_idx]['document']
        if doc_name not in selected_documents:
            selected_documents[doc_name] = {
                'chunks': [],
                'pages': set()
            }
        selected_documents[doc_name]['chunks'].append(all_chunks[chunk_idx]['chunk_text'])
        selected_documents[doc_name]['pages'] |= set(range(
            all_chunks[chunk_idx]['page_range'][0],
            all_chunks[chunk_idx]['page_range'][1]
        ))
    for doc_name in selected_documents:
        selected_documents[doc_name]['pages'] = sorted(list(selected_documents[doc_name]['pages']))
    return selected_documents


def prepare_messages_for_search(user_question: str,
                                llm_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> str:
    messages = [
        {
            'role': 'system',
            'content': SEARCH_SYSTEM_PROMPT
        }
    ]
    for cur_example in EXAMPLES_OF_SEARCH_QUERIES:
        messages += [
            {
                'role': 'user',
                'content': SEARCH_USER_PROMPT.format(user_question=cur_example['question'])
            },
            {
                'role': 'assistant',
                'content': '\n'.join(cur_example['search_queries']).strip()
            }
        ]
    messages.append({
        'role': 'user',
        'content': SEARCH_USER_PROMPT.format(user_question=user_question)
    })
    if THINKING_END_TOKEN in llm_tokenizer.vocab:
        text = llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    else:
        text = llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    del messages
    return text


def calculate_distance_between_texts(reference: str, hypothesis: str) -> float:
    words_of_reference = ' '.join(list(filter(lambda it: it.isalnum(), wordpunct_tokenize(reference.lower()))))
    words_of_hypothesis = ' '.join(list(filter(lambda it: it.isalnum(), wordpunct_tokenize(hypothesis.lower()))))
    if words_of_reference == words_of_hypothesis:
        return 0.0
    return cer(reference=words_of_reference, hypothesis=words_of_hypothesis)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.random.manual_seed(RANDOM_SEED)
    if not torch.cuda.is_available():
        err_msg = 'The CUDA is not available!'
        qa_logger.error(err_msg)
        raise RuntimeError(err_msg)
    torch.cuda.random.manual_seed(RANDOM_SEED)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The LLM for answer generation.')
    parser.add_argument('--gpu', dest='gpu_mem_part', type=float, required=False,
                        default=0.9, help='The GPU memory part for the vLLM-based inference.')
    parser.add_argument('-d', '--dataset', dest='dataset_name', type=str, required=False,
                        default='FractalGPT/RRNCBPublic', help='The HF dataset for benchmarking.')
    parser.add_argument('-s', '--split', dest='split_name', type=str, required=False,
                        default='train', help='The split of the HF dataset for benchmarking.')
    parser.add_argument('--beam', dest='beam_size', type=int, required=False,
                        default=1, help='The assistant LLM for answer generation.')
    parser.add_argument('--reranker', dest='reranker_type', type=str, choices=['llm', 'encoder'],
                        help='The used reranker type: the `llm` type corresponds to BAAI/bge-reranker-v2-gemma, '
                             'and the `encoder` type corresponds to `BAAI/bge-reranker-v2-m3`.')
    parser.add_argument('--minibatch', dest='reranker_minibatch', type=int, required=False,
                        default=16, help='The mini-batch size for reranker.')
    parser.add_argument('--eval', dest='evaluate_quality', action='store_true',
                        help='The HF dataset for benchmarking contains reference answers, which will be used to '
                             'evaluate the quality of the generated answers.')
    parser.add_argument('--output', dest='output_dataset', type=str, required=False, default=None,
                        help='The output JSON-line file with generated dataset for RLHF.')
    args = parser.parse_args()

    if args.evaluate_quality:
        chrf = evaluate.load('chrf')
        qa_logger.info('The Chrf++ metric will be used to evaluate the quality of answers.')
    else:
        chrf = None

    emb_name = os.path.join('..', 'models', 'FRIDA')
    if not os.path.isdir(emb_name):
        err_msg = f'The model {emb_name} does not exist!'
        qa_logger.error(err_msg)
        raise IOError(err_msg)

    if args.reranker_type == 'llm':
        reranker_name = os.path.join('..', 'models', 'bge-reranker-v2-gemma')
    else:
        reranker_name = os.path.join('..', 'models', 'bge-reranker-v2-m3')
    if not os.path.isdir(reranker_name):
        err_msg = f'The model {reranker_name} does not exist!'
        qa_logger.error(err_msg)
        raise IOError(err_msg)

    database_dir = os.path.join('..', 'data')
    if not os.path.isdir(database_dir):
        err_msg = f'The database {database_dir} does not exist!'
        qa_logger.error(err_msg)
        raise IOError(err_msg)
    annoy_index_fname = os.path.join(database_dir, 'chunk_vectors_v2.ann')
    if not os.path.isfile(annoy_index_fname):
        err_msg = f'The Annoy index {annoy_index_fname} does not exist!'
        qa_logger.error(err_msg)
        raise IOError(err_msg)
    text_corpus_fname = os.path.join(database_dir, 'final_chunk_list_v2.json')
    if not os.path.isfile(text_corpus_fname):
        err_msg = f'The text corpus {text_corpus_fname} does not exist!'
        qa_logger.error(err_msg)
        raise IOError(err_msg)

    submission_dir = os.path.join('..', 'submissions')
    if not os.path.isdir(submission_dir):
        err_msg = f'The submission directory {submission_dir} does not exist!'
        qa_logger.error(err_msg)
        raise IOError(err_msg)
    submit_fname = os.path.join(submission_dir, 'submit.csv')
    qa_logger.info(f'All answers will be submitted into {submit_fname}')

    if args.output_dataset is None:
        output_dataset_fname = ''
    else:
        if args.evaluate_quality:
            output_dataset_fname = os.path.normpath(args.output_dataset)
            if not os.path.isfile(output_dataset_fname):
                base_dir = os.path.dirname(output_dataset_fname)
                if not os.path.isdir(base_dir):
                    err_msg = f'The directory {base_dir} does not exist!'
                    qa_logger.error(err_msg)
                    raise IOError(err_msg)
            qa_logger.info(f'All training samples for RLHF will be written into {output_dataset_fname}')
        else:
            output_dataset_fname = ''
            warn_msg = ('It makes sense to set an output JSON-line file only in the quality evaluation mode, '
                        'and this mode is not currently set.')
            qa_logger.warning(warn_msg)

    try:
        russian_stemmer = SnowballStemmer(language='russian')
    except Exception as err:
        qa_logger.error(str(err))
        raise
    qa_logger.info('The Russian stemmer is loaded.')

    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as err:
        qa_logger.error(str(err))
        raise
    if llm_tokenizer.padding_side != 'left':
        llm_tokenizer.padding_side = 'left'

    try:
        with codecs.open(text_corpus_fname, mode='r', encoding='utf-8') as src_fp:
            chunk_list = json.load(src_fp)
    except Exception as err:
        qa_logger.error(str(err))
        raise
    num_chunks = len(chunk_list)
    chunk_lengths = []
    for cur_chunk in tqdm(chunk_list, desc='Chunk processing'):
        chunk_lengths.append(len(llm_tokenizer.tokenize(cur_chunk['chunk_text'], add_special_tokens=True)))
    mean_chunk_length = round(np.mean(chunk_lengths))
    std_chunk_length = round(np.std(chunk_lengths))
    max_chunk_length = max(chunk_lengths)
    min_chunk_length = min(chunk_lengths)
    info_msg = (f'The text corpus is loaded from {text_corpus_fname}. This corpus consists of {num_chunks} chunks. '
                f'The mean chunk length (in LLM tokens) is {mean_chunk_length} ± {std_chunk_length}. The maximal '
                f'chunk length is {max_chunk_length}, and the minimal one is {min_chunk_length}.')
    qa_logger.info(info_msg)

    try:
        llm_gen_config = GenerationConfig.from_pretrained(args.model_name)
        llm_config = AutoConfig.from_pretrained(args.model_name)
    except Exception as err:
        qa_logger.error(str(err))
        raise
    if not llm_gen_config.do_sample:
        llm_gen_config.do_sample = True
    llm_gen_config.max_new_tokens = MAX_NUMBER_OF_GENERATED_TOKENS
    llm_sampling_params = SamplingParams(
        temperature=llm_gen_config.temperature,
        top_p=llm_gen_config.top_p,
        top_k=llm_gen_config.top_k,
        repetition_penalty=1.0,
        max_tokens=llm_gen_config.max_new_tokens,
        seed=RANDOM_SEED,
        skip_special_tokens=True
    )
    max_input_len = min(
        max_chunk_length,
        mean_chunk_length + std_chunk_length,
        3 * MAX_TOKENS_PER_CHUNK
    )
    max_input_len *= (FINAL_NUM_BEST_CHUNKS * MAX_NUMBER_OF_SEARCH_QUERIES)
    max_model_len = min(
        llm_gen_config.max_new_tokens + max_input_len,
        llm_tokenizer.model_max_length,
        llm_config.max_position_embeddings
    )
    max_input_len = max_model_len - llm_gen_config.max_new_tokens
    del llm_config, llm_gen_config
    qa_logger.info(f'The LLM context length is {max_model_len}.')

    try:
        main_llm = LLM(
            model=args.model_name,
            gpu_memory_utilization=args.gpu_mem_part,
            max_model_len=max_model_len,
            max_num_batched_tokens=max(16384, max_model_len),
            seed=RANDOM_SEED,
        )
    except Exception as err:
        qa_logger.error(str(err))
        raise
    qa_logger.info(f'The main LLM is loaded from {args.model_name}')

    try:
        emb_tokenizer = AutoTokenizer.from_pretrained(emb_name)
        emb_model = T5EncoderModel.from_pretrained(emb_name, torch_dtype=torch.float32).cpu()
    except Exception as err:
        qa_logger.error(str(err))
        finalize_vllm()
        raise
    qa_logger.info(f'The embedder is loaded from {emb_name}')

    # devices_for_reranker = [f'cuda:{device_id}' for device_id in range(torch.cuda.device_count())]
    devices_for_reranker = ['cuda:0']
    try:
        tokenizer_for_reranker = AutoTokenizer.from_pretrained(reranker_name)
    except Exception as err:
        qa_logger.error(str(err))
        finalize_vllm()
        raise
    max_chunk_length_for_reranker = max(map(
        lambda it: len(tokenizer_for_reranker.tokenize(it['chunk_length'], add_special_tokens=True)),
        tqdm(chunk_list, desc='Chunk processing for reranker')
    ))
    del tokenizer_for_reranker
    qa_logger.info(f'The maximal chunk length for reranker is {max_chunk_length_for_reranker}.')
    if args.reranker_type == 'llm':
        try:
            reranker = FlagLLMReranker(reranker_name, use_fp16=True,
                                       max_length=max(max_chunk_length_for_reranker, MAX_TOKENS_PER_CHUNK),
                                       query_max_length=max(max_chunk_length_for_reranker, MAX_TOKENS_PER_CHUNK),
                                       normalize=False, batch_size=args.reranker_minibatch,
                                       devices=devices_for_reranker)
        except Exception as err:
            qa_logger.error(str(err))
            finalize_vllm()
            raise
        qa_logger.info(f'The LLM-based reranker is loaded from {reranker_name}')
    else:
        try:
            reranker = FlagReranker(reranker_name, use_fp16=False, use_bf16=False,
                                    max_length=max(max_chunk_length_for_reranker, MAX_TOKENS_PER_CHUNK),
                                    query_max_length=max(max_chunk_length_for_reranker, MAX_TOKENS_PER_CHUNK),
                                    normalize=False, batch_size=args.reranker_minibatch,
                                    devices=devices_for_reranker)
        except Exception as err:
            qa_logger.error(str(err))
            finalize_vllm()
            raise
        qa_logger.info(f'The encoder-based reranker is loaded from {reranker_name}')

    vector_dim = emb_model.config.d_model
    try:
        annoy_index = AnnoyIndex(vector_dim, 'angular')
        annoy_index.load(annoy_index_fname)
    except Exception as err:
        qa_logger.error(str(err))
        finalize_vllm()
        raise
    qa_logger.info(f'The Annoy index is loaded from {annoy_index_fname} (the item vector length is {vector_dim}).')

    qa_logger.info('The BM25 index building is started.')
    try:
        bm25_index = prepare_bm25([cur_chunk['chunk_text'] for cur_chunk in chunk_list], russian_stemmer)
    except Exception as err:
        qa_logger.error(str(err))
        finalize_vllm()
        raise
    qa_logger.info('The BM25 index building is successfully finished.')

    try:
        benchmark = load_dataset(args.dataset_name, split=args.split_name, trust_remote_code=True)
        test_questions = [str(it['question']) for it in benchmark]
        if args.evaluate_quality:
            reference_answers = [str(it['answer']) for it in benchmark]
        else:
            reference_answers = None
    except Exception as err:
        qa_logger.error(str(err))
        finalize_vllm()
        raise
    del benchmark
    num_questions = len(test_questions)
    info_msg = (f'The test questions are loaded from {args.dataset_name}[{args.split_name}] '
                f'(there are {num_questions} questions).')
    qa_logger.info(info_msg)

    list_of_generated_answers = []
    list_of_reference_answers = []
    samples_for_rlhf = []
    with codecs.open(submit_fname, mode='w', encoding='utf-8', buffering=0) as dst_fp:
        data_writer = csv.writer(dst_fp, delimiter=';', quotechar='"')
        if args.evaluate_quality:
            data_writer.writerow(['question', 'search queries', 'relevant chunks number', 'answer', 'document',
                                  'reference_answer', 'chrF++'])
        else:
            data_writer.writerow(['question', 'search queries', 'relevant chunks number', 'answer', 'document'])
        for question_count, cur_question in enumerate(test_questions):
            qa_logger.info(f'Processing of the {question_count + 1} question out of {num_questions} has started.')
            num_question_tokens = len(llm_tokenizer.tokenize(cur_question, add_special_tokens=True))
            prompt_for_search = prepare_messages_for_search(
                user_question=cur_question,
                llm_tokenizer=llm_tokenizer
            )
            try:
                search_queries = generate_answer(
                    input_prompt=prompt_for_search,
                    large_language_model=main_llm,
                    llm_tokenizer=llm_tokenizer,
                    sampling_params=llm_sampling_params
                )
            except Exception as err:
                qa_logger.error(str(err))
                finalize_vllm()
                raise
            list_of_search_queries = list(filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: it1.strip(), search_queries.split('\n'))
            ))
            number_of_chunks_for_llm = 0
            if len(list_of_search_queries) == 0:
                warn_msg = (f'Couldn\'t get a response from the large language model. This response is empty. '
                            f'The input prompt:\n{prompt_for_search}')
                qa_logger.warning(warn_msg)
                answer = INSUFFICIENT_INFORMATION_ANSWER
                relevant_documents = []
            else:
                is_question_meaningful = True
                for cur_query in list_of_search_queries:
                    dist = calculate_distance_between_texts(reference=MEANINGLESS_REQUEST_ANSWER, hypothesis=cur_query)
                    if dist <= LEXICAL_DISTANCE_THRESHOLD:
                        is_question_meaningful = False
                        break
                if not is_question_meaningful:
                    answer = INSUFFICIENT_INFORMATION_ANSWER
                    relevant_documents = []
                    info_msg = (f'The question {question_count + 1} does not make sense and therefore cannot be turned '
                                f'into a set of search queries.')
                    qa_logger.info(info_msg)
                else:
                    info_msg = (f'The question {question_count + 1} generated {len(list_of_search_queries)} additional'
                                f' search queries to the database.')
                    qa_logger.info(info_msg)
                    union_of_relevant_indices = dict()
                    for query_count, cur_search_query in enumerate(list_of_search_queries + [cur_question]):
                        try:
                            indices_from_vector_search = find_relevant_chunks_with_vector_search(
                                user_question=cur_search_query,
                                sent_emb_tokenizer=emb_tokenizer,
                                sent_embedder=emb_model,
                                vector_db=annoy_index
                            )
                        except Exception as err:
                            qa_logger.error(str(err))
                            finalize_vllm()
                            raise
                        info_msg = (f'The question {question_count + 1}, query {query_count + 1}: the vector search '
                                    f'returned {len(indices_from_vector_search)} relevant chunks.')
                        qa_logger.info(info_msg)
                        try:
                            indices_from_bm25 = find_relevant_chunks_with_bm25_search(
                                user_question=cur_search_query,
                                stemmer=russian_stemmer,
                                bm25_db=bm25_index,
                                num_chunks=num_chunks
                            )
                        except Exception as err:
                            qa_logger.error(str(err))
                            finalize_vllm()
                            raise
                        info_msg = (f'The question {question_count + 1}, query {query_count + 1}: the BM25 search '
                                    f'returned {len(indices_from_bm25)} relevant chunks.')
                        qa_logger.info(info_msg)
                        united_indices_of_retrieved_chunks = sorted(list(
                            set(indices_from_vector_search) | set(indices_from_bm25)
                        ))
                        del indices_from_vector_search, indices_from_bm25
                        info_msg = (f'The question {question_count + 1}, query {query_count + 1}: there were '
                                    f'{len(united_indices_of_retrieved_chunks)} relevant chunks after combining '
                                    f'the search results.')
                        qa_logger.info(info_msg)
                        if len(united_indices_of_retrieved_chunks) > 0:
                            # extended_list_of_indices = extend_list_of_selected_chunks(
                            #     indices_of_selected_chunks=united_indices_of_retrieved_chunks,
                            #     chunks=chunk_list
                            # )
                            # info_msg = (f'The question {question_count + 1}, query {query_count + 1}: '
                            #             f'after expanding with neighboring chunks, there were '
                            #             f'{len(extended_list_of_indices)} chunks in the search results.')
                            # qa_logger.info(info_msg)
                            try:
                                indices_of_reranked_chunks = rerank_chunks(
                                    user_question=cur_search_query,
                                    # selected_chunks=extended_list_of_indices,
                                    selected_chunks=united_indices_of_retrieved_chunks,
                                    all_chunks=chunk_list,
                                    reranker=reranker,
                                    num_best=FINAL_NUM_BEST_CHUNKS
                                )
                            except Exception as err:
                                qa_logger.error(str(err))
                                finalize_vllm()
                                raise
                            # del extended_list_of_indices
                            num_relevant_chunks = len(indices_of_reranked_chunks)
                            if num_relevant_chunks > 0:
                                info_msg = (f'The question {question_count + 1}, query {query_count + 1}: '
                                            f'{num_relevant_chunks} relevant chunks were found after the reranking.')
                                for chunk_idx in indices_of_reranked_chunks:
                                    if chunk_idx in union_of_relevant_indices:
                                        union_of_relevant_indices[chunk_idx] = max(
                                            union_of_relevant_indices[chunk_idx],
                                            indices_of_reranked_chunks[chunk_idx]
                                        )
                                    else:
                                        union_of_relevant_indices[chunk_idx] = indices_of_reranked_chunks[chunk_idx]
                            else:
                                info_msg = (f'The question {question_count + 1}, query {query_count + 1}: '
                                            f'no relevant chunk was found after the reranking.')
                            qa_logger.info(info_msg)
                            del indices_of_reranked_chunks
                        del united_indices_of_retrieved_chunks
                    number_of_chunks_for_llm = len(union_of_relevant_indices)
                    if number_of_chunks_for_llm == 0:
                        answer = INSUFFICIENT_INFORMATION_ANSWER
                        relevant_documents = []
                    else:
                        info_msg = (f'The question {question_count + 1}: the total number of chunks found for all '
                                    f'search queries was {number_of_chunks_for_llm}.')
                        qa_logger.info(info_msg)
                        prepared_context = chunk_indices_to_context(
                            selected_chunks=union_of_relevant_indices,
                            all_chunks=chunk_list,
                            llm_tokenizer=llm_tokenizer,
                            max_context_len=round(0.9 * max_input_len) - num_question_tokens
                        )
                        relevant_documents = sorted(list(prepared_context.keys()))
                        reduced_number_of_chunks_for_llm = 0
                        for cur_doc in relevant_documents:
                            reduced_number_of_chunks_for_llm += len(prepared_context[cur_doc]['chunks'])
                        if reduced_number_of_chunks_for_llm < number_of_chunks_for_llm:
                            info_msg = (f'The question {question_count + 1}: the total number of chunks found for all '
                                        f'search queries has decreased from {number_of_chunks_for_llm} to '
                                        f'{reduced_number_of_chunks_for_llm}.')
                            qa_logger.info(info_msg)
                        num_docs = len(relevant_documents)
                        if num_docs > 1:
                            info_msg = f'{num_docs} reference documents will be used to answer the question. They are '
                        else:
                            info_msg = 'Only one reference document will be used to answer the question. This is '
                        info_msg += prepare_pretty_description_of_references(prepared_context)
                        qa_logger.info(info_msg)
                        prompt_for_qa, jsonified_promt = prepare_messages_for_answering(
                            user_question=cur_question,
                            context=prepared_context,
                            llm_tokenizer=llm_tokenizer,
                            add_json=True
                        )
                        try:
                            answer = generate_answer(
                                input_prompt=prompt_for_qa,
                                large_language_model=main_llm,
                                llm_tokenizer=llm_tokenizer,
                                sampling_params=llm_sampling_params
                            )
                        except Exception as err:
                            qa_logger.error(str(err))
                            finalize_vllm()
                            raise
                        if len(answer) == 0:
                            warn_msg = (f'Couldn\'t get a response from the large language model. '
                                        f'This response is empty. The input prompt:\n{prompt_for_qa}')
                            qa_logger.warning(warn_msg)
                            answer = INSUFFICIENT_INFORMATION_ANSWER
                        else:
                            if (len(output_dataset_fname) > 0) and args.evaluate_quality:
                                new_sample_for_rlhf = copy.copy(jsonified_promt)
                                new_sample_for_rlhf['response'] = reference_answers[question_count]
                                new_sample_for_rlhf['rejected_response'] = answer
                                new_sample_for_rlhf['history'] = []
                                samples_for_rlhf.append(new_sample_for_rlhf)
                                del new_sample_for_rlhf
                        del jsonified_promt
                    del union_of_relevant_indices
            list_of_generated_answers.append(answer)
            if args.evaluate_quality:
                instant_quality_score = chrf.compute(
                    predictions=[answer],
                    references=[[reference_answers[question_count]]]
                )['score']
                data_writer.writerow([
                    cur_question,
                    search_queries,
                    number_of_chunks_for_llm,
                    answer,
                    ', '.join(relevant_documents) if (len(relevant_documents) > 0) else '',
                    reference_answers[question_count],
                    round(instant_quality_score, 2)
                ])
                list_of_reference_answers.append([reference_answers[question_count]])
            else:
                data_writer.writerow([
                    cur_question,
                    search_queries,
                    number_of_chunks_for_llm,
                    answer,
                    ', '.join(relevant_documents) if (len(relevant_documents) > 0) else ''
                ])
            qa_logger.info(f'Processing of the {question_count + 1} question out of {num_questions} has finished.')
            torch.cuda.empty_cache()
            gc.collect()
    info_msg = 'The question answering is successfully finished.'
    if args.evaluate_quality:
        quality_score = chrf.compute(
            predictions=list_of_generated_answers,
            references=list_of_reference_answers
        )['score']
        info_msg += f' The total ChrF++ for {args.dataset_name}[{args.split_name}] is {round(quality_score, 2)}.'
    qa_logger.info(info_msg)
    finalize_vllm()
    if (len(output_dataset_fname) > 0) and (len(samples_for_rlhf) > 0):
        with codecs.open(output_dataset_fname, mode='w', encoding='utf-8') as jsonl_fp:
            for it in samples_for_rlhf:
                jsonl_fp.write(json.dumps(it, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    qa_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    qa_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('question_answering_with_vllm.log')
    file_handler.setFormatter(formatter)
    qa_logger.addHandler(file_handler)
    main()
