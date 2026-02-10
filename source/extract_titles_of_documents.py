from argparse import ArgumentParser
import codecs
import json
import logging
import os
import random
import signal
import sys
from typing import List, Union

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoConfig, GenerationConfig
from vllm import LLM, SamplingParams
from vllm import EngineArgs, LLMEngine
from tqdm import tqdm


DOC_EXTRACT_SYSTEM_PROMPT = ('Вы - эксперт в области науки, техники и юриспруденции. Вы безукоризненно вежливы, а '
                             'ваши ответы на сложные вопросы человека всегда логичны и точны. Вы способны разобраться '
                             'даже в самой сложной документации, чтобы помочь человеку извлечь из этой документации '
                             'полезные сведения для ответа на вопрос. Вы глубоко разбираетесь в документообороте на '
                             'русском языке.')
DOC_EXTRACT_USER_PROMPT = ('Внимательно проанализируйте первые фрагменты некоторого документа, представленные '
                           'в формате JSON:\n\n```json\n{jsonified_context}\n```\n\nОпираясь на сведения из этих '
                           'фрагментов, ответьте на вопрос, какое название носит этот документ? В качестве ответа '
                           'напишите только название документа без дополнительных пояснений или комментариев. '
                           'При написании названия не пишите всё подряд капслоком, а используйте заглавные буквы '
                           'только там, где это использование обосновано правилами русского языка. Для аббревиатур '
                           'типа ГОСТ, СНиП и тому подобных использование капслока разрешается (если в исходном '
                           'документе они записаны большими буквами, то такими и должны оставаться). Не используйте '
                           'markdown. Если вы считаете, что это вообще не похоже на отдельный документ, то сообщите '
                           'об этом пользователю примерно так: "Это не отдельный документ".')
FEW_SHOT_PROMPTS = [
    (
        (
            "В раздел Документация\nРаспечатать\nСИСТЕМА НОРМАТИВНЫХ ДОКУМЕНТОВ В СТРОИТЕЛЬСТВЕ\nСТРОИТЕЛЬНЫЕ НОРМЫ И "
            "ПРАВИЛА РОССИЙСКОЙ ФЕДЕРАЦИИ\nУтверждены и введены в действие с 15 июля 2001 г.\nПостановлением Госстроя "
            "России от 23 июля 2001 года № 85\nУКАЗАНИЯ\nПО ПРИМЕНЕНИЮ ГОСУДАРСТВЕННЫХ\nЭЛЕМЕНТНЫХ СМЕТНЫХ НОРМ\n"
            "НА СТРОИТЕЛЬНЫЕ И СПЕЦИАЛЬНЫЕ\nСТРОИТЕЛЬНЫЕ РАБОТЫ\n(ГЭСН-2001)\nМДС 81-28.2001\nГОСУДАРСТВЕННЫЙ КОМИТЕТ "
            "РОССИЙСКОЙ ФЕДЕРАЦИИ",  # 0
            "СИСТЕМА НОРМАТИВНЫХ ДОКУМЕНТОВ В СТРОИТЕЛЬСТВЕ\nСТРОИТЕЛЬНЫЕ НОРМЫ И ПРАВИЛА РОССИЙСКОЙ ФЕДЕРАЦИИ\n"
            "Утверждены и введены в действие с 15 июля 2001 г.\nПостановлением Госстроя России от 23 июля 2001 года "
            "№ 85\nУКАЗАНИЯ\nПО ПРИМЕНЕНИЮ ГОСУДАРСТВЕННЫХ\nЭЛЕМЕНТНЫХ СМЕТНЫХ НОРМ\nНА СТРОИТЕЛЬНЫЕ И СПЕЦИАЛЬНЫЕ\n"
            "СТРОИТЕЛЬНЫЕ РАБОТЫ\n(ГЭСН-2001)\nМДС 81-28.2001\nГОСУДАРСТВЕННЫЙ КОМИТЕТ РОССИЙСКОЙ ФЕДЕРАЦИИ\n"
            "ПО СТРОИТЕЛЬСТВУ И ЖИЛИЩНО-КОММУНАЛЬНОМУ КОМПЛЕКСУ",  # 1
            "(ГОССТРОЙ РОССИИ)\nМосква 2001\nНастоящие Указания к сборникам государственных элементных сметных норм "
            "на\nстроительные и специальные строительные работы ГЭСН-2001 содержат основные\nсведения\nпо\n"
            "применению\nгосударственных\nэлементных\nсметных\nнорм,\nраспространяются на ГЭСН по приведенной "
            "номенклатуре (Приложение 1).\nРАЗРАБОТАНЫ\nМежрегиональным\nцентром\nпо\nценообразованию\nв\n"
            "строительстве и промышленности строительных материалов (МЦЦС) Госстроя\n"
            "России (И.И. Дмитренко, Г.П. Шпунт), при участии специалистов - 31 ГПИ СС МО\nРФ, "
            "(В.Г. Гурьев, А.Н. Жуков), ЦНИИЭУС Госстроя России, (к.т.н. Ж.Г. Чернышова).",  # 2
            "РАССМОТРЕНЫ Управлением ценообразования и сметного нормирования в\nстроительстве и "
            "жилищно-коммунальном комплексе Госстроя России (Редакционная\nкомиссия: В.А. Степанов - руководитель, "
            "В.Н. Маклаков, Г.А. Шанин, Т.Л.\nГрищенкова).\nВНЕСЕНЫ\nУправлением\nценообразования\nи\nсметного\n"
            "нормирования\nв\nстроительстве и жилищно-коммунальном комплексе Госстроя России.\nПРИНЯТЫ И ВВЕДЕНЫ "
            "В ДЕЙСТВИЕ с 15 июля 2001 постановлением\nГосстроя России от 23 июля 2001 г. № 85.\nВВЕДЕНИЕ\n"
            "Настоящие Указания по применению государственных элементных сметных норм\nна строительные и "
            "специальные строительные работы ГЭСН-2001 (в дальнейшем\n«Указания»)\nустанавливают\nединый\nпорядок",  # 3
        ),
        'Указания по применению государственных элементных сметных норм на строительные и специальные строительные '
        'работы (ГЭСН-2001) МДС 81-28.2001'
    ),
    (
        (
            "Сооружения инженерной защиты городских территорий от подтопления (дренажная система)\nК таблице 42\n"
            "№ п/п\nНомер пункта\nНаименование объекта\nпроектирования\nСтадия проектирования\nГидротехническая "
            "часть\nТехнико-экономические\nпоказатели\nОхрана окружающей\nприродной среды\nВедомости спецификации "
            "материалов и\nоборудования\nПОС\nСметная\nдокументация\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n1.\nпп. 1 - 5\n"
            "Дренажная система\nП\n62\n5\n5\n-\n15\n13\nР\n89,5\n-\n-\n1\n-\n9,5\nРП\n83,5\n1\n1\n1\n3\n10,5\n"
            "В раздел Документация\nРаспечатать",  # 0
            "В раздел Документация\nРаспечатать\nГосударственный строительный комитет СССР\n(Госстрой СССР)\n"
            "ПОЛОЖЕНИЕ\nо заказчике-застройщике\n(едином заказчике, дирекции строящего предприятия)\nи техническом "
            "надзоре\nУтверждено постановлением Госстроя СССР\nот 2 февраля 1988 г. № 16\nпо согласованию с Госпланом "
            "СССР, Минфином СССР,\nПромстройбанком СССР, Госкомтрудом СССР\nМосква 1989\nРАЗРАБОТАНО ЦНИИЭУС Госстроя "
            "СССР (кандидаты техн. наук Т.Н. Комарова, А.К.\nБчемян), Госкомархитектуры (канд. техн. наук А.В. "
            "Охрименко; И.И. Макаров), ЦНИИП\nградостроительства (Н.П. Сугробов).\nПОДГОТОВЛЕНО\nК",  # 1
            "Государственный строительный комитет СССР\n(Госстрой СССР)\nПОЛОЖЕНИЕ\nо заказчике-застройщике\n"
            "(едином заказчике, дирекции строящего предприятия)\nи техническом надзоре\nУтверждено постановлением "
            "Госстроя СССР\nот 2 февраля 1988 г. № 16\nпо согласованию с Госпланом СССР, Минфином СССР,\n"
            "Промстройбанком СССР, Госкомтрудом СССР\nМосква 1989\nРАЗРАБОТАНО ЦНИИЭУС Госстроя СССР (кандидаты "
            "техн. наук Т.Н. Комарова, А.К.\nБчемян), Госкомархитектуры (канд. техн. наук А.В. Охрименко; "
            "И.И. Макаров), ЦНИИП\nградостроительства (Н.П. Сугробов).\nПОДГОТОВЛЕНО\nК\nУТВЕРЖДЕНИЮ\nГлавным\n"
            "управлением",  # 2
            "экономики\nи\nсовершенствования хозяйственного механизма Госстроя СССР (Н.М. Балыкова, В.Г.\nМильто).\n"
            "Определяет основные задачи и функции служб заказчика-застройщика и технического\nнадзора. Устанавливает "
            "порядок их создания, управления и ликвидации.\nДля инженерно-технических работников служб "
            "заказчика-застройщика, технического\nнадзора, планирующих и финансирующих органов, проектных и "
            "строительных\nорганизаций.\nС введением в действие «Положения о заказчике-застройщике "
            "(едином заказчике,\nдирекции строящегося предприятия) и техническом надзоре» утрачивает силу «Положение\n"
            "о дирекции строящегося предприятия» (утвержденное постановлением Госстроя СССР от\n12 марта 1971 № 17) и "
            "«Положение о службе единого заказчика по строительству в городах\nжилых домов, объектов "
            "культурно-бытового назначения и коммунального хозяйства»",  # 3
        ),
        'Положение о заказчике-застройщике (едином заказчике, дирекции строящего предприятия) и техническом надзоре'
    )
]
RANDOM_SEED: int = 42
MAX_OUTPUT_TOKENS: int = 256
THINKING_END_TOKEN = '</think>'
title_extraction_logger = logging.getLogger(__name__)


def finalize_vllm():
    if torch.distributed.is_initialized():
        if hasattr(LLMEngine, 'shutdown'):
            LLMEngine.shutdown()
        torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()


def handle_exit(signal, frame):
    finalize_vllm()
    sys.exit(0)


def prepare_messages_for_search(chunks: List[str],
                                llm_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> str:
    if len(chunks) > len(FEW_SHOT_PROMPTS[0][0]):
        err_msg = (f'The number of input chunks is wrong! Expected less than or equal to {len(FEW_SHOT_PROMPTS[0][0])},'
                   f' got {len(chunks)}.')
        raise ValueError(err_msg)
    messages = [
        {
            'role': 'system',
            'content': DOC_EXTRACT_SYSTEM_PROMPT
        }
    ]
    chunk_names = ['first chunk', 'second chunk', 'third chunk', 'fourth chunk']
    for cur_example in FEW_SHOT_PROMPTS:
        jsonified_context = dict(zip(chunk_names, cur_example[0]))
        messages += [
            {
                'role': 'user',
                'content': DOC_EXTRACT_USER_PROMPT.format(
                    jsonified_context=json.dumps(jsonified_context, ensure_ascii=False, indent=4)
                )
            },
            {
                'role': 'assistant',
                'content': cur_example[1]
            }
        ]
        del jsonified_context
    if len(chunks) < len(chunk_names):
        jsonified_context = dict(zip(chunk_names[0:len(chunks)], chunks))
    else:
        jsonified_context = dict(zip(chunk_names, chunks))
    messages.append({
        'role': 'user',
        'content': DOC_EXTRACT_USER_PROMPT.format(jsonified_context=jsonified_context)
    })
    del jsonified_context
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


def generate_answers(input_prompts: List[str], large_language_model: LLM,
                     llm_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                     sampling_params: SamplingParams) -> List[str]:
    outputs = large_language_model.generate(input_prompts, sampling_params, use_tqdm=True)
    responses = []
    for answer_idx in range(len(input_prompts)):
        output_ids = outputs[answer_idx].outputs[0].token_ids
        think_end_pos = 0
        if THINKING_END_TOKEN in llm_tokenizer.vocab:
            try:
                think_end_pos = len(output_ids) - output_ids[::-1].index(llm_tokenizer.vocab[THINKING_END_TOKEN])
            except ValueError:
                pass
        response = llm_tokenizer.decode(output_ids[think_end_pos:], skip_special_tokens=True).strip()
        del output_ids
        responses.append(response)
    del outputs
    return responses


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.random.manual_seed(RANDOM_SEED)
    if not torch.cuda.is_available():
        err_msg = 'The CUDA is not available!'
        title_extraction_logger.error(err_msg)
        raise RuntimeError(err_msg)
    torch.cuda.random.manual_seed(RANDOM_SEED)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The LLM for answer generation.')
    parser.add_argument('--gpu', dest='gpu_mem_part', type=float, required=False,
                        default=0.9, help='The GPU memory part for the vLLM-based inference.')
    args = parser.parse_args()

    database_dir = os.path.join('..', 'data')
    if not os.path.isdir(database_dir):
        err_msg = f'The database {database_dir} does not exist!'
        title_extraction_logger.error(err_msg)
        raise IOError(err_msg)
    text_corpus_fname = os.path.join(database_dir, 'final_chunk_list_v2.json')
    if not os.path.isfile(text_corpus_fname):
        err_msg = f'The text corpus {text_corpus_fname} does not exist!'
        title_extraction_logger.error(err_msg)
        raise IOError(err_msg)
    doc_titles_fname = os.path.join(database_dir, 'titles_of_documents.json')

    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as err:
        title_extraction_logger.error(str(err))
        raise
    if llm_tokenizer.padding_side != 'left':
        llm_tokenizer.padding_side = 'left'

    try:
        with codecs.open(text_corpus_fname, mode='r', encoding='utf-8') as src_fp:
            chunk_list = json.load(src_fp)
    except Exception as err:
        title_extraction_logger.error(str(err))
        raise
    if not isinstance(chunk_list, list):
        err_msg = (f'The chunk list from {text_corpus_fname} has a wrong type! '
                   f'Expected {type([1, 2])}, got {type(chunk_list)}.')
        title_extraction_logger.error(err_msg)
        raise ValueError(err_msg)
    documents = dict()
    for idx, val in enumerate(chunk_list):
        if not isinstance(val, dict):
            err_msg = (f'The chunk {idx} from {text_corpus_fname} has a wrong type! '
                       f'Expected {type({"a": 1})}, got {type(val)}.')
            title_extraction_logger.error(err_msg)
            raise ValueError(err_msg)
        if 'chunk_text' not in val:
            err_msg = (f'The chunk {idx} from {text_corpus_fname} has a wrong content! '
                       f'The field "chunk_text" is not found.')
            title_extraction_logger.error(err_msg)
            raise ValueError(err_msg)
        if not isinstance(val['chunk_text'], str):
            err_msg = (f'The chunk {idx} from {text_corpus_fname} has a wrong type of the "chunk_text" field! '
                       f'Expected {type("123")}, got {type(val["chunk_text"])}.')
            title_extraction_logger.error(err_msg)
            raise ValueError(err_msg)
        if 'document' not in val:
            err_msg = (f'The chunk {idx} from {text_corpus_fname} has a wrong content! '
                       f'The field "document" is not found.')
            title_extraction_logger.error(err_msg)
            raise ValueError(err_msg)
        if not isinstance(val['document'], str):
            err_msg = (f'The chunk {idx} from {text_corpus_fname} has a wrong type of the "document" field! '
                       f'Expected {type("123")}, got {type(val["document"])}.')
            title_extraction_logger.error(err_msg)
            raise ValueError(err_msg)
        if 'chunk_index' not in val:
            err_msg = (f'The chunk {idx} from {text_corpus_fname} has a wrong content! '
                       f'The field "chunk_index" is not found.')
            title_extraction_logger.error(err_msg)
            raise ValueError(err_msg)
        if not isinstance(val['chunk_index'], int):
            err_msg = (f'The chunk {idx} from {text_corpus_fname} has a wrong type of the "chunk_index" field! '
                       f'Expected {type(123)}, got {type(val["chunk_index"])}.')
            title_extraction_logger.error(err_msg)
            raise ValueError(err_msg)
        if val['document'] not in documents:
            documents[val['document']] = []
        if val['chunk_index'] < 4:
            documents[val['document']].append(val['chunk_text'])
    if len(documents) == 0:
        err_msg = f'The text corpus {text_corpus_fname} is empty!'
        title_extraction_logger.error(err_msg)
        raise ValueError(err_msg)
    for doc_fname in documents:
        if len(documents[doc_fname]) < 1:
            err_msg = f'The document {doc_fname} does not contain any chunk!'
            title_extraction_logger.error(err_msg)
            raise ValueError(err_msg)
    num_chunks = len(chunk_list)
    num_documents = len(documents)

    chunk_lengths = []
    for cur_chunk in tqdm(chunk_list, desc='Chunk processing'):
        chunk_lengths.append(len(llm_tokenizer.tokenize(cur_chunk['chunk_text'], add_special_tokens=True)))
    mean_chunk_length = round(np.mean(chunk_lengths))
    std_chunk_length = round(np.std(chunk_lengths))
    max_chunk_length = max(chunk_lengths)
    min_chunk_length = min(chunk_lengths)
    info_msg = (f'The text corpus is loaded from {text_corpus_fname}. This corpus consists of {num_documents} '
                f'documents, split into {num_chunks} chunks. The mean chunk length (in LLM tokens) is '
                f'{mean_chunk_length} ± {std_chunk_length}. The maximal chunk length is {max_chunk_length}, '
                f'and the minimal one is {min_chunk_length}.')
    title_extraction_logger.info(info_msg)

    list_of_input_prompts = []
    lengths_of_input_prompts = []
    for doc_fname in tqdm(sorted(list(documents.keys())), desc='Prompts preparing'):
        list_of_input_prompts.append(
            prepare_messages_for_search(chunks=documents[doc_fname], llm_tokenizer=llm_tokenizer)
        )
        lengths_of_input_prompts.append(
            len(llm_tokenizer.tokenize(list_of_input_prompts[-1], add_special_tokens=True))
        )
    max_input_len = max(lengths_of_input_prompts)

    try:
        llm_gen_config = GenerationConfig.from_pretrained(args.model_name)
        llm_config = AutoConfig.from_pretrained(args.model_name)
    except Exception as err:
        title_extraction_logger.error(str(err))
        raise
    if not llm_gen_config.do_sample:
        llm_gen_config.do_sample = True
    llm_gen_config.max_new_tokens = MAX_OUTPUT_TOKENS
    llm_sampling_params = SamplingParams(
        temperature=llm_gen_config.temperature,
        top_p=llm_gen_config.top_p,
        top_k=llm_gen_config.top_k,
        repetition_penalty=1.0,
        max_tokens=llm_gen_config.max_new_tokens,
        seed=RANDOM_SEED,
        skip_special_tokens=True
    )
    max_model_len = min(
        llm_gen_config.max_new_tokens + max_input_len,
        llm_tokenizer.model_max_length,
        llm_config.max_position_embeddings
    )
    del llm_config, llm_gen_config
    title_extraction_logger.info(f'The LLM context length is {max_model_len}.')

    try:
        main_llm = LLM(
            model=args.model_name,
            gpu_memory_utilization=args.gpu_mem_part,
            max_model_len=max_model_len,
            max_num_batched_tokens=max(16384, max_model_len),
            seed=RANDOM_SEED,
        )
    except Exception as err:
        title_extraction_logger.error(str(err))
        raise
    title_extraction_logger.info(f'The main LLM is loaded from {args.model_name}')

    try:
        titles_of_documents = generate_answers(
            input_prompts=list_of_input_prompts,
            large_language_model=main_llm,
            llm_tokenizer=llm_tokenizer,
            sampling_params=llm_sampling_params
        )
    except Exception as err:
        title_extraction_logger.error(str(err))
        finalize_vllm()
        raise
    finalize_vllm()
    if len(titles_of_documents) != len(documents):
        err_msg = (f'The number of source documents does not correspond to the number of generated titles! '
                   f'{len(documents)} != {len(titles_of_documents)}.')
        title_extraction_logger.error(err_msg)
        raise RuntimeError(err_msg)
    title_dictionary = dict()
    for doc_fname, src_doc_title in zip(sorted(list(documents.keys())), titles_of_documents):
        prep_doc_title = ' '.join(src_doc_title.strip().split())
        title_prefix = 'название документа:'
        if prep_doc_title.lower().startswith(title_prefix):
            prep_doc_title = prep_doc_title[len(title_prefix):].strip()
        if prep_doc_title.lower().find('не отдельный документ') >= 0:
            prep_doc_title = ''
        elif prep_doc_title.lower().find('не документ') >= 0:
            prep_doc_title = ''
        title_dictionary[doc_fname] = prep_doc_title
    with codecs.open(doc_titles_fname, mode='w', encoding='utf-8') as fp:
        json.dump(obj=title_dictionary, fp=fp, ensure_ascii=False, indent=4)
    title_extraction_logger.info(f'The titles of {len(title_dictionary)} documents were successfully generated.')


if __name__ == '__main__':
    title_extraction_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    title_extraction_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('title_extraction.log')
    file_handler.setFormatter(formatter)
    title_extraction_logger.addHandler(file_handler)
    main()
