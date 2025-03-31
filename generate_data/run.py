"""Generate questions from documents"""

import os
import logging
import argparse
from typing import List, Optional
import json
from tqdm import tqdm

import yaml
from litellm import completion
from pydantic import BaseModel

# from loguru import logger

import config
import parser
import prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_config(config_path: str) -> config.Config:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    return config.Config(**yaml_config)


def run_llm(
    llm_config: config.LLMConfig,
    messages: List[dict],
) -> str:
    """use litellm to generate json data"""
    response = completion(
        model=llm_config.model,
        messages=messages,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        max_tokens=llm_config.max_tokens,
        temperature=llm_config.temperature,
    )
    return response.choices[0].message.content


class Questions(BaseModel):
    questions: List[str]


def generate_questions(
    llm_config: config.LLMConfig,
    context: str,
    target_num: Optional[int] = 2,
) -> List[str]:
    """generate questions from chunk"""
    messages = [
        {
            "role": "user",
            "content": prompts.GENERATE_QUESTION_PROMPT.format(
                context=context,
                target_num=target_num,
            ),
        },
    ]
    resp = run_llm(llm_config, messages)
    questions = resp.split(",")
    return questions


class QuestionAnswerPair(BaseModel):
    question: str
    answer: str
    source: str
    chunk_id: int
    chunk: str


def generate_answer(
    question: str,
    context: str,
    llm_config: config.LLMConfig,
) -> str:
    """generate answer with reference from question and chunk"""
    messages = [
        {
            "role": "user",
            "content": prompts.ANSWER_WITH_REFERENCE_PROMPT.format(
                context=context, question=question
            ),
        },
    ]
    resp = run_llm(llm_config, messages)
    return resp


class QAPair(BaseModel):
    question: str
    answer: str
    source: str
    chunk_id: int
    chunk: str

    @property
    def train_data(self) -> str:
        return json.dumps(
            {
                "instruction": self.question,
                "input": "",
                "response": self.answer,
            },
            ensure_ascii=False,
        )


def generate_qa_pairs(
    doc_path: str,
    output_dir: str,
    llm_config: config.LLMConfig,
    generate_num: int,
    chunk_size: int,
    chunk_overlap: int,
) -> List[QuestionAnswerPair]:
    """generate question answer pairs from chunk"""
    logger.info(f"Start processing document: {doc_path}")
    # convert docx to markdown
    chunks = parser.load_chunks(doc_path, chunk_size, chunk_overlap)
    doc_name = os.path.split(doc_path)[-1].replace(".", "_")
    output_save_path = os.path.join(
        output_dir,
        f"{doc_name}.jsonl",
    )

    if os.path.exists(output_save_path):
        qa_pairs = [QAPair(**json.loads(line)) for line in open(output_save_path, "r")]
    else:
        qa_pairs = []

    for chunk_idx, chunk in enumerate(tqdm(chunks)):
        # generate questions
        questions = generate_questions(
            context=chunk,
            llm_config=llm_config,
            target_num=generate_num,
        )
        chunk_qa_pairs_num = len([x for x in qa_pairs if x.chunk_id == chunk_idx])
        if chunk_qa_pairs_num >= generate_num:
            logger.info(f"Chunk {chunk_idx} has {chunk_qa_pairs_num} qa pairs, skip")
            continue
        for question in questions:
            answer = generate_answer(
                question=question,
                context=chunk,
                llm_config=llm_config,
            )

            qa_pairs.append(
                QAPair(
                    question=question,
                    answer=answer,
                    source=doc_name,
                    chunk_id=chunk_idx,
                    chunk=chunk,
                )
            )
            # overwrite the file
            with open(output_save_path, "w") as f:
                for qa_pair in qa_pairs:
                    f.write(qa_pair.model_dump_json() + "\n")
    return output_save_path


def merge_qa_pairs(save_path: str, qa_pairs_paths: List[str]):
    """merge qa pairs from multiple files to one json file"""
    if os.path.exists(save_path):
        os.remove(save_path)
    for qa_pairs_path in qa_pairs_paths:
        qa_pairs = [
            QAPair(**json.loads(line)).dict() for line in open(qa_pairs_path, "r")
        ]
    with open(save_path, "w") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)


def main(config_path: str, input_path: str, output_dir: str):
    """generate questions from documents"""
    # create output dir
    os.makedirs(output_dir, exist_ok=True)

    # load config
    config = read_config(config_path)
    logger.info(f"Config Loaded: \n{config}")

    # load data
    if os.path.isdir(input_path):
        docs_paths = [
            x.path
            for x in os.scandir(input_path)
            if x.is_file()
            and x.name.split(".")[-1].lower() in parser.get_valid_extensions()
        ]
    else:
        docs_paths = [input_path]
    if len(docs_paths) == 0:
        raise ValueError(f"No valid documents found in input path: ({input_path})")
    logger.info(f"Docs Paths: \n{docs_paths}")
    qa_pairs_paths = []
    for idx, doc_path in enumerate(docs_paths):
        logger.info(
            f"Start processing document: {doc_path}, {len(docs_paths) - idx} left, {idx} processed"
        )
        qa_pairs_path = generate_qa_pairs(
            doc_path=doc_path,
            output_dir=output_dir,
            llm_config=config.llm,
            generate_num=config.args.generate_num,
            chunk_size=config.args.chunk_size,
            chunk_overlap=config.args.chunk_overlap,
        )
        qa_pairs_paths.append(qa_pairs_path)

    # merge qa pairs
    train_data_path = os.path.join(output_dir, "train_data.json")
    merge_qa_pairs(train_data_path, qa_pairs_paths)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/generate-example.yaml")
    parser.add_argument("--input", type=str, default="test_data/test.txt")
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--mode", type=str, default="llm", choices=["llm"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.input, args.output)
