import json
import os
import sys
from transformers import AutoTokenizer

from finetune import finetune
from inference import LLaMaEvaluator
# from llama_train import llama_finetune
from utils.parser import parse_args
from utils.prompter import Prompter
from utils.utils import dir_init, createLogFile, load_dataset, prepare_dataset, merge_dataset_passages, augment_dataset, topicList
import pickle
from loguru import logger
import wandb


def initLogging(args):
    try:
        import git  ## pip install gitpython
    except:
        pass
    filename = args.log_name  # f'{args.time}_{"DEBUG" if args.debug else args.log_name}_{args.model_name.replace("/", "_")}_log.txt'
    filename = os.path.join(args.log_dir, filename)
    logger.remove()
    fmt = "<green>{time:YYMMDD_HH:mm:ss}</green> | {message}"
    if not args.debug: logger.add(filename, format=fmt, encoding='utf-8')
    logger.add(sys.stdout, format=fmt, level="INFO", colorize=True)
    logger.info(f"FILENAME: {filename}")
    try:
        logger.info(f"Git commit massages: {git.Repo(search_parent_directories=True).head.object.hexsha[:7]}")
    except:
        pass
    logger.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
    return logger


if __name__ == "__main__":

    # fire.Fire(llama_finetune)
    args = parse_args()
    args = dir_init(args)
    args = createLogFile(args)
    print(args)

    initLogging(args)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    train_raw_dataset, test_raw_dataset = load_dataset(args)
    args.topicList = topicList(train_raw_dataset, test_raw_dataset)

    train_know_dataset, train_labels, train_topics = prepare_dataset(args, tokenizer, train_raw_dataset)
    test_know_dataset, test_labels, test_topics = prepare_dataset(args, tokenizer, test_raw_dataset)

    prompter = Prompter(args, args.prompt)
    train_instructions = prompter.generate_instructions('train', train_know_dataset, train_labels)
    test_instructions = prompter.generate_instructions('test', test_know_dataset, test_labels)

    if args.mode == 'train':
        finetune(args, tokenizer=tokenizer, instructions=train_instructions, labels=train_labels, num_epochs=args.epoch)
    elif args.mode == 'test':
        LLaMaEvaluator(args=args, tokenizer=tokenizer, insturctions=test_instructions, labels=test_labels, topics=test_topics, prompt_template_name=args.prompt).test()
