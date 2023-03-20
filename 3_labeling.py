""""
This module contains several functions for manipulating words, specifically for inserting, removing,
replacing, and swapping characters in a word. It also includes a function for checking if a word is
valid, which checks if a word is composed of only alphabetic characters, has at least a certain number
of characters, and is not in a list of stop words. These functions are used to generate variations of
words, possibly for the purpose of data augmentation or error analysis.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple
import argparse
import random
from string import ascii_lowercase

from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import set_seed


def is_valid_word(
    word: str,
    vocab: dict,
    min_num_characters: int = 2,
    stop_words: Optional[List[str]] = None,
) -> bool:
    """
    Check if a word is valid.

    A word is considered valid if it is composed only of alphabetic characters, 
    has at least `min_num_characters` characters (default is 2), and is not in 
    the list of `stop_words` (if provided).

    Args:
        word (str): The word to check.
        min_num_characters (int, optional): The minimum number of characters required
            for the word to be considered valid. Defaults to 2.
        stop_words (Optional[List[str]], optional): A list of stop words to exclude.
            Defaults to None.

    Raises:
        ValueError: If `min_num_characters` is less than 2.

    Returns:
        bool: True if the word is valid, False otherwise.
    """
    if min_num_characters < 2:
        raise ValueError("The minimum number of characters must be greater than or equal to 2")
    if stop_words is None:
        stop_words = []
    return word.encode().isalpha() and len(word) >= min_num_characters and word not in stop_words and word.lower() in vocab


def insert_char(word: str) -> str:
    """
    Insert a random lowercase letter into a word.

    This function takes a word as input and randomly selects a position in the word
    at which to insert a lowercase letter from the alphabet. It returns the resulting
    string.

    Args:
        word (str): The word to insert a character into.

    Returns:
        str: The resulting string with the character inserted at a random position.
    """
    chars = list(word)
    sample_index = random.randint(0, len(word) - 1)
    sample_alpha = random.choice(ascii_lowercase)
    chars.insert(sample_index, sample_alpha)
    return "".join(chars)


def remove_char(word: str) -> str:
    """
    Remove a random character from a word.

    This function takes a word as input and randomly selects a character from the word
    to remove. It returns the resulting string.

    Args:
        word (str): The word to remove a character from.

    Returns:
        str: The resulting string with the character removed.
    """
    chars = list(word)
    sample_index = random.randint(0, len(word) - 1)
    chars.pop(sample_index)
    return "".join(chars)


def replace_char(word: str) -> str:
    """
    Replace a random character in a word with a random lowercase letter.

    This function takes a word as input and randomly selects a character from the word
    to replace with a lowercase letter from the alphabet. It returns the resulting string.

    Args:
        word (str): The word to replace a character in.

    Returns:
        str: The resulting string with the character replaced.
    """
    chars = list(word)
    sample_index = random.randint(0, len(word) - 1)
    sample_alpha = random.choice(ascii_lowercase)
    chars[sample_index] = sample_alpha
    return "".join(chars)


def swap_adjacent_char(word: str) -> str:
    """
    Swap two adjacent characters in a word.

    This function takes a word as input and randomly selects a pair of adjacent characters
    from the word to swap. It returns the resulting string. If the word has only one
    character or no characters, a `ValueError` is raised.

    Args:
        word (str): The word to swap characters in.

    Raises:
        ValueError: If the word has only one character or no characters.

    Returns:
        str: The resulting string with the two characters swapped.
    """
    chars = list(word)
    index_range = len(word) - 2
    if index_range < 0:
        raise ValueError("Word length is at least 2")
    sample_index = random.randint(0, index_range)
    tmp_alpha = chars[sample_index]
    chars[sample_index] = chars[sample_index + 1]
    chars[sample_index + 1] = tmp_alpha
    return "".join(chars)


STRATEGY_FN_CHAR: Dict[int, Callable] = {
    0: insert_char,
    1: remove_char,
    2: replace_char,
    3: swap_adjacent_char,
}


def generate_typo(word: str) -> str:
    """
    Generate a typo for a word.

    This function takes a word as input and randomly applies one of the following
    transformations to it:
    - Insert a random lowercase letter at a random position in the word
    - Remove a random character from the word
    - Replace a random character in the word with a random lowercase letter
    - Swap two adjacent characters in the word

    If the resulting word is the same as the original word, the function will try
    again up to a maximum of 10 times. If the maximum number of attempts is reached,
    a `ValueError` is raised.

    Args:
        word (str): The word to generate a typo for.

    Raises:
        ValueError: If the maximum number of attempts is reached.

    Returns:
        str: The resulting word with a randomly generated typo.
    """
    max_iter = 10
    while True:
        max_iter -= 1
        typo = STRATEGY_FN_CHAR[random.choice(tuple(STRATEGY_FN_CHAR))](word)
        if typo != word:
            return typo
        if max_iter == 0:
            raise ValueError(f"Typos are regenerated too many times (maximum {max_iter} times)")


def prepare_misspelling_label(
    sentence: str,
    typo_probability: float,
    max_num_typos: int,
    word_to_index: Dict[str, int],
    data_multiple: int = 1,
    min_num_characters: Optional[int] = None,
    num_typo_weights: Sequence[int] = None,
    stop_words: Optional[List[str]] = None,
) -> List[Tuple[List[str], List[str], List[int], int]]:
    """
    This function prepares a label for a sentence with a certain probability of having typos.

    Args:
        sentence (str): The sentence to be labeled.
        typo_probability (float): The probability of generating typos in the sentence.
        max_num_typos (int): The maximum number of typos allowed in the sentence.
        word_to_index (Dict[str, int]): A dictionary mapping words to their IDs.
        data_multiple (int, optional): Number of times to repeat the process. Defaults to 1.
        min_num_characters (Optional[int], optional): The minimum number of characters a word must have to be considered valid. Defaults to None.
        num_typo_weights (Sequence[int], optional): Weights for the number of typos to generate, used for random.choices. Defaults to None.
        stop_words (Optional[List[str]], optional): List of words to be considered stop words and ignored. Defaults to None.

    Returns:
        List[Tuple[List[str], List[str], List[int], int]]:  A list of tuples, each containing the original sentence, the sentence with the typos,
            a list of the word IDs for the original sentence, and the length of the original sentence.
    """
    label_data = []
    if len(sentence) < 5:
        return []

    words = sentence.split(" ")
    is_typo = random.random() < typo_probability
    valid_mask = np.array([is_valid_word(word, word_to_index, min_num_characters, stop_words) for word in words])
    if not is_typo or not any(valid_mask):
        label_data.append(
            (
                words,
                words,
                [word_to_index[word.lower()] if word.lower() in word_to_index else 0 for word in words],
                len(words),
            )
        )
        return label_data

    n_valid_words = sum(valid_mask)
    while True:
        num_typos = random.choices([n + 1 for n in range(max_num_typos)], weights=num_typo_weights)[0]
        if num_typos <= n_valid_words:
            break

    valid_indexes = np.arange(len(words))[valid_mask]
    sample_indexes: List[int] = random.sample(valid_indexes.tolist(), num_typos)
    for _ in range(data_multiple):
        typo_words = list(words)
        for sample_index in sample_indexes:
            typo_words[sample_index] = generate_typo(typo_words[sample_index])
        label_data.append(
            (
                words,
                typo_words,
                [word_to_index[word.lower()] if word.lower() in word_to_index else 0 for word in words],
                len(words),
            )
        )
    return label_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="The path to the input file in LineSentence format. (one line = one sentence.)")
    parser.add_argument("--output_file", type=str, required=True, help="The path to the output CSV file. This file will contain the sentences with typos added.")
    parser.add_argument("--vocab_file", type=str, required=True, help="The path to the vocabulary file in LineWord format. (one line = one word.)")
    parser.add_argument("--stop_words_file", default=None, type=str, help="The path to a file containing stop words in LineWord format. These words will not be modified by the script.")
    parser.add_argument("--typo_probability", default=0.5, type=float, help="The probability of adding typos to the sentence.")
    parser.add_argument("--max_num_typos", default=3, type=int, help="The maximum number of typos to add to the sentence.")
    parser.add_argument("--data_multiple", default=4, type=int, help="The multiple of the original data size that will be generated.")
    parser.add_argument("--min_num_characters", default=4, type=int, help="The minimum number of characters required for a word to be considered as will generate a misspelling.")
    parser.add_argument("--num_typo_weights", default=(5, 4, 1), type=Sequence[int], help="The weights for sampling the number of typos to add.")
    parser.add_argument("--max_length", default=None, type=int, help="The maximum length of the generated sentences.")
    parser.add_argument("--seed", default=2330, type=int, help="The random seed to use.")
    args = parser.parse_args()

    WORDS = "words"
    TYPO_WORDS = "typo_words"
    LABELS = "labels"
    LENGTH = "length"

    # ===== Set seed =====
    set_seed(args.seed)

    # ===== Load vocabulary file=====
    with open(args.vocab_file, "r", encoding="utf-8") as f:
        word_to_index = {line.strip(): i for i, line in enumerate(f.readlines())}

    # ===== Load stop words =====
    stop_words = [
        word.strip() for word in open(args.stop_words_file, encoding="utf-8").readlines()
    ]

    # ===== generate label data =====
    label_data = []
    with open(args.input_file, "r", encoding="utf-8") as finput:
        for sentence in tqdm(finput.readlines(), desc="Generating label data..."):
            outputs = prepare_misspelling_label(
                sentence.strip(),
                typo_probability=args.typo_probability,
                max_num_typos=args.max_num_typos,
                word_to_index=word_to_index,
                data_multiple=args.data_multiple,
                min_num_characters=args.min_num_characters,
                num_typo_weights=args.num_typo_weights,
                stop_words=stop_words,
            )
            label_data.extend(outputs)

        df_label = pd.DataFrame(label_data, columns=[WORDS, TYPO_WORDS, LABELS, LENGTH])
        if args.max_length is None:
            df_label.to_csv(args.output_file, index=False)
        else:
            df_label[df_label[LENGTH] <= args.max_length].to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
