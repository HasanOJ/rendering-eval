"""Load the original pinned Hub JSONL (no dataset-script execution)."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .config import Config


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Example:
    sample_id: str
    passage_id: str
    language: str
    passage: str
    question: str
    choices: tuple[str, str, str, str]
    answer: str

    def task_text(self):
        # The exact same string is sent as text and rendered for the image condition.
        options = "\n".join(f"{label}. {value}" for label, value in zip("ABCD", self.choices))
        return f"{self.passage}\n\n{self.question}\n\n{options}"


def normalize_rows(rows, language):
    examples = {}
    for row in rows:
        if row.get("dialect") != language:
            raise ValueError(f"Unexpected dialect in {language}")
        # Original docs include a passage-level split. The pinned Hub export omits it.
        # Use it if present, otherwise require (link, question_number) to be unique.
        passage_key = [row["link"], row.get("split")]
        sample_id = digest([*passage_key, int(row["question_number"])])
        if sample_id in examples:
            raise ValueError(f"Ambiguous question identity in {language}: {passage_key}")
        fields = [row["flores_passage"], row["question"], *(row[f"mc_answer{i}"] for i in range(1, 5))]
        if any(not isinstance(x, str) or not x.strip() for x in fields):
            raise ValueError(f"Empty or non-string content in {language}: {sample_id}")
        answer_num = str(row["correct_answer_num"])
        if answer_num not in "1234" or len(answer_num) != 1:
            raise ValueError(f"Invalid answer index: {answer_num}")
        examples[sample_id] = Example(
            sample_id, digest(passage_key), language, fields[0], fields[1],
            tuple(fields[2:]), "ABCD"[int(answer_num) - 1],
        )
    return examples


def align_and_select(rows_by_language, config: Config):
    if set(rows_by_language) != set(config.languages):
        raise ValueError("Downloaded language set differs from configuration")
    normalized = {lang: normalize_rows(rows_by_language[lang], lang) for lang in config.languages}
    first = normalized[config.languages[0]]
    for lang, examples in normalized.items():
        if set(examples) != set(first):
            raise ValueError(f"Question IDs do not align for {lang}")
        if any(examples[k].answer != first[k].answer for k in first):
            raise ValueError(f"Correct answers do not align for {lang}")
    if len(first) < config.n_ids:
        raise ValueError("Not enough aligned questions")
    # Stable across Python versions, unaffected by input row ordering.
    ids = sorted(first, key=lambda key: digest([config.seed, key]))[:config.n_ids]
    return [normalized[lang][key] for key in ids for lang in config.languages]


def load_examples(config: Config):
    from huggingface_hub import hf_hub_download
    rows = {}
    for lang in config.languages:
        path = hf_hub_download(
            repo_id=config.dataset_id, repo_type="dataset", revision=config.dataset_revision,
            filename=f"data/{lang}.jsonl",
        )
        rows[lang] = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if len(rows[lang]) != 900:
            raise ValueError(f"Expected 900 rows in {lang}, found {len(rows[lang])}")
    return align_and_select(rows, config)
