"""
Sentence Autocomplete using GPT-Neo model.

This script takes a partial sentence as input and returns
3 autocomplete suggestions using GPT-Neo with
Top-K sampling and temperature control.
"""

import warnings
warnings.filterwarnings("ignore")

import logging
from transformers import GPTNeoForCausalLM, GPT2TokenizerFast


# ---------------- Logging Setup ----------------
def setup_logger():
    """
    Configure logging to write to autocomplete.log by default.
    """
    log_file = "autocomplete.log"

    # Remove existing handlers (important for reruns)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    logger = logging.getLogger()
    logger.info("Logger initialized. Writing logs to autocomplete.log")
    return logger


# ---------------- Model Configuration ----------------
MODEL_NAME = "EleutherAI/gpt-neo-125M"
TOP_K = 50
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 30
NUM_OUTPUTS = 3


# ---------------- Load Model & Tokenizer ----------------
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token
model.eval()


# ---------------- Stopping Criteria ----------------
def stop_at_sentence(text: str) -> str:
    """
    Stop generation at the first complete sentence.
    """
    if "." in text:
        return text[: text.find(".") + 1]
    return text


# ---------------- Autocomplete Function ----------------
def autocomplete_sentence(prompt: str, logger):
    """
    Generate top 3 autocomplete sentences for a given prompt.
    Logs model details, parameters, and generated outputs.
    """

    # Log request metadata
    logger.info("=" * 60)
    logger.info(f"MODEL_NAME: {MODEL_NAME}")
    logger.info(f"PROMPT: {prompt}")
    logger.info(
        f"GEN_PARAMS | top_k={TOP_K}, temperature={TEMPERATURE}, "
        f"max_new_tokens={MAX_NEW_TOKENS}, num_outputs={NUM_OUTPUTS}"
    )

    encodings = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True
    )

    outputs = model.generate(
        input_ids=encodings.input_ids,
        attention_mask=encodings.attention_mask,
        do_sample=True,
        top_k=TOP_K,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=NUM_OUTPUTS,
        pad_token_id=tokenizer.eos_token_id
    )

    completions = []
    for idx, out in enumerate(outputs, start=1):
        text = tokenizer.decode(out, skip_special_tokens=True)
        text = stop_at_sentence(text)
        completions.append(text)

        # Log each generated answer
        logger.info(f"OUTPUT_{idx}: {text}")

    return completions


# ---------------- Main Execution ----------------
if __name__ == "__main__":
    logger = setup_logger()

    prompt = input("Enter a partial sentence: ").strip()
    results = autocomplete_sentence(prompt, logger)

    print("\nAutocomplete Suggestions:")
    for i, r in enumerate(results, start=1):
        print(f"{i}. {r}")
