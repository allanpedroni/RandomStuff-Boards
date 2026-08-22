from openai import BadRequestError, NotFoundError, OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import math
import os

# export AZURE_OPENAI_ENDPOINT="https://YOUR-RESOURCE.openai.azure.com" #bash
# $env:AZURE_OPENAI_ENDPOINT="https://foundryagentproject-resource.services.ai.azure.com/openai/v1" #PowerShell

ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
DEPLOYMENT = "gpt-4.1"
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")

#Only show top 10 to keep the display on screen.
TOP_N = 10
# Start with low variability for the first demonstration. 1.2 would be more creative
temperature = 0.2

CONTINUATION_INSTRUCTIONS = (
    "You are a text-completion engine for an educational token-prediction demo. "
    "Continue the user's text directly. Do not answer, discuss, quote, or introduce "
    "the text. Your first output token must be the natural next token of the text. "
    "Whitespace is part of the token. Never omit required leading whitespace because "
    "this is a new assistant message. If the user's text ends in a letter or digit "
    "and the continuation starts with a word, the output token must begin with an "
    "ASCII space; a word token without that space is invalid."
)

client = OpenAI(
    base_url=ENDPOINT,
    api_key=token_provider
)

# HELPERS

def clear_screen():
    """Clear the terminal and return the cursor to the top-left corner."""
    print("\033[2J\033[H", end="", flush=True)


def to_probability(logprob):
    """Convert natural-log probability to percentage."""
    return math.exp(logprob) * 100


def visible_token(token):
    """
    Make spaces, newlines and tabs visible.

    For example:
        " technology" -> "␠technology"
        "\n"           -> "↵"
    """

    return (
        token
        .replace(" ", "␠")
        .replace("\n", "↵")
        .replace("\t", "⇥")
    )


def display_distribution(token_data, inference_number):
    """
    Display the top possible next tokens and their probabilities.
    """

    selected_token = token_data.token
    candidates = token_data.top_logprobs

    print(f"\nNEXT-TOKEN DISTRIBUTION FROM INFERENCE #{inference_number}")

    if not candidates:
        print("No top-logprob information returned.")
        return

    probabilities = [
        to_probability(candidate.logprob)
        for candidate in candidates
    ]

    max_probability = max(probabilities)

    total_top_n_probability = sum(probabilities)

    print(
        f"{'#':<4}"
        f"{'TOKEN':<25}"
        f"{'PROBABILITY':>12}   "
        f"DISTRIBUTION"
    )

    print("-" * 90)

    for index, candidate in enumerate(candidates, start=1):

        probability = to_probability(candidate.logprob)

        # Maximum visual bar size = 35 characters
        bar_length = round(
            (probability / max_probability) * 35
        ) if max_probability else 0

        bar = "█" * bar_length

        token_text = visible_token(candidate.token)

        selected = candidate.token == selected_token
        marker = "  ◀ SELECTED" if selected else ""

        print(
            f"{index:<4}"
            f"{repr(token_text):<25}"
            f"{probability:>10.3f}%   "
            f"{bar}"
            f"{marker}"
        )

    print("-" * 90)

    print(
        f"Top {len(candidates)} coverage: {total_top_n_probability:.3f}%   "
        f"All other tokens: {max(0.0, 100 - total_top_n_probability):.3f}%"
    )


# INTRO

clear_screen()
print("=" * 90)
print("              HOW AN LLM GENERATES TEXT")
print("=" * 90)

print()
print("This demonstration requests ONE TOKEN from the model")
print(f"and shows the {TOP_N} most likely tokens it could have generated.")
print()


original_prompt = input("Enter the starting text:\n\n> ")
prompt = original_prompt

generated = ""
token_number = 1
last_token_data = None


# GENERATION LOOP

while True:

    full_context = prompt + generated

    clear_screen()
    print("=" * 90)
    print("              HOW AN LLM GENERATES TEXT")
    print("=" * 90)

    print(f"\nCURRENT TEXT — READY FOR INFERENCE #{token_number}")
    print(full_context)

    if last_token_data:
        display_distribution(last_token_data, token_number - 1)

    print(f"\nTemperature: {temperature}")

    command = input(
        "[ENTER] Infer next token   "
        "[T] Change temperature   "
        "[Q] Quit\n> "
    ).strip().lower()


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if command == "q":
        break


    # --------------------------------------------------------
    # CHANGE TEMPERATURE
    # --------------------------------------------------------

    if command == "t":

        value = input(
            "\nEnter temperature "
            "(for example 0.2, 0.5, 1.0, 1.2): "
        )

        try:

            new_temperature = float(value)

            if 0 <= new_temperature <= 2:
                temperature = new_temperature

                new_prompt = input(
                    f"\nEnter starting text [{original_prompt}]: "
                )

                prompt = new_prompt if new_prompt else original_prompt
                generated = ""
                token_number = 1
                last_token_data = None

            else:
                print("\nUse a value between 0 and 2.")

        except ValueError:
            print("\nInvalid temperature.")

        continue


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print()
    print(">>> RUNNING INFERENCE...")
    print()

    try:
        response = client.chat.completions.create(

            model=DEPLOYMENT,

            messages=[
                {
                    "role": "system",
                    "content": CONTINUATION_INSTRUCTIONS
                },
                {
                    "role": "user",
                    "content": full_context
                }
            ],

            # Generate exactly ONE output token.
            max_completion_tokens=1,

            # Sampling control.
            temperature=temperature,

            # Return probability information.
            logprobs=True,

            # Give us the top candidates.
            top_logprobs=TOP_N
        )
    except NotFoundError:
        print(
            f"Deployment {DEPLOYMENT!r} was not found at {ENDPOINT!r}.\n"
            "Set AZURE_OPENAI_DEPLOYMENT to a deployed model name."
        )
        break
    except BadRequestError as error:
        if "logprobs" not in str(error).lower():
            raise

        print(
            f"Deployment {DEPLOYMENT!r} does not support log probabilities.\n"
            "Use a deployed model with logprobs support, such as GPT-4.1."
        )
        break


    # ========================================================
    # EXTRACT THE ONE GENERATED TOKEN
    # ========================================================

    choice = response.choices[0]

    if not choice.logprobs:
        print("The model did not return log probabilities.")
        break

    if not choice.logprobs.content:
        print("No output token was returned.")
        break

    token_data = choice.logprobs.content[0]

    selected_token = token_data.token


    # APPEND TOKEN TO CONTEXT

    generated += selected_token
    last_token_data = token_data

    token_number += 1


# FINISHED

clear_screen()
print("=" * 90)
print("FINAL GENERATED TEXT")
print("=" * 90)

print()
print(prompt + generated)
print()