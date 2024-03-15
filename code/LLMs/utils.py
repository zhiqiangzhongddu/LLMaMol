import re
import tiktoken


# Percentage Difference
def compute_percentage_difference(pred1, pred2):
    return abs((pred1 - pred2) / ((pred1 + pred2) / 2)) * 100


# TODO
def extract_prediction_from_response():
    return None


def check_response(response, pred, task_type, previous_response=None):
    if "classification" in task_type:
        pred_label = 1 if pred > 0.5 else 0

        if "Prediction: <number>" in response:
            return "format"
        else:
            if response.find("Prediction: ") != -1:
                prediction = response.split("Prediction: ")[1][:7]

                if "true" in prediction.lower():
                    if pred_label == 1:
                        return "next"
                    else:
                        return "double-check"
                elif "false" in prediction.lower():
                    if pred_label == 0:
                        return "next"
                    else:
                        return "double-check"
                else:
                    return "format"

            else:
                return "format"
    else:
        if "Prediction: <True or False>" in response or "Probability: N/A" in response:
            return "format"
        else:
            if response.find("Prediction: ") != -1:
                prediction = response.split("Prediction: ")[1]
                match = re.search(r'[-+]?\d+(\.\d+)?', prediction)
                if match:
                    prediction_number = float(match.group())
                    if previous_response is not None:
                        previous_number = float(re.search(r'[-+]?\d+(\.\d+)?', previous_response).group())
                    else:
                        previous_number = pred
                    difference = compute_percentage_difference(
                        pred1=prediction_number, pred2=previous_number
                    )
                    # print("difference: ", difference)
                    if difference > 20:
                        return "double-check"
                    else:
                        return "next"
                else:
                    return "format"
            else:
                return "format"


def get_context_window_size_limit(llm_model):
    if llm_model in ["gpt-3.5-turbo-1106", "gpt-35-turbo-1106", "gpt-35-turbo-16k"]:
        return 16385
    elif llm_model in ["gpt-4-32k"]:
        return 32768
    elif llm_model in ["gpt-4-0125-preview", "gpt-4-turbo-preview", "gpt-4-1106"]:
        return 128000
    else:
        return 0


def num_tokens_from_messages(messages, original_model="gpt-3.5-turbo-0613"):
    # Align the self-defined model deployment on Azure with OpenAI's model name
    model = original_model if original_model not in ["gpt-35-turbo-16k", "gpt-4-32k", "gpt-4-1106"] else "gpt-3.5-turbo-0613"

    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, original_model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, original_model="gpt-4-0613")
    else:
        raise NotImplementedError(
            "num_tokens_from_messages() is not implemented for model {}. "
            "See https://github.com/openai/openai-python/blob/main/chatml.md "
            "for information on how messages are converted to tokens.".format(model)
        )

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

    return num_tokens

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
