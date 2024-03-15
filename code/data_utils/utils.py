import os
import numpy as np
import json
import pickle
from pathlib import PurePath
from sklearn.metrics.pairwise import cosine_similarity
import torch

from code.utils import init_path, project_root_path


def save_rag_knowledge(
        dataset_name, list_knowledge, knowledge_type,
        gnn_model=None, seed=None, demo_test=False
):
    if not demo_test:
        if gnn_model is None:
            file_path = PurePath(
                project_root_path, "input", "rag_knowledge", dataset_name,
                "rag_knowledge_{}_{}.json".format(dataset_name, knowledge_type)
            )
        else:
            file_path = PurePath(
                project_root_path, "input", "rag_knowledge", dataset_name,
                "rag_knowledge_{}_{}_{}_seed{}.json".format(
                    dataset_name, knowledge_type, gnn_model, seed
                )
            )
    else:
        if gnn_model is None:
            file_path = PurePath(
                project_root_path, "input", "rag_knowledge", dataset_name,
                "demo_rag_knowledge_{}_{}.json".format(dataset_name, knowledge_type)
            )
        else:
            file_path = PurePath(
                project_root_path, "input", "rag_knowledge", dataset_name,
                "demo_rag_knowledge_{}_{}_{}_seed{}.json".format(
                    dataset_name, knowledge_type, gnn_model, seed
                )
            )

    init_path(dir_or_file=file_path)

    with open(file_path, 'w') as file:
        json.dump(list_knowledge, file, indent=2)


def load_rag_knowledge(
        dataset_name, knowledge_type, gnn_model=None, seed=None, demo_test=False
):
    if not demo_test:
        if gnn_model is None:
            file_path = PurePath(
                project_root_path, "input", "rag_knowledge", dataset_name,
                "rag_knowledge_{}_{}.json".format(dataset_name, knowledge_type)
            )
        else:
            file_path = PurePath(
                project_root_path, "input", "rag_knowledge", dataset_name,
                "rag_knowledge_{}_{}_{}_seed{}.json".format(
                    dataset_name, knowledge_type, gnn_model, seed
                )
            )
    else:
        if gnn_model is None:
            file_path = PurePath(
                project_root_path, "input", "rag_knowledge", dataset_name,
                "demo_rag_knowledge_{}_{}.json".format(dataset_name, knowledge_type)
            )
        else:
            file_path = PurePath(
                project_root_path, "input", "rag_knowledge", dataset_name,
                "demo_rag_knowledge_{}_{}_{}_seed{}.json".format(
                    dataset_name, knowledge_type, gnn_model, seed
                )
            )

    with open(file_path, 'r') as file:
        list_knowledge = json.load(file)

    return list_knowledge


def load_gnn_predictions(
        dataset_name, gnn_model_name, feature, lm_model_name, seed
):
    if feature == 'raw':
        output_dir = PurePath(
            project_root_path, "output", "gnns", dataset_name,
            "{}-{}-seed{}".format(gnn_model_name, feature, seed)
        )
    else:
        output_dir = PurePath(
            project_root_path, "output", "gnns", dataset_name,
            "{}-{}-{}-seed{}".format(gnn_model_name, feature, lm_model_name, seed)
        )
    
    file_path = "{}/predictions.pt".format(output_dir)
    predictions = torch.load(file_path)

    return predictions


def check_gnn_predictions(dataset_name, gnn_model_name, feature, lm_model_name, seed):
    if feature == 'raw':
        output_dir = PurePath(
            project_root_path, "output", "gnns", dataset_name,
            "{}-{}-seed{}".format(gnn_model_name, feature, seed)
        )
    else:
        output_dir = PurePath(
            project_root_path, "output", "gnns", dataset_name,
            "{}-{}-{}-seed{}".format(gnn_model_name, feature, lm_model_name, seed)
        )

    file_path = "{}/predictions.pt".format(output_dir)
    if os.path.exists(file_path):
        return True
    else:
        return False


def save_description(dataset_name, list_description, description_type, demo_test=False):

    if not demo_test:
        file_path = PurePath(
            project_root_path, "input", "description", dataset_name,
            "description_{}_{}.json".format(dataset_name, description_type)
        )
    else:
        file_path = PurePath(
            project_root_path, "input", "description", dataset_name,
            "demo_description_{}_{}.json".format(dataset_name, description_type)
        )

    init_path(dir_or_file=file_path)
    with open(file_path, 'w') as file:
        json.dump(list_description, file, indent=2)


def load_description(dataset_name, description_type="full", demo_test=False):

    if not demo_test:
        file_path = PurePath(
            project_root_path, "input", "description", dataset_name,
            "description_{}_{}.json".format(dataset_name, description_type)
        )
    else:
        file_path = PurePath(
            project_root_path, "input", "description", dataset_name,
            "demo_description_{}_{}.json".format(dataset_name, description_type)
        )

    with open(file_path, 'r') as file:
        list_description = json.load(file)

    return list_description


def save_open_ai_embedding(dataset_name, embedding, embedding_model, embedding_type, demo_test=False):

    if not demo_test:
        file_path = PurePath(
            project_root_path, "input", "embedding", embedding_type, dataset_name,
            "embedding_{}_{}_{}.npz".format(dataset_name, embedding_type, embedding_model)
        )
    else:
        file_path = PurePath(
            project_root_path, "input", "embedding", embedding_type, dataset_name,
            "demo_embedding_{}_{}_{}.npz".format(dataset_name, embedding_type, embedding_model)
        )

    init_path(dir_or_file=file_path)
    # np.savetxt(file_path, np.array(list_embedding))
    np.savez(file_path, embedding=embedding)


def load_openai_embedding(dataset_name, embedding_model, embedding_type, demo_test=False):

    if not demo_test:
        file_path = PurePath(
            project_root_path, "input", "embedding", embedding_type, dataset_name,
            "embedding_{}_{}_{}.npz".format(dataset_name, embedding_type, embedding_model)
        )
    else:
        file_path = PurePath(
            project_root_path, "input", "embedding", embedding_type, dataset_name,
            "demo_embedding_{}_{}_{}.npz".format(dataset_name, embedding_type, embedding_model)
        )

    # array_embedding = np.loadtxt(file_path)
    embedding = np.load(file_path)['embedding']

    return embedding


def save_similarity_matrix(
        dataset_name, similarity_matrix, embedding_model, embedding_type,
        similarity_type="cosine", demo_test=False
):
    if not demo_test:
        file_path = PurePath(
            project_root_path, "input", "distance", embedding_type, dataset_name,
            "distance_{}_{}_{}_{}.npz".format(
                dataset_name, embedding_type, embedding_model, similarity_type
            )
        )
    else:
        file_path = PurePath(
            project_root_path, "input", "distance", embedding_type, dataset_name,
            "demo_distance_{}_{}_{}_{}.npz".format(
                dataset_name, embedding_type, embedding_model, similarity_type
            )
        )

    init_path(dir_or_file=file_path)
    np.savez(file_path, cosine_distance_matrix=similarity_matrix)


def load_similarity_matrix(
        dataset_name, embedding_model, embedding_type,
        similarity_type="cosine", demo_test=False
):
    if not demo_test:
        file_path = PurePath(
            project_root_path, "input", "distance", embedding_type, dataset_name,
            "distance_{}_{}_{}_{}.npz".format(
                dataset_name, embedding_type, embedding_model, similarity_type
            )
        )
    else:
        file_path = PurePath(
            project_root_path, "input", "distance", embedding_type, dataset_name,
            "demo_distance_{}_{}_{}_{}.npz".format(
                dataset_name, embedding_type, embedding_model, similarity_type
            )
        )

    similarity_matrix = np.load(file_path)['cosine_distance_matrix']

    return similarity_matrix


def get_similarity_matrix(
        dataset_name, embedding_model, embedding_type,
        similarity_type="cosine", demo_test=False
):
    if not demo_test:
        file_path = PurePath(
            project_root_path, "input", "distance", embedding_type, dataset_name,
            "distance_{}_{}_{}_{}.npz".format(
                dataset_name, embedding_type, embedding_model, similarity_type
            )
        )
    else:
        file_path = PurePath(
            project_root_path, "input", "distance", embedding_type, dataset_name,
            "demo_distance_{}_{}_{}_{}.npz".format(
                dataset_name, embedding_type, embedding_model, similarity_type
            )
        )
    if os.path.exists(file_path):
        # print("load similarity matrix")
        similarity_matrix = load_similarity_matrix(
            dataset_name=dataset_name, embedding_model=embedding_model,
            embedding_type=embedding_type, similarity_type=similarity_type
        )
    else:
        # print("generate similarity matrix")
        embeddings = load_openai_embedding(
            dataset_name=dataset_name,
            embedding_model=embedding_model,
            embedding_type=embedding_type
        )

        if similarity_type == "cosine":
            similarity_matrix = cosine_similarity(embeddings)
            save_similarity_matrix(
                dataset_name=dataset_name, embedding_model=embedding_model, embedding_type=embedding_type,
                similarity_matrix=similarity_matrix, similarity_type=similarity_type
            )
        else:
            raise Exception("Unknown similarity metric: {}".format(similarity_type))

    return similarity_matrix


def get_llm_file_path(
        dataset_name, template, llm_model, data_format,
        object_model, seed=None, sample_id=-1, demo_test=False,
):
    if sample_id < 0:
        if not demo_test:
            if object_model is None:
                folder_path = PurePath(
                    project_root_path, "output", data_format,
                    dataset_name, llm_model,
                )
                file_path = PurePath(
                    folder_path,
                    "{}_{}_{}.json".format(data_format, dataset_name, template)
                )
            else:
                folder_path = PurePath(
                    project_root_path, "output", data_format,
                    dataset_name, llm_model,
                )
                file_path = PurePath(
                    folder_path,
                    "{}_{}_{}_{}_seed{}.json".format(
                        data_format, dataset_name, template, object_model, seed
                    )
                )
        else:
            if object_model is None:
                folder_path = PurePath(
                    project_root_path, "output", data_format,
                    dataset_name, llm_model,
                )
                file_path = PurePath(
                    folder_path,
                    "demo_{}_{}_{}.json".format(data_format, dataset_name, template)
                )
            else:
                folder_path = PurePath(
                    project_root_path, "output", data_format,
                    dataset_name, llm_model,
                )
                file_path = PurePath(
                    folder_path,
                    "demo_{}_{}_{}_{}_seed{}.json".format(
                        data_format, dataset_name, template, object_model, seed
                    )
                )
    else:
        if not demo_test:
            if object_model is None:
                folder_path = PurePath(
                    project_root_path, "output", data_format,
                    dataset_name, llm_model,
                    "cache", template,
                )
                file_path = PurePath(
                    folder_path,
                    "{}_{}_{}_{}.json".format(data_format, dataset_name, template, sample_id)
                )
            else:
                folder_path = PurePath(
                    project_root_path, "output", data_format,
                    dataset_name, llm_model,
                    "cache", "_".join([template, object_model, str(seed)]),
                )
                file_path = PurePath(
                    folder_path,
                    "{}_{}_{}_{}_seed{}_{}.json".format(
                        data_format, dataset_name, template, object_model, seed, sample_id
                    )
                )
        else:
            if object_model is None:
                folder_path = PurePath(
                    project_root_path, "output", data_format,
                    dataset_name, llm_model, "cache", template,
                )
                file_path = PurePath(
                    folder_path,
                    "demo_{}_{}_{}_{}.json".format(data_format, dataset_name, template, sample_id)
                )
            else:
                folder_path = PurePath(
                    project_root_path, "output", data_format, dataset_name, llm_model,
                    "cache", "_".join([template, object_model, str(seed)])
                )
                file_path = PurePath(
                    folder_path,
                    "demo_{}_{}_{}_{}_seed{}_{}.json".format(
                        data_format, dataset_name, template, object_model, seed, sample_id
                    )
                )
    return file_path, folder_path


def check_llm_cache(
        dataset_name, template, llm_model, data_format,
        operate_object="gnn", gnn_model=None, lm_model=None, seed=None, sample_id=-1,
        demo_test=False,
):
    lm_model = lm_model.replace("/", "-") if lm_model is not None else lm_model
    object_model = gnn_model if operate_object == "gnn" else lm_model

    file_path, _ = get_llm_file_path(
        dataset_name=dataset_name, template=template, llm_model=llm_model, data_format=data_format,
        object_model=object_model, seed=seed, sample_id=sample_id, demo_test=demo_test,
    )

    if os.path.exists(file_path):
        return True
    else:
        return False


def save_llm_outputs(
        dataset_name, outputs, template, llm_model, data_format,
        operate_object="gnn", gnn_model=None, lm_model=None, seed=None, sample_id=-1,
        demo_test=False
):
    lm_model = lm_model.replace("/", "-") if lm_model is not None else lm_model
    object_model = gnn_model if operate_object == "gnn" else lm_model

    file_path, _ = get_llm_file_path(
        dataset_name=dataset_name, template=template, llm_model=llm_model, data_format=data_format,
        object_model=object_model, seed=seed, sample_id=sample_id, demo_test=demo_test,
    )

    init_path(dir_or_file=file_path)
    with open(file_path, 'w') as file:
        json.dump(outputs, file, indent=2)


def load_llm_outputs(
        dataset_name, template, llm_model, data_format,
        operate_object="gnn", gnn_model=None, lm_model=None, seed=None, sample_id=-1,
        demo_test=False
):
    lm_model = lm_model.replace("/", "-") if lm_model is not None else lm_model
    object_model = gnn_model if operate_object == "gnn" else lm_model

    file_path, _ = get_llm_file_path(
        dataset_name=dataset_name, template=template, llm_model=llm_model, data_format=data_format,
        object_model=object_model, seed=seed, sample_id=sample_id, demo_test=demo_test,
    )

    with open(file_path, 'r') as file:
        outputs = json.load(file)

    return outputs


def save_message(
        dataset_name, list_message, message_type,
        gnn_model=None, seed=None, demo_test=False
):
    if not demo_test:
        if gnn_model is None:
            file_path = PurePath(
                project_root_path, "input", "message", dataset_name,
                "message_{}_{}.json".format(dataset_name, message_type)
            )
        else:
            file_path = PurePath(
                project_root_path, "input", "message", dataset_name,
                "message_{}_{}_{}_seed{}.json".format(
                    dataset_name, message_type, gnn_model, seed
                )
            )
    else:
        if gnn_model is None:
            file_path = PurePath(
                project_root_path, "input", "message", dataset_name,
                "demo_message_{}_{}.json".format(dataset_name, message_type)
            )
        else:
            file_path = PurePath(
                project_root_path, "input", "message", dataset_name,
                "demo_message_{}_{}_{}_seed{}.json".format(
                    dataset_name, message_type, gnn_model, seed
                )
            )

    init_path(dir_or_file=file_path)
    with open(file_path, 'w') as file:
        json.dump(list_message, file, indent=2)


def load_message(
        dataset_name, message_type,
        gnn_model=None, seed=None,  demo_test=False
):
    if not demo_test:
        if gnn_model is None:
            file_path = PurePath(
                project_root_path, "input", "message", dataset_name,
                "message_{}_{}.json".format(dataset_name, message_type)
            )
        else:
            file_path = PurePath(
                project_root_path, "input", "message", dataset_name,
                "message_{}_{}_{}_seed{}.json".format(
                    dataset_name, message_type, gnn_model, seed
                )
            )
    else:
        if gnn_model is None:
            file_path = PurePath(
                project_root_path, "input", "message", dataset_name,
                "demo_message_{}_{}.json".format(dataset_name, message_type)
            )
        else:
            file_path = PurePath(
                project_root_path, "input", "message", dataset_name,
                "demo_message_{}_{}_{}_seed{}.json".format(
                    dataset_name, message_type, gnn_model, seed
                )
            )

    with open(file_path, 'r') as file:
        list_message = json.load(file)

    return list_message


def save_prompt(dataset_name, list_prompt, prompt_type):

    file_path = PurePath(
        project_root_path, "input", "prompt",
        dataset_name, "prompt_{}_{}.json".format(dataset_name, prompt_type)
    )
    init_path(dir_or_file=file_path)
    with open(file_path, 'w') as file:
        json.dump(list_prompt, file, indent=2)


def load_prompt(dataset_name, prompt_type):

    file_path = PurePath(
        project_root_path, "input", "prompt",
        dataset_name, "prompt_{}_{}.json".format(dataset_name, prompt_type)
    )
    with open(file_path, 'r') as file:
        list_prompt = json.load(file)

    return list_prompt


def save_caption(dataset_name, list_caption, demo_test=False):

    if demo_test:
        file_name = PurePath(
            project_root_path, "input", "caption",
            "test_smiles2caption_{}.json".format(dataset_name)
        )
    else:
        file_name = PurePath(
            project_root_path, "input", "caption",
            "smiles2caption_{}.json".format(dataset_name)
        )

    init_path(dir_or_file=file_name)
    with open(file_name, 'w') as file:
        json.dump(list_caption, file, indent=2)


def load_caption(dataset_name, demo_test=False):

    if demo_test:
        file_name = PurePath(
            project_root_path, "input", "caption",
            "test_smiles2caption_{}.json".format(dataset_name)
        )
    else:
        file_name = PurePath(
            project_root_path, "input", "caption",
            "smiles2caption_{}.json".format(dataset_name)
        )

    with open(file_name, 'r') as file:
        list_caption = json.load(file)

    return list_caption


def load_lm_predictions(dataset_name, task, num_graphs, template, lm_model_name, seed):
    file_name = PurePath(
        project_root_path, "output", "prt_lms", dataset_name, template,
        "{}-seed{}.pred".format(lm_model_name, seed)
    )

    if "classification".lower() in task.lower():
        predictions = np.memmap(
            filename=file_name,
            dtype=np.float16,
            mode='r',
            shape=(num_graphs, 2)
        )
    else:
        predictions = np.memmap(
            filename=file_name,
            dtype=np.float16,
            mode='r',
            shape=(num_graphs, 1)
        )

    return torch.tensor(predictions)


def check_lm_predictions(dataset_name, template, lm_model_name, seed):
    file_name = PurePath(
        project_root_path, "output", "prt_lms", dataset_name, template,
        "{}-seed{}.pred".format(lm_model_name, seed)
    )

    if os.path.exists(file_name):
        return True
    else:
        return False


def load_embeddings(dataset_name, num_graphs, template, lm_model_name, seed):
    file_name = PurePath(
        project_root_path, "output", "prt_lms", dataset_name, template,
        "{}-seed{}.emb".format(lm_model_name, seed)
    )

    embeddings = np.memmap(
        filename=file_name,
        dtype=np.float16,
        mode='r',
        shape=(num_graphs, 768)
    )

    return torch.tensor(embeddings)


def remove_files_in_folder(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate through the files and remove them
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                # print("Removed: {}".format(file_path))
            else:
                print("Skipped: {} (not a file)".format(file_path))
        except Exception as e:
            print("Error removing {}: {}".format(file_path, e))


def clean_llm_cache(
        dataset_name, template, llm_model,
        operate_object="gnn", gnn_model=None, lm_model=None, seed=None,
        clean_response=True, clean_completion=True, clean_conversation=False, clean_chat_history=False
):
    lm_model = lm_model.replace("/", "-") if lm_model is not None else lm_model
    object_model = gnn_model if operate_object == "gnn" else lm_model

    if clean_response:
        _, folder_path = get_llm_file_path(
            dataset_name=dataset_name, template=template, llm_model=llm_model, data_format="response",
            object_model=object_model, seed=seed, sample_id=0, demo_test=False,
        )
        remove_files_in_folder(folder_path=folder_path)
    if clean_completion:
        _, folder_path = get_llm_file_path(
            dataset_name=dataset_name, template=template, llm_model=llm_model, data_format="chat_completion",
            object_model=object_model, seed=seed, sample_id=0, demo_test=False,
        )
        remove_files_in_folder(folder_path=folder_path)
    if clean_conversation:
        _, folder_path = get_llm_file_path(
            dataset_name=dataset_name, template=template, llm_model=llm_model, data_format="conversation",
            object_model=object_model, seed=seed, sample_id=0, demo_test=False,
        )
        remove_files_in_folder(folder_path=folder_path)
    if clean_chat_history:
        _, folder_path = get_llm_file_path(
            dataset_name=dataset_name, template=template, llm_model=llm_model, data_format="chat_history",
            object_model=object_model, seed=seed, sample_id=0, demo_test=False,
        )
        remove_files_in_folder(folder_path=folder_path)
