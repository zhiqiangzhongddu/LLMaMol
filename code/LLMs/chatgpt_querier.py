import time

from code.data_utils.utils import (check_cache_response,
                                   save_chatcompletion, save_response,
                                   load_chatcompletion, load_response,)


def query_chatgpt_gnn_preds_batch(
        client, llm_model, dataset_name, template, gnn_model, seed,
        batch_message_list, batch_start_id, rpm_limit, demo_test
):

    batch_chat_completion_list = []
    batch_response_list = []

    for send_id, send_message in enumerate(batch_message_list):
        flag = check_cache_response(
            dataset_name=dataset_name, template=template,
            gnn_model=gnn_model, seed=seed, llm_model=llm_model,
            sample_id=batch_start_id + send_id, demo_test=demo_test
        )
        if flag:
            chat_completion = load_chatcompletion(
                dataset_name=dataset_name,
                gnn_model=gnn_model, seed=seed,
                template=template, llm_model=llm_model,
                sample_id=batch_start_id + send_id, demo_test=demo_test
            )
            response = load_response(
                dataset_name=dataset_name,
                gnn_model=gnn_model, seed=seed,
                template=template, llm_model=llm_model,
                sample_id=batch_start_id + send_id, demo_test=demo_test
            )
            batch_chat_completion_list.append(chat_completion)
            batch_response_list.append(response)
        else:
            chat_completion = client.chat.completions.create(
                model=llm_model, messages=send_message,
                # temperature=0.6,
                # top_p=0.9,
            )
            response = chat_completion.choices[0].message.content
            batch_chat_completion_list.append(chat_completion)
            batch_response_list.append(response)

            # Save response of each request, in case any errors stop the execution
            save_chatcompletion(
                dataset_name=dataset_name, chat_completion=chat_completion,
                gnn_model=gnn_model, seed=seed,
                template=template, llm_model=llm_model,
                sample_id=batch_start_id + send_id, demo_test=demo_test
            )
            # Save response of each request, in case any errors stop the execution
            save_response(
                dataset_name=dataset_name, list_response=response,
                gnn_model=gnn_model, seed=seed,
                template=template, llm_model=llm_model,
                sample_id=batch_start_id + send_id, demo_test=demo_test
            )
            # print(batch_start_id + send_id)

            # Rate limiting to stay within RPM limit
            time.sleep(60 / rpm_limit)

    return batch_chat_completion_list, batch_response_list


def query_chatgpt_batch(
        client, llm_model, dataset_name, template,
        batch_message_list, batch_start_id, rpm_limit
):

    batch_chat_completion_list = []
    batch_response_list = []

    for send_id, send_message in enumerate(batch_message_list):
        flag = check_cache_response(
            dataset_name=dataset_name, template=template,
            llm_model=llm_model, sample_id=batch_start_id + send_id
        )
        if flag:
            chat_completion = load_chatcompletion(
                dataset_name=dataset_name,
                template=template, llm_model=llm_model,
                sample_id=batch_start_id + send_id
            )
            response = load_response(
                dataset_name=dataset_name,
                template=template, llm_model=llm_model,
                sample_id=batch_start_id + send_id
            )
            batch_chat_completion_list.append(chat_completion)
            batch_response_list.append(response)
        else:
            chat_completion = client.chat.completions.create(
                model=llm_model, messages=send_message,
                # temperature=0.6,
                # top_p=0.9,
            )
            response = chat_completion.choices[0].message.content
            batch_chat_completion_list.append(chat_completion)
            batch_response_list.append(response)

            # Save response of each request, in case any errors stop the execution
            save_chatcompletion(
                dataset_name=dataset_name, chat_completion=chat_completion,
                template=template, llm_model=llm_model,
                sample_id=batch_start_id + send_id
            )
            # Save response of each request, in case any errors stop the execution
            save_response(
                dataset_name=dataset_name, list_response=response,
                template=template, llm_model=llm_model,
                sample_id=batch_start_id + send_id
            )
            # print(batch_start_id + send_id)

            # Rate limiting to stay within RPM limit
            time.sleep(60 / rpm_limit)

    return batch_chat_completion_list, batch_response_list
