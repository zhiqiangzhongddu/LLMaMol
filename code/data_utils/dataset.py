import pandas as pd
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from pathlib import PurePath

from code.data_utils.utils import load_llm_outputs, load_embeddings, project_root_path


class DatasetLoader():
    def __init__(
            self, name="ogbg-molbace", text='raw',
            feature='raw', lm_model_name='microsoft/deberta-base', llm_model_name='gpt-35-turbo-16k',
            seed=42
    ):
        self.name = name
        self.text = text
        self.feature = feature
        self.lm_model_name = lm_model_name.lower()
        self.llm_model_name = llm_model_name
        self.seed = seed

        self.dataset, self.text = self.load_data()

    def load_data(self):
        # Download and process data at root
        dataset = PygGraphPropPredDataset(
            name=self.name, root=PurePath(project_root_path, "data")
        )

        if self.text == '':
            text = None
        elif self.text == 'raw':
            df = pd.read_csv(
                filepath_or_buffer=PurePath(
                    project_root_path, "data",
                    "ogbg_{}".format(self.name.split("-")[1]),
                    "mapping", "mol.csv.gz"
                ),
                compression='gzip'
            )
            text = df.smiles.tolist()
        elif self.text in [
            'IF', 'IE', 'IP', 'FS-1', 'FS-2', 'FS-3',
            'IFD', 'IED', 'IPD', 'FSD-1', 'FSD-2', 'FSD-3'
            'IFC', 'IEC', 'IPC', 'FSC-1', 'FSC-2', 'FSC-3'
        ]:
            text = load_llm_outputs(
                dataset_name=self.name, template=self.text, data_format="response",
                llm_model=self.llm_model_name
            )
        elif self.text in [
            'SIF', 'SIE', 'SIP', 'SFS-1', 'SFS-2', 'SFS-3',
            'SIFD', 'SIED', 'SIPD', 'SFSD-1', 'SFSD-2', 'SFSD-3'
            'SIFC', 'SIEC', 'SIPC', 'SFSC-1', 'SFSC-2', 'SFSC-3'
        ]:
            # load SMILES string
            df = pd.read_csv(
                filepath_or_buffer=PurePath(
                    project_root_path, "data",
                    "ogbg_{}".format(self.name.split("-")[1]),
                    "mapping", "mol.csv.gz"
                ),
                compression='gzip'
            )
            smiles = df.smiles.tolist()
            # load LLM responses
            template = self.text[1:]
            response = load_llm_outputs(
                dataset_name=self.name, template=template, data_format="response",
                llm_model=self.llm_model_name
            )
            text = [smi + ". " + res for smi, res in zip(smiles, response)]
        else:
            raise ValueError("{} is an invalid text option".format(self.text))

        if self.feature in [
            'IF', 'IE', 'IP', 'FS-1', 'FS-2', 'FS-3',
            'IFD', 'IED', 'IPD', 'FSD-1', 'FSD-2', 'FSD-3',
            'IFC', 'IEC', 'IPC', 'FSC-1', 'FSC-2', 'FSC-3',
            'SIF', 'SIE', 'SIP', 'SFS-1', 'SFS-2', 'SFS-3',
            'SIFD', 'SIED', 'SIPD', 'SFSD-1', 'SFSD-2', 'SFSD-3',
            'SIFC', 'SIEC', 'SIPC', 'SFSC-1', 'SFSC-2', 'SFSC-3',
        ]:
            # load LLM response embeddings
            response_embedding = load_embeddings(
                dataset_name=self.name, num_graphs=dataset.y.size(0),
                template=self.feature,
                lm_model_name=self.lm_model_name,
                seed=self.seed
            )
            dataset._data.g_x = response_embedding
            dataset.slices["graph_x"] = torch.arange(len(dataset) + 1)
        elif self.feature in ['S']:
            # load SMILES string embeddings
            smiles_embedding = load_embeddings(
                dataset_name=self.name, num_graphs=dataset.y.size(0),
                template="raw",
                lm_model_name=self.lm_model_name,
                seed=self.seed
            )
            dataset._data.g_x = smiles_embedding
            dataset.slices["graph_x"] = torch.arange(len(dataset) + 1)
        elif self.feature in [
            'S-IF', 'S-IE', 'S-IP', 'S-FS-1', 'S-FS-2', 'S-FS-3',
            'S-IFD', 'S-IED', 'S-IPD', 'S-FSD-1', 'S-FSD-2', 'S-FSD-3',
            'S-IFC', 'S-IEC', 'S-IPC', 'S-FSC-1', 'S-FSC-2', 'S-FSC-3'
        ]:
            # load SMILES string embeddings
            smiles_embedding = load_embeddings(
                dataset_name=self.name, num_graphs=dataset.y.size(0),
                template="raw",
                lm_model_name=self.lm_model_name,
                seed=self.seed
            )
            # load LLM response embeddings
            template = self.feature[2:]
            response_embedding = load_embeddings(
                dataset_name=self.name, num_graphs=dataset.y.size(0),
                template=template,
                lm_model_name=self.lm_model_name,
                seed=self.seed
            )
            dataset._data.g_x = torch.concat([smiles_embedding, response_embedding], dim=1)
            dataset.slices["graph_x"] = torch.arange(len(dataset) + 1)
        elif self.feature == "raw":
            pass
        else:
            raise ValueError("{} is an invalid feature option".format(self.feature))

        return dataset, text


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
