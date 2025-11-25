import torch.nn as nn
import adapters
from transformers import AutoModel


class BertWithTwoHeads(nn.Module):
    """BERT model with adapter and two classification heads: task and gender."""

    def __init__(
        self,
        model_id: str,
        num_task_labels: int = 2,
        intermediate_dim: int = 256,
        adapter_name: str = "task_adapter"
    ):
        super().__init__()
        self.model_id = model_id
        self.bert = AutoModel.from_pretrained(model_id)

        # Initialize and add adapter
        adapters.init(self.bert)
        self.bert.add_adapter(adapter_name)
        self.bert.train_adapter(adapter_name)
        self.bert.set_active_adapters(adapter_name)

        hidden_size = self.bert.config.hidden_size
        self.intermediate = nn.Linear(hidden_size, intermediate_dim)
        self.activation = nn.ReLU()
        self.task_classifier = nn.Linear(intermediate_dim, num_task_labels)
        self.gender_classifier = nn.Linear(intermediate_dim, 2)

    def forward(self, input_ids, attention_mask, head="both"):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.activation(self.intermediate(cls_output))

        if head == "task":
            return self.task_classifier(x)
        elif head == "gender":
            return self.gender_classifier(x)
        else:
            return self.task_classifier(x), self.gender_classifier(x)


class BertWithTwoHeadsAge(nn.Module):
    """BERT model with adapter and two classification heads: task and age (5 groups)."""

    def __init__(
        self,
        model_id: str,
        num_task_labels: int = 2,
        num_age_groups: int = 5,
        intermediate_dim: int = 256,
        adapter_name: str = "task_adapter"
    ):
        super().__init__()
        self.model_id = model_id
        self.bert = AutoModel.from_pretrained(model_id)

        # Initialize and add adapter
        adapters.init(self.bert)
        self.bert.add_adapter(adapter_name)
        self.bert.train_adapter(adapter_name)
        self.bert.set_active_adapters(adapter_name)

        hidden_size = self.bert.config.hidden_size
        self.intermediate = nn.Linear(hidden_size, intermediate_dim)
        self.activation = nn.ReLU()
        self.task_classifier = nn.Linear(intermediate_dim, num_task_labels)
        self.age_classifier = nn.Linear(intermediate_dim, num_age_groups)

    def forward(self, input_ids, attention_mask, head="both"):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.activation(self.intermediate(cls_output))

        if head == "task":
            return self.task_classifier(x)
        elif head == "age":
            return self.age_classifier(x)
        else:
            return self.task_classifier(x), self.age_classifier(x)


class BertWithThreeHeads(nn.Module):
    """BERT model with adapter and three classification heads: task, gender, and age."""

    def __init__(
        self,
        model_id: str,
        num_task_labels: int = 2,
        num_gender_labels: int = 2,
        num_age_groups: int = 5,
        intermediate_dim: int = 256,
        adapter_name: str = "task_adapter"
    ):
        super().__init__()
        self.model_id = model_id
        self.bert = AutoModel.from_pretrained(model_id)

        # Initialize and add adapter
        adapters.init(self.bert)
        self.bert.add_adapter(adapter_name)
        self.bert.train_adapter(adapter_name)
        self.bert.set_active_adapters(adapter_name)

        hidden_size = self.bert.config.hidden_size
        self.intermediate = nn.Linear(hidden_size, intermediate_dim)
        self.activation = nn.ReLU()
        self.task_classifier = nn.Linear(intermediate_dim, num_task_labels)
        self.gender_classifier = nn.Linear(intermediate_dim, num_gender_labels)
        self.age_classifier = nn.Linear(intermediate_dim, num_age_groups)

    def forward(self, input_ids, attention_mask, head="all"):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.activation(self.intermediate(cls_output))

        if head == "task":
            return self.task_classifier(x)
        elif head == "gender":
            return self.gender_classifier(x)
        elif head == "age":
            return self.age_classifier(x)
        else:  # head == "all"
            return self.task_classifier(x), self.gender_classifier(x), self.age_classifier(x)
