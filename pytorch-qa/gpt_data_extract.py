import re

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    T5Tokenizer,
)
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer


class SODataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
    ):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        max_length = 1024
        question = data_row["Question"]
        question = re.sub(r"\s+", " ", question)
        input_ids = self.tokenizer(
            question,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        attention_mask = input_ids["attention_mask"]
        input_ids = input_ids["input_ids"]
        answer = data_row["Answer"]
        answer = re.sub(r"\s+", " ", answer)
        labels = self.tokenizer(
            answer,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )["input_ids"]
        return dict(
            input_ids=input_ids.squeeze(0),
            attention_mask=attention_mask.squeeze(0),
            labels=labels.squeeze(0),
        )


class SODataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        batch_size: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage):

        df = pd.read_csv("PytorchDocs50.csv")
        df["Question"] = df.Question.str.lower()
        df["Answer"] = df.Answer.str.lower()

        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

        self.train_dataset = SODataset(train_df, self.tokenizer)

        self.val_dataset = SODataset(val_df, self.tokenizer)

        self.test_dataset = SODataset(test_df, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )


class SOModel(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer = tokenizer

    def forward(self, input_ids, labels=None, attention_mask=None):
        outputs = self.model(input_ids, labels=input_ids)
        return outputs[0]

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # labels = input_ids
        loss = self(input_ids, labels=labels, attention_mask=attention_mask)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # labels = input_ids
        loss = self(input_ids, labels=labels, attention_mask=attention_mask)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # labels = input_ids
        loss = self(input_ids, labels=labels, attention_mask=attention_mask)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.0001)
        return optimizer

    def extract_paragraphs(self, topic: str):
        input_ids = torch.tensor(self.tokenizer.encode(topic, return_tensors="pt"))
        output_ids = self.model.generate(input_ids, max_length=1024)
        paragraphs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return paragraphs


if __name__ == "__main__":
    data = pd.read_csv("PytorchDocs.csv")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    data_module = SODataModule(tokenizer, batch_size=1)
    data_module.setup("fit")

    model = SOModel(tokenizer=tokenizer)

    early_stop_callback = EarlyStopping(monitor="train_loss", patience=5, mode="min")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint-{epoch}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        save_last=True,
        mode="min",
    )
    logger = TensorBoardLogger("lightning_logs")
    callbacks = [checkpoint_callback, early_stop_callback]
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, max_epochs=1, gpus=1)

    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    # Extract the relevant paragraphs
    topic = "model saving"
    paragraphs = model.extract_paragraphs(topic)
    print(paragraphs)

    torch.save(model, "model.pth")
