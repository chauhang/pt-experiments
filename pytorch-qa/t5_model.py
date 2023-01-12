import re

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

CLEANR = re.compile("<.*?>")


def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, "", raw_html)
    return cleantext


class SODataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        source_encoding = self.tokenizer(
            data_row["pt_title"],
            data_row["pt_body"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            data_row["pt_answer"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        labels = target_encoding["input_ids"]
        labels[labels == 0] = -100
        return dict(
            question=data_row["pt_title"],
            context=data_row["pt_body"],
            answer_text=data_row["pt_answer"],
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
        )


class SODataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 4,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self, stage):
        self.train_dataset = SODataset(
            self.train_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len
        )
        self.test_dataset = SODataset(
            self.test_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)


class SOModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.0001)
        return optimizer


if __name__ == "__main__":
    data = pd.read_csv("pt_question_answers.csv")

    data["pt_body"] = data["pt_body"].apply(lambda x: cleanhtml(x))

    MODEL_NAME = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    train_df, val_df = train_test_split(data, test_size=0.05)

    BATCH_SIZE = 4
    data_module = SODataModule(train_df, val_df, tokenizer, batch_size=BATCH_SIZE)
    data_module.setup("fit")

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

    model = SOModel()

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint-{epoch}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        save_last=True,
        mode="min",
    )
    logger = TensorBoardLogger("training-logs", name="so-qa")
    trainer = pl.Trainer(
        logger=logger, callbacks=[checkpoint_callback, early_stop_callback], max_epochs=100, gpus=4
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)
