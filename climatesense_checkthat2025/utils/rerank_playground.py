import logging
import traceback

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderNanoBEIREvaluator,
    CrossEncoderRerankingEvaluator,
)
from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import (
    BinaryCrossEntropyLoss,
)
from sentence_transformers.util import mine_hard_negatives

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
print("#A ------ ")


def main():
    print("#B ------ ")
    model_name = "answerdotai/ModernBERT-base"

    train_batch_size = 16
    num_epochs = 1
    num_hard_negatives = 5  # How many hard negatives should be mined for each question-answer pair

    print("#C ------ ")

    # 1a. Load a model to finetune with 1b. (Optional) model card data
    print("#1a start ------ ")
    model = CrossEncoder(
        model_name,
        model_card_data=CrossEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name="ModernBERT-base trained on GooAQ",
        ),
    )
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)
    print("#1a done ------ ")

    # 2a. Load the GooAQ dataset: https://huggingface.co/datasets/sentence-transformers/gooaq
    print("#2a start ------ ")
    logging.info("Read the gooaq training dataset")
    full_dataset = load_dataset("sentence-transformers/gooaq", split="train").select(range(100_000))
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)
    print("#2a done ------ ")

    # 2b. Modify our training dataset to include hard negatives using a very efficient embedding model
    print("#2b start ------ ")
    embedding_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
    hard_train_dataset = mine_hard_negatives(
        train_dataset,
        embedding_model,
        num_negatives=num_hard_negatives,  # How many negatives per question-answer pair
        margin=0,  # Similarity between query and negative samples should be x lower than query-positive similarity
        range_min=0,  # Skip the x most similar samples
        range_max=100,  # Consider only the x most similar samples
        sampling_strategy="top",  # Sample the top negatives from the range
        batch_size=4096,  # Use a batch size of 4096 for the embedding model
        output_format="labeled-pair",  # The output format is (query, passage, label), as required by BinaryCrossEntropyLoss  # noqa: E501
        use_faiss=False,  # ED: changed to False
    )
    logging.info(hard_train_dataset)
    print("#2b done ------ ")

    # 2c. (Optionally) Save the hard training dataset to disk
    # hard_train_dataset.save_to_disk("gooaq-hard-train")
    # Load again with:
    # hard_train_dataset = load_from_disk("gooaq-hard-train")

    # 3. Define our training loss.
    # pos_weight is recommended to be set as the ratio between positives to negatives, a.k.a. `num_hard_negatives`
    print("#3 start ------ ")
    loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(num_hard_negatives))
    print("#3 done ------ ")

    # 4a. Define evaluators. We use the CrossEncoderNanoBEIREvaluator, which is a light-weight evaluator for English reranking  # noqa: E501
    print("#4a start ------ ")
    CrossEncoderNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq"],
        batch_size=train_batch_size,
    )
    print("#4a done ------ ")

    # 4b. Define a reranking evaluator by mining hard negatives given query-answer pairs
    # We include the positive answer in the list of negatives, so the evaluator can use the performance of the
    # embedding model as a baseline.
    print("#4b start ------ ")
    hard_eval_dataset = mine_hard_negatives(
        eval_dataset,
        embedding_model,
        corpus=full_dataset["answer"],  # Use the full dataset as the corpus
        num_negatives=30,  # How many documents to rerank
        batch_size=4096,
        include_positives=True,
        output_format="n-tuple",
        use_faiss=False,  # ED: changed to False (# Using FAISS is recommended to keep memory usage low (pip install faiss-gpu or pip install faiss-cpu)  # noqa: E501
    )
    logging.info(hard_eval_dataset)
    reranking_evaluator = CrossEncoderRerankingEvaluator(
        samples=[
            {
                "query": sample["question"],
                "positive": [sample["answer"]],
                "documents": [sample[column_name] for column_name in hard_eval_dataset.column_names[2:]],
            }
            for sample in hard_eval_dataset
        ],
        batch_size=train_batch_size,
        name="gooaq-dev",
        always_rerank_positives=False,
    )
    print("#4b done ------ ")

    # 4c. Combine the evaluators & run the base model on them
    print("#4c start ------ ")
    evaluator = reranking_evaluator  # SequentialEvaluator([reranking_evaluator, nano_beir_evaluator])
    evaluator(model)
    print("#4c done ------ ")

    # 5. Define the training arguments
    print("#5 start ------ ")
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-{short_model_name}-gooaq-bce"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_gooaq-dev_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=4000,
        save_strategy="steps",
        save_steps=4000,
        save_total_limit=2,
        logging_steps=1000,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
    )
    print("#5 done ------ ")

    # 6. Create the trainer & start training
    print("#6 start ------ ")
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=hard_train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    print("#6 done ------ ")

    # 7. Evaluate the final model, useful to include these in the model card
    print("#7 start ------ ")
    evaluator(model)
    print("#7 done ------ ")

    # 8. Save the final model
    print("#8 start ------ ")
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)
    print("#8 done ------ ")

    # 9. (Optional) save the model to the Hugging Face Hub!
    # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
    print("#9 start ------ ")
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "  # noqa: E501
            f"`huggingface-cli login`, followed by loading the model using `model = CrossEncoder({final_output_dir!r})` "  # noqa: E501
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )
    print("#9 done ------ ")


if __name__ == "__main__":
    main()
