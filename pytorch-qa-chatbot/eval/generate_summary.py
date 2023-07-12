import os
import glob
import pandas as pd


def get_matching_files(root_dir, pattern):
    result = []

    for root, dirs, files in os.walk(root_dir):
        matching_files = glob.glob(os.path.join(root, pattern))
        if matching_files:
            result += matching_files

    return result


def read_multiple_csv_files(csv_list):
    df_list = []
    for csv_path in csv_list:
        df = pd.read_csv(csv_path)
        model_name = csv_path.split("/")[-1]
        df["model_name"] = model_name
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df


def generate_summary(df, output_file_name="summary.xlsx"):
    # fetch the unique prompts
    unique_prompts = df["prompt"].unique()

    print("Prompts:\n", unique_prompts)

    # create a writer object
    # you may need to do pip install openpyxl if openpyxl is not installed
    writer = pd.ExcelWriter(output_file_name)

    # first lets write down the prompts into the sheets

    # Create Prompt Template sheet
    prompt_template_df = pd.DataFrame(
        {
            "Prompt Short Name": [f"prompt{i + 1}" for i in range(len(unique_prompts))],
            "Prompt Value": unique_prompts,
        }
    )
    prompt_template_df.to_excel(writer, sheet_name="Prompt Template", index=False)

    # Iterate over each prompt
    for i, prompt in enumerate(unique_prompts):
        print("\nProcessing prompt: ", prompt)
        prompt_short_name = f"prompt{i + 1}"
        prompt_df = df[df["prompt"] == prompt].copy()
        prompt_df["answer"] = prompt_df["answer"].str.strip()

        sheet_df = pd.DataFrame()

        unique_questions = prompt_df["question"].unique()
        for question in unique_questions:
            question_df = prompt_df[prompt_df["question"] == question]
            answers = question_df["answer"].values
            model_names = question_df["model_name"].values

            answer_dict = {model: answer for model, answer in zip(model_names, answers)}

            row = {"question": question, **answer_dict}
            sheet_df = pd.concat([sheet_df, pd.DataFrame([row])])

        print("Writing to sheet name: ", prompt_short_name)
        sheet_df.to_excel(writer, sheet_name=prompt_short_name, index=False)

    writer.close()


if __name__ == "__main__":
    # only_question
    only_question_csv_files_list = get_matching_files(
        root_dir="test_results", pattern="*only_question.csv"
    )
    print("Processing: ", len(only_question_csv_files_list))
    only_question_df = read_multiple_csv_files(only_question_csv_files_list)
    only_question_df = only_question_df[["question", "answer", "prompt", "model_name"]]
    generate_summary(df=only_question_df, output_file_name="only_question_summary.xlsx")

    # question_with_context
    question_with_context_csv_files_list = get_matching_files(
        root_dir="test_results", pattern="*question_with_context.csv"
    )
    print("Processing: ", len(question_with_context_csv_files_list))
    question_with_context_df = read_multiple_csv_files(question_with_context_csv_files_list)
    question_with_context_df = question_with_context_df[
        ["question", "answer", "prompt", "model_name"]
    ]
    generate_summary(
        df=question_with_context_df, output_file_name="question_with_context_summary.xlsx"
    )
