import os
import pickle
import json
import tempfile
import subprocess
import pathlib
import argparse


def get_blogs(folder_path, source_name):
    repo_owner = "pytorch"
    repo_name = "pytorch.github.io"

    data = []

    with tempfile.TemporaryDirectory() as d:
        subprocess.check_call(
            f"git clone --depth 1 https://github.com/{repo_owner}/{repo_name}.git .",
            cwd=d,
            shell=True,
        )
        git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d).decode("utf-8").strip()
        )
        repo_path = pathlib.Path(d)
        markdown_files = list(repo_path.glob("_posts/*.md"))
        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                filename = markdown_file.parts[-1]
                title = os.path.splitext("-".join(filename.split("-")[3:]))[0]
                blog_url = f"https://pytorch.org/blog/{title}/"
                data.append({"text": f.read(), "metadata": {"source": blog_url}})

    output_folder = folder_path + "/" + source_name
    if not os.path.exists(output_folder):
        print(f"creating folder {output_folder}")
        os.makedirs(output_folder)

    print(f"saving data into {output_folder} as {source_name}.json")
    with open(f"{output_folder}/{source_name}.json", "w") as f:
        json.dump(data, f)

    pickle.dump(data, open(f"{output_folder}/{source_name}.pkl", "wb"))

    # return blogs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and save data files.")
    parser.add_argument(
        "--folder_path",
        type=str,
        default="knowledgebase",
        help="Path where the output files will be saved",
    )
    parser.add_argument("--source_name", type=str, default="file", help="Name of the output files")
    args = parser.parse_args()

    get_blogs(args.folder_path, args.source_name)
