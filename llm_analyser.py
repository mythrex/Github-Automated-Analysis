import json
import random
import os
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Optional
from joblib import Parallel, delayed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chat_models.openai import ChatOpenAI

load_dotenv()

EXTENSION_SET = {
    ".py",  # Python
    ".java",  # Java
    ".js",  # JavaScript
    ".cpp",  # C++
    ".c",  # C
    ".html",  # HTML
    ".css",  # CSS
    ".php",  # PHP
    ".rb",  # Ruby
    ".swift",  # Swift
    ".go",  # Go
    ".ts",  # TypeScript
    ".sh",  # Shell script
    ".pl",  # Perl
    ".r",  # R
    ".scala",  # Scala
    ".lua",  # Lua
    ".md",  # Markdown
    ".json",  # JSON
    ".xml",  # XML
    ".yaml",  # YAML
    ".sql",  # SQL
    ".h",  # Header file
    ".hpp",  # C++ header file
    ".cs",  # C#
    ".vb",  # Visual Basic
    ".asm",  # Assembly
    ".dockerfile",  # Dockerfile
    ".yml",  # YAML (alternative extension)
    ".kt",  # Kotlin
    ".jl",  # Julia
    ".groovy",  # Groovy
    ".pl",  # Prolog
    ".ps1",  # PowerShell
    ".tex",  # LaTeX
    ".matlab",  # MATLAB
    ".m",  # MATLAB (alternative extension)
    ".dart",  # Dart
    ".bash",  # Bash script
    ".jsx",  # JSX (JavaScript extension)
    ".tsx",  # TSX (TypeScript extension)
    ".cfg",  # Configuration file
    ".ini",  # INI file
    ".md",  # Markdown file
}


class LLMCodeAnalyser:
    map_template_string = """Given the following code information, for the following 
    and access the code quality in terms of following points.
    Answer the following with a score 0-10.
        1. Cyclomatic complexity
        2. Nesting depth
        3. Code duplication
        4. Code coupling
        5. Code readability
        6. Code maintanibility
        7. Proper documentation
        8. Proper function doc strings
        9. Proper maintained readme (1=yes or 0=no).

    Code:
        {text}
    """

    reduce_template_string = """Given the information about the code quality, 
    Mean Aggregate the results below and ensure the following keys in the output 
    python dictionary
        1. Cyclomatic complexity
        2. Nesting depth
        3. Code duplication
        4. Code coupling
        5. Code readability
        6. Code maintanibility
        7. Proper documentation
        8. Proper function doc strings
        9. Proper maintained readme.
        
        {text}
        Answer:
    """

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        """Initializer

        Args:
            model (str, optional): Chat Model Type. Defaults to "gpt-3.5-turbo".
        """
        self.llm = ChatOpenAI(model=model)

    def _convert_notebook_to_code_string(self, notebook_path: str) -> str:
        """Converts notebook to a str. It uses code as it is and
        markdown as comments. No output is used.

        Args:
            notebook_path (str): Notebook path

        Returns:
            str: This is represented as python
            file with markdown as comments.
        """
        # Open the Jupyter Notebook file
        with open(notebook_path, "r") as f:
            notebook_content = json.load(f)

        code_cells = []
        for cell in notebook_content["cells"]:
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                code_cells.append(code)
            elif cell["cell_type"] == "mardown":
                comment = "".join(cell["source"])
                code_cells.append("'''" + comment + "'''")

        code_string = "\n".join(code_cells[:25])
        return code_string

    def _load_text_files(self, file: str) -> str:
        """Opens a code/notebook file and convert them into a string

        Args:
            file (str): filename

        Returns:
            _type_: _description_
        """
        # handle notebooks
        extension = "." + file.split(".")[-1]
        if extension == ".ipynb":
            code = self._convert_notebook_to_code_string(file)
            code = f"------------------------------------\n{file}\n{code}"
            return code
        elif extension in EXTENSION_SET:
            try:
                f = open(file, "r")
                code = f.read()
                code = "\n".join(code.split("\n")[:100])
                code = f"------------------------------------\n{file}\n{code}"
                return code
            except:
                return None

    def _get_file_paths(self, directory: str) -> List[str]:
        """Gets all the files into a directory

        Args:
            directory (str): Path to repo

        Returns:
            List[str]: All the files in a repo
        """
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        return file_paths

    def get_code(self, repo_path: str, sampling_rate: float = 0.25) -> str:
        """Concatenate all the files in repo. For chatgpt

        Args:
            repo_path (str): Repo Path
            sampling_rate (float, optional): Cheap approximation.
            Instead for taking all files we only take subsample. Defaults to 0.25.

        Returns:
            str: Codebase as single str
        """
        files = self._get_file_paths(repo_path)
        files = sorted(files, key=lambda x: len(x.split("/")))
        min_level = min(map(lambda x: len(x.split("/")), files))
        # ? Get all files at level 0 i.e README.md etc
        files_at_level0 = list(filter(lambda x: len(x.split("/")) == min_level, files))
        # ? instead of all files only subsample of files are
        # ? used for approximation
        random_sampled_files = random.sample(
            list(
                filter(lambda x: (len(x.split("/")) != min_level) and "." in x, files)
            ),
            k=min(3, int(sampling_rate * len(files))),
        )
        files = files_at_level0 + random_sampled_files
        files = list(
            filter(
                lambda x: x is not None and len(x) > 0,
                map(lambda x: self._load_text_files(x), files),
            )
        )
        files = "".join(files)
        return files

    def analyse_repo_gpt(self, repo_path: str) -> dict:
        """Send request to chat gpt in map reduce fashion

        Args:
            repo_path (str): _description_

        Returns:
            dict: A dictionary of features that llm access code on.
        """
        prompt = PromptTemplate(
            template=self.map_template_string, input_variables=["text"]
        )
        reduce_prompt = PromptTemplate(
            input_variables=["text"],
            template=self.reduce_template_string,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3500,
            chunk_overlap=10,
            length_function=len,
        )

        codebase = self.get_code(repo_path, sampling_rate=0.1)
        texts = text_splitter.split_text(codebase)
        docs = [Document(page_content=t) for t in texts]

        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            return_intermediate_steps=False,
            map_prompt=prompt,
            combine_prompt=reduce_prompt,
        )
        print("No of Docs: ", len(docs))
        if docs:
            output = chain({"input_documents": docs[:20]}, return_only_outputs=True)
        else:
            return {}
        output = output.get("output_text")
        if output:
            num_lines = len(codebase.split("\n"))
            try:
                output = json.loads(output)
            except:
                return {}
            output["Num Lines"] = num_lines
        return output


def callback(repo_path, code_analyser: LLMCodeAnalyser) -> dict:
    """Callback for Parallel job"""
    output = code_analyser.analyse_repo_gpt(repo_path)
    output["Repo"] = repo_path.split("/")[-1]
    return output


def to_float(x: str) -> Optional[float]:
    """Func to conver str to float for results

    Args:
        x (str): element

    Returns:
        float: Float value or NaN
    """
    try:
        return float(x)
    except:
        return np.NaN


def rerank(code_analysis: pd.DataFrame, user: str) -> pd.DataFrame:
    """Rerank logic for final ranking of repos

    Args:
        code_analysis (pd.DataFrame): Dataframe of repos with their
        code quality/popularity features
        user (str): Github Username

    Returns:
        pd.DataFrame: _description_
    """
    repo_stats = pd.read_csv(f"dumped/{user}.repo_stats.csv")

    code_analysis = code_analysis.merge(repo_stats, on="repo", how="inner")
    cols = list(code_analysis.columns)
    cols.remove("repo")

    for c in cols:
        code_analysis[c] = code_analysis[c].apply(to_float)
    code_analysis = code_analysis.fillna(0)
    code_analysis = code_analysis[["repo"] + cols]
    code_analysis["code duplication"] *= -1
    cols_to_check = [
        "cyclomatic complexity",
        "nesting depth",
        "code duplication",
        "code coupling",
        "code readability",
        "code maintainability",
        "proper documentation",
        "proper function doc strings",
        "proper maintained readme",
    ]
    # ? calculate score
    code_analysis["score"] = (
        code_analysis[cols_to_check].mean(1)
        + code_analysis["num lines"] / code_analysis["num lines"].max()
    )
    code_analysis = code_analysis.sort_values(
        by=[
            "score",
            "pulls_count",
            "open_pulls_count",
            "stars_count",
            "closed_pulls_count",
            "forks_count",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    )
    return code_analysis


def run(user: str):
    """Entrypoint

    Args:
        user (str): Github username
    """
    code_analyser = LLMCodeAnalyser()
    folders = glob.glob(f"repos/{user}/*")

    if not os.path.exists(f"dumped/{user}.code_analysis.csv"):
        outputs = Parallel(n_jobs=-1, prefer="threads")(
            delayed(callback)(repo_path, code_analyser) for repo_path in tqdm(folders)
        )
        result = pd.DataFrame(
            data=[{k.lower(): v for k, v in x.items()} for x in outputs]
        )
        os.makedirs("dumped", exist_ok=True)
        result.to_csv(f"dumped/{user}.code_analysis.csv", index=False)
        result = rerank(result, user)
        result.to_csv(f"dumped/{user}.code_analysis.ranked.csv", index=False)


if __name__ == "__main__":
    run("mythrex")
