# %%
import os
from git import Repo
import shutil
import requests
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd

ACCESS_TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")


def callback(user: str, repo: dict) -> dict:
    """Callback for multithreading

    Args:
        user (str): username
        repo (dict): repo request output from github api

    Returns:
        dict: A dict containing repo popularity and activity stats
    """
    repo_name = repo["name"]
    clone_url = repo["clone_url"]
    clone_repo(user, repo_name, clone_url)
    return get_repo_stats(user, repo_name)


def get_repo_stats(user: str, repo_name: str) -> dict:
    """Function for repo stats

    Args:
        user (str): username
        repo_name (str): reponame

    Returns:
        dict: A dict containing repo popularity and activity stats
    """
    api_url = "https://api.github.com"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    pulls_url = f"{api_url}/repos/{user}/{repo_name}/pulls"
    pulls_response = requests.get(pulls_url, headers=headers)
    pulls_count = len(pulls_response.json())

    stars_url = f"{api_url}/repos/{user}/{repo_name}/stargazers"
    stars_response = requests.get(stars_url, headers=headers)
    stars_count = len(stars_response.json())

    open_pulls_url = f"{pulls_url}?state=open"
    open_pulls_response = requests.get(open_pulls_url, headers=headers)
    open_pulls_count = len(open_pulls_response.json())

    closed_pulls_url = f"{pulls_url}?state=closed"
    closed_pulls_response = requests.get(closed_pulls_url, headers=headers)
    closed_pulls_count = len(closed_pulls_response.json())

    forks_url = f"{api_url}/repos/{user}/{repo_name}/forks"
    forks_response = requests.get(forks_url, headers=headers)
    forks_count = len(forks_response.json())

    return {
        "repo": repo_name,
        "pulls_count": pulls_count,
        "open_pulls_count": open_pulls_count,
        "stars_count": stars_count,
        "closed_pulls_count": closed_pulls_count,
        "forks_count": forks_count,
    }


def get_all_repos(user: str) -> pd.DataFrame:
    """This function clones the repo and gets the repo stats

    Args:
        user (str): username

    Returns:
        pd.DataFrame: A dataframe for repo stats
    """
    url = f"https://api.github.com/users/{user}/repos"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    repo_stats = Parallel(n_jobs=-1, prefer="threads")(
        delayed(callback)(user, repo) for repo in tqdm(data)
    )
    df = pd.DataFrame(data=repo_stats)
    df = df.sort_values(by=list(df.columns)[1:], ascending=False)
    return df


def clone_repo(user: str, repo_name: str, clone_url: str):
    """Function for cloning repo

    Args:
        user (str): User name
        repo_name (str): Repo name
        clone_url (str): Repo URL
    """
    local_path = f"./repos/{user}/{repo_name}"
    if not os.path.exists(local_path):
        Repo.clone_from(clone_url, local_path)


def main(user: str):
    """entrypoint function

    Args:
        user (str): A string for username
    """
    os.makedirs(f"repos/{user}", exist_ok=True)

    os.makedirs("dumped", exist_ok=True)
    df = get_all_repos(user)
    df.to_csv(f"dumped/{user}.repo_stats.csv", index=False)


if __name__ == "__main__":
    main(user="mythrex")

# %%
