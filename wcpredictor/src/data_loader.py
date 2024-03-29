import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def get_teams_data(year: str = "2022", womens: bool = False) -> pd.DataFrame:
    if year not in ["2014", "2018", "2022", "2023"]:
        raise RuntimeError(f"Unknown year {year}")
    current_dir = os.path.dirname(__file__)
    file_name = f"teams_{year}.csv" if not womens else f"teams_{year}_womens.csv"
    csv_path = os.path.join(current_dir, "..", "data", file_name)
    print(f"Loading teams data from {csv_path}")
    
    return pd.read_csv(csv_path)


def get_fixture_data(year: str = "2022", womens: bool = False) -> pd.DataFrame:
    if year not in ["2014", "2018", "2022", "2023"]:
        raise RuntimeError(f"Unknown year {year}")
    current_dir = os.path.dirname(__file__)
    file_name = f"fixtures_{year}.csv" if not womens else f"fixtures_{year}_womens.csv"
    csv_path = os.path.join(current_dir, "..", "data", file_name)
    print(f"Loading fixtures data from {csv_path}")
    
    return pd.read_csv(csv_path, parse_dates=["date"])


def get_confederations_data() -> pd.DataFrame:
    """
    Which teams belong to which federations
    """
    current_dir = os.path.dirname(__file__)
    filename = "confederations.csv"
    csv_path = os.path.join(current_dir, "..", "data", filename)
    print(f"Loading confederations data from {csv_path}")
    
    return pd.read_csv(csv_path)


def load_game_rankings(womens: bool = False) -> pd.DataFrame:
    print("Using FIFA videogame rankings")
    current_dir = os.path.dirname(__file__)
    filename = "fifa_game_rankings.csv" if not womens else f"fifa_game_rankings_womens.csv"
    csv_path = os.path.join(current_dir, "..", "data", filename)
    print(f"Loading FIFA game rankings from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # assign default values to teams not otherwise covered
    confederations = get_confederations_data()
    confed_dict = dict(zip(confederations.Team, confederations.Confederation))
    all_teams = confederations.Team.unique()
    current_teams = df.Team.unique()
    new_teams = list(set(all_teams) - set(current_teams))
    teams = []
    attacks = []
    midfields = []
    defences = []
    overalls = []
    for conf in set(confederations.Confederation):
        # define default value for Fifa ratings conditional on their confederation
        if conf in ["AFC", "CAF", "CONCACAF"]:
            default = 60
        elif conf in ["CONMEBOL", "UEFA"]:
            default = 65
        else:
            default = 50
        new_teams_in_conf = [team for team in new_teams if confed_dict[team] == conf]
        teams += new_teams_in_conf
        attacks += len(new_teams_in_conf) * [default]
        midfields += len(new_teams_in_conf) * [default]
        defences += len(new_teams_in_conf) * [default]
        overalls += len(new_teams_in_conf) * [default]
    new_df = pd.DataFrame(
        {
            "Team": teams,
            "Attack": attacks,
            "Midfield": midfields,
            "Defence": defences,
            "Overall": overalls,
        }
    )
    
    return pd.concat([df, new_df]).reset_index(drop=True)


def load_org_rankings(womens: bool = False) -> pd.DataFrame:
    print("Using FIFA organisation rankings")
    current_dir = os.path.dirname(__file__)
    filename = "fifa_rankings.csv" if not womens else "fifa_rankings_womens.csv"
    csv_path = os.path.join(current_dir, "..", "data", filename)
    print(f"Loading FIFA organisation ratings from {csv_path}")
    
    return pd.read_csv(csv_path)


def get_fifa_rankings_data(source: str = "game",
                           womens: bool = False) -> pd.DataFrame:
    """
    Get the FIFA rankings, either from FIFA (the organisation), if source == 'org'
    or from the FIFA video game (with default values for teams not in the game)
    if source == 'game', or combine both if source == 'both'
    """
    if source == "game":
        return load_game_rankings(womens=womens)
    elif source == "org":
        return load_org_rankings(womens=womens)
    elif source == "both":
        return pd.merge(
            load_game_rankings(womens=womens),
            load_org_rankings(womens=womens),
            how="inner",
            on="Team"
        )


def get_results_data(
    start_date: str = "2018-06-01",
    end_date: str = "2022-11-20",
    womens: bool = False,
    competitions: List[str] = None,
    rankings_source: str = "org",
    world_cup_weight: float = 1.0,
) -> Tuple[pd.DataFrame, dict[str, float]]:
    """
    filter the results dataframe by date and competition.
    Key for competitions:
    "W": world cup finals,
    "C1": top-level continental cup,
    "WQ": world cup qualifiers",
    "CQ": continental cup qualifiers"
    "C2": 2nd-tier continental, e.g. UEFA Nations League,
    "F": friendly/other.
    """
    if competitions is None:
        competitions = ["W", "C1", "WQ", "CQ", "C2", "F"]
    current_dir = os.path.dirname(__file__)
    results_file_path = "results.csv" if not womens else "results_womens.csv"
    csv_path = os.path.join(current_dir, "..", "data", results_file_path)
    print(f"Using results data from {csv_path}")
    results_df = pd.read_csv(csv_path, parse_dates=["date"])
    
    # get an index of what competition is in what category
    competition_file_path = "competition_index.json" if not womens else "competition_index_womens.json"
    json_path = os.path.join(current_dir, "..", "data", competition_file_path)
    print(f"Using competitions index file from {csv_path}")
    competitions_index = json.load(open(json_path))
    
    print(f"Filtering games for period: {start_date} to {end_date}")
    # filter by date
    results_df = results_df[
        (results_df.date >= start_date) & (results_df.date <= end_date)
    ]
    
    # replace any names that we have written differently elsewhere
    results_df = results_df.replace("United States", "USA")
    results_df = results_df.replace(
        "United States Virgin Islands", "USA Virgin Islands"
    )
    
    # filter matches with non-fifa recognised teams
    if rankings_source:
        rankings_df = get_fifa_rankings_data(source=rankings_source, womens=womens)
        fifa_teams = rankings_df.Team.values
        results_df = results_df[
            (results_df.home_team.isin(fifa_teams))
            & (results_df.away_team.isin(fifa_teams))
        ]
        
    # filter by competition
    print(f"Only using competitons from {competitions}")
    comp_filter = [competitions_index[comp] for comp in competitions]
    
    # flatten this nested list
    comp_filter = [comp for complist in comp_filter for comp in complist]
    results_df = results_df[results_df.tournament.isin(comp_filter)]
    
    # obtain time difference to the latest date in the dataframe
    # number of years back from end_date as a fraction
    end_date = pd.Timestamp(end_date)
    results_df["time_diff"] = (end_date - results_df.date) / pd.Timedelta(days=365)
    # compute game weights
    weight_dict = dict(
        zip(["F", "C2", "CQ", "WQ", "C1", "W"], np.linspace(1, world_cup_weight, 6))
    )
    reverse_competitions_index = {
        comp: key for key, value in competitions_index.items() for comp in value
    }
    comp_quality = [
        reverse_competitions_index[comp] for comp in results_df["tournament"]
    ]
    results_df["game_weight"] = [weight_dict[comp] for comp in comp_quality]

    results_df = results_df.reset_index(drop=True)
    return results_df, weight_dict


def get_wcresults_data(year: str) -> pd.DataFrame:
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "..", "data", f"wcresults_{year}.csv")
    return pd.read_csv(csv_path)


def get_alias_data(year: str, womens: bool = False) -> pd.DataFrame:
    current_dir = os.path.dirname(__file__)
    file_name = f"aliases_{year}.csv" if not womens else f"aliases_{year}_womens.csv"
    csv_path = os.path.join(current_dir, "..", "data", file_name)
    print(f"Loading in alias data from {csv_path}")
    
    return pd.read_csv(csv_path, index_col="alias")
