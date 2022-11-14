"""
Assorted functions to get the BPL model, and predict results.
"""
from typing import Optional, Union, List, Tuple

from .data_loader import (
    get_results_data,
    get_teams_data,
    get_fifa_rankings_data
)
from .bpl_interface import WCPred


def get_and_train_model(start_date: str = "2018-06-01",
                        end_date: str = "2022-11-20",
                        competitions: List[str] = ["W","C1","WQ","CQ","C2","F"],
                        rankings_source: str = "game",
                        epsilon: float = 0.0,
                        world_cup_weight: float = 1.0,
                        ) -> WCPred:
    """
    Use 'competitions' argument to specify which rows to include in training data.
    Key for competitions:
    "W": world cup finals,
    "C1": top-level continental cup,
    "WQ": world cup qualifiers",
    "CQ": continental cup qualifiers"
    "C2": 2nd-tier continental, e.g. UEFA Nations League,
    "F": friendly/other.

    rankings_source determines whether we use the FIFA video game rankings for prior values
    for the covariates ("game"), or use the FIFA organisation ones ("org"), or neither (None).
    """
    print("in get_and_train_model")
    results, weights_dict = get_results_data(start_date,
                                             end_date,
                                             competitions,
                                             world_cup_weight)
    print(f"Using {len(results)} rows in training data")
    # if we are getting results up to 2018, maybe we are simulating
    # the 2018 world cup?
    if "2018" in end_date:
        tournament_year = "2018"
    # and same logic for 2014
    if "2014" in end_date:
        tournament_year = "2014"
    else:
        tournament_year = "2022"
    teams = list(get_teams_data(tournament_year).Team)

    if rankings_source:
        ratings = get_fifa_rankings_data(rankings_source)
        wc_pred = WCPred(results=results,
                         ratings=ratings,
                         epsilon=epsilon,
                         world_cup_weight=world_cup_weight,
                         weights_dict=weights_dict)
    else:
        wc_pred = WCPred(results=results,
                         epsilon=epsilon,
                         world_cup_weight=world_cup_weight,
                         weights_dict=weights_dict)
    wc_pred.set_training_data()
    wc_pred.fit_model()
    return wc_pred


def find_group(team, teams_df):
    """
    Look in teams_df and find the group for a given team

    Parameters
    ==========
    team: str, team name, as given in teams.csv
    teams_df: Pandas dataframe

    Returns
    =======
    group_name: str, "A"-"H", or None if team not found
    """
    for idx, row in teams_df.iterrows():
        if row.Team == team:
            return row.Group
    print("Unable to find {} in teams.csv".format(team))
    return None


def sort_teams_by(table_dict, metric):
    """
    Given a dictionary in the same format as self.table (i.e. keyed by
    team name), return a list of dicts
    [{"team": <team_name>, "points": <points>, ...},{...}]
    in order of whatever metric is supplied

    Parameters
    ==========
    table_dict: dict, keyed by team name, in the same format as self.table

    Returns
    =======
    team_list: list of dicts, [{"team": "abc","points": x},...], ordered
               according to metric
    """
    if not metric in ["points","goal_difference","goals_for","goals_against"]:
        raise RuntimeError(f"unknown metric for sorting: {metric}")
    team_list = [{"team": k, **v} for k,v in table_dict.items()]
    team_list = sorted(team_list, key=lambda t: t[metric],reverse=True)
    return team_list


def predict_knockout_match(wc_pred: WCPred,
                           team_1: str,
                           team_2: str,
                           seed: Optional[int] = None) -> str:
    """
    Parameters
    ==========
    team_1, team_2: both str, names of two teams

    Returns:
    ========
    winning_team: str, one of team_1 or team_2
    """
    return wc_pred.get_fixture_probabilities(
        fixture_teams = [(team_1, team_2)],
        knockout = True,
        seed = seed
    )["simulated_outcome"][0]


def predict_group_match(wc_pred: WCPred,
                        team_1: str,
                        team_2: str,
                        seed: Optional[int] = None) -> Tuple[int, int]:
    """
    Parameters
    ==========
    team_1, team_2: both str, names of two teams

    Returns:
    ========
    score_1, score_2: both int, score for each team
    """
    return wc_pred.get_fixture_goal_probabilities(
        fixture_teams = [(team_1, team_2)],
        seed = seed
    )[1][0]


def get_difference_in_stages(stage_1:str, stage_2:str) -> int:
    """
    Give an integer value to the differences between two
    'stages' i.e. how far a team got in the tournament.
    This can be used to calculate a loss function, i.e. difference
    between predicted and actual results for past tournaments.

    Parameters
    ==========
    stage_1, stage_2: both str, can be "G","R16","QF","SF","RU","W"

    Returns
    =======
    diff: int, how far apart the two stages are.
    """
    stages = ["G","R16","QF","SF","RU","W"]
    if not stage_1 in stages and stage_2 in stages:
        raise RuntimeError(f"Unknown value for stage - must be in {stages}")
    return abs(stages.index(stage_1) - stages.index(stage_2))