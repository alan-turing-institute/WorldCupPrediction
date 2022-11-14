"""
Interface to the NumPyro team model in bpl-next:
https://github.com/anguswilliams91/bpl-next
"""
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from bpl import NeutralDixonColesMatchPredictor, NeutralDixonColesMatchPredictorWC
from jax.numpy import DeviceArray

from wcpredictor.src.data_loader import get_confederations_data

class WCPred:
    def __init__(
        self,
        results: pd.DataFrame,
        fixtures: Optional[pd.DataFrame] = None,
        ratings: Optional[pd.DataFrame] = None,
        teams: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        epsilon: float = 0.0,
        world_cup_weight: float = 1.0,
        weights_dict: Optional[dict[str, float]] = None,
    ):
        self.results = results
        self.fixtures = fixtures
        self.ratings = ratings
        if teams is None:
            # teams is just every team that has played in results
            self.teams = list(
                set(self.results["home_team"]) | set(self.results["away_team"])
            )
        else:
            # disregard any games which involves a team that isnt in teams
            self.teams = teams
            self.results = self.results[
                self.results.home_team.isin(teams) & self.results.away_team.isin(teams)
            ]
        if years is not None:
            self.results = self.results[self.results.date.dt.year.isin(years)]
        confed = get_confederations_data()
        self.confed_dict = dict(zip(confed["Team"], confed["Confederation"]))
        self.training_data = None
        self.epsilon = epsilon
        self.world_cup_weight = world_cup_weight
        self.weights_dict = weights_dict
        self.model = None

    def get_result_dict(self) -> dict[str, np.array]:
        """
        put results into dictionary to train model
        """
        return {
            "home_team": np.array(self.results.home_team),
            "away_team": np.array(self.results.away_team),
            "home_conf": np.array([self.confed_dict[team]
                                   for team in self.results.home_team]),
            "away_conf": np.array([self.confed_dict[team]
                                   for team in self.results.away_team]),
            "home_goals": np.array(self.results.home_score),
            "away_goals": np.array(self.results.away_score),
            "neutral_venue": np.array(self.results.neutral),
            "time_diff": np.array(self.results.time_diff),
            "game_weights": np.array(self.results.game_weight),
        }

    def get_ratings_dict(self) -> dict:
        """Create a dataframe containing the fifa team ratings."""
        ratings = self.ratings[self.ratings.Team.isin(self.teams)]
        ratings_dict = {
            row.Team: np.array(row.drop("Team").values.astype(float))
            for _, row in ratings.iterrows()
        }
        if len(ratings_dict) != len(self.teams):
            raise ValueError(
                f"Must have FIFA ratings and results for all teams. {len(ratings_dict)} "
                + f"teams with FIFA ratings but {len(self.teams)} teams with results."
            )
        return ratings_dict

    def check_teams_in_ratings(self) -> bool:
        if self.ratings is None:
            return True
        teams = pd.Series(self.teams)
        print("---len(teams[~teams.isin(self.ratings.Team)])")
        if len(teams[~teams.isin(self.ratings.Team)]) > 0:
            raise ValueError(
                "Must have FIFA ratings and results for all teams. "
                + f"There are {len(teams[~teams.isin(self.ratings.Team)])} teams with no FIFA ratings:\n\n"
                + ", ".join(teams[~teams.isin(self.ratings.Team)])
            )
        return True

    def set_training_data(self) -> None:
        """Get training data for team model including FIFA ratings as covariates.
        Data returned is for all matches up to specified gameweek and season.
        """
        print("[MODEL FITTING] Setting training data for the model")
        training_data = self.get_result_dict()
        if self.ratings is not None:
            if self.ratings.isna().any().any():
                raise ValueError(
                    "There are some NaN values in ratings, please fix or remove"
                )
            if self.check_teams_in_ratings:
                training_data["team_covariates"] = self.get_ratings_dict()
        self.training_data = training_data

    def fit_model(self, **fit_args) -> None:
        """
        Get the team-level stan model, which can give probabilities of
        each potential scoreline in a given fixture.
        """
        if self.training_data is None:
            self.set_training_data()
        print("[MODEL FITTING] Fitting the model")
        self.model = NeutralDixonColesMatchPredictorWC().fit(
            self.training_data, self.epsilon, **fit_args
        )

    def get_fixture_teams(self) -> List[Tuple[str, str]]:
        if self.fixtures is None:
            raise ValueError("No fixtures have been set into self.fixtures")
        return [(row.Team_1, row.Team_2) for index, row in self.fixtures.iterrows()]

    def get_fixture_probabilities(
        self,
        fixture_teams: Optional[List[Tuple[str, str]]] = None,
        knockout: bool = False,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Returns probabilities and predictions for all fixtures in a given gameweek and season, as a data
        frame with a row for each fixture and columns being home_team,
        away_team, home_win_probability, draw_probability, away_win_probability.
        """
        if seed is not None:
            np.random.seed(seed)
        if self.model is None:
            self.fit_model()
        # if fixture_teams is not passed, we just predict for all games in self.fixtures by default
        if fixture_teams is None:
            fixture_teams = self.get_fixture_teams()
        Team_1, Team_2 = zip(*fixture_teams)
        Team_1_conference = [self.confed_dict[team] for team in Team_1]
        Team_2_conference = [self.confed_dict[team] for team in Team_2]
        venue = np.ones(len(Team_1))
        p = self.model.predict_outcome_proba(Team_1, Team_2, Team_1_conference, Team_2_conference, venue)
        # predict outcome of the game
        simulated_outcome = []
        for i in range(len(fixture_teams)):
            if knockout:
                prob = np.array([p["home_win"][i], p["away_win"][i]])
                simulated_outcome.append(
                    np.random.choice(a=[Team_1[i], Team_2[i]], p=prob / prob.sum())
                )
            else:
                prob = np.array([p["home_win"][i], p["draw"][i], p["away_win"][i]])
                simulated_outcome.append(
                    np.random.choice(
                        a=[Team_1[i], "Draw", Team_2[i]], p=prob / prob.sum()
                    )
                )
        return pd.DataFrame(
            {
                "Team_1": Team_1,
                "Team_2": Team_2,
                "Team_1_win_probability": p["home_win"],
                "draw_probability": p["draw"],
                "Team_2_win_probability": p["away_win"],
                "simulated_outcome": simulated_outcome,
            }
        )

    def get_fixture_goal_probabilities(
        self,
        fixture_teams: Optional[List[Tuple[str, str]]] = None,
        seed: Optional[int] = None,
        max_goals: int = 10,
    ) -> Tuple[dict[int, dict[str, DeviceArray]], List[Tuple[int, int]]]:
        """
        Get the probability that each team in a fixture scores any number of goals up to max_goals,
        and prediction of goals scored.
        """
        if seed is not None:
            np.random.seed(seed)
        if self.model is None:
            self.fit_model()
        goals = np.arange(0, max_goals + 1)
        probs = {}
        simulated_scores = []
        # if fixture_teams is not passed, we just predict for all games in self.fixtures by default
        if fixture_teams is None:
            fixture_teams = self.get_fixture_teams()
        Team_1, Team_2 = zip(*fixture_teams)
        Team_1_conference = [self.confed_dict[team] for team in Team_1]
        Team_2_conference = [self.confed_dict[team] for team in Team_2]
        venue = np.ones(len(Team_1))
        for i in range(len(fixture_teams)):
            home_team_goal_prob = self.model.predict_score_n_proba(
                goals, Team_1[i], Team_2[i], Team_1_conference[i], Team_2_conference[i], home=False, neutral_venue=venue[i]
            )
            away_team_goal_prob = self.model.predict_score_n_proba(
                goals, Team_2[i], Team_1[i], Team_2_conference[i], Team_1_conference[i], home=False, neutral_venue=venue[i]
            )
            probs[i] = {
                Team_1[i]: {g: p for g, p in zip(goals, home_team_goal_prob)},
                Team_2[i]: {g: p for g, p in zip(goals, away_team_goal_prob)},
            }
            # predict number of goals scored by Team 1
            T1_prob = np.array(list(probs[i][Team_1[i]].values()))
            T1_goals = np.random.choice(
                a=range(max_goals + 1), p=T1_prob / T1_prob.sum()
            )
            # predict number of goals scored by Team 2
            T2_prob = np.array(list(probs[i][Team_2[i]].values()))
            T2_goals = np.random.choice(
                a=range(max_goals + 1), p=T2_prob / T2_prob.sum()
            )
            simulated_scores.append((T1_goals, T2_goals))
        return probs, simulated_scores
