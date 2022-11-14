"""
Code to run the World Cup tournament, from group stages through the
knockout stages, to the final, and produce a winner.
"""

import os
import pandas as pd
import random
from typing import Optional, Union, List, Tuple
from .bpl_interface import WCPred
from .data_loader import (
    get_teams_data,
    get_fixture_data,
    get_fifa_rankings_data,
    get_results_data
)
from .utils import (
    find_group,
    sort_teams_by,
    predict_knockout_match,
    predict_group_match
)


class Group:
    def __init__(self, name: str, teams: List[str]):
        self.name = name
        self.teams = teams
        # "table" is a dictionary keyed by team name, with the points, gf,ga
        self.table = {}
        for t in self.teams:
            self.table[t] = {"points":0, "goals_for": 0, "goals_against": 0, "goal_difference": 0}
        # "standings" is a dictionary with keys "1st", "2nd", "3rd", "4th"
        # and values being the team names.
        self.standings = {"1st": None, "2nd": None, "3rd": None, "4th": None}
        # results is a list of played matches, where each entry is a
        # dict {"<team1_name>": <score>, "<team2_name>" <score>}
        self.results = []
        # order of criteria for deciding group order
        self.metrics = ["points","goal_difference","goals_for", "head-to-head", "random"]

    def play_match(self,
                   wc_pred: WCPred,
                   fixture: pd.Series,
                   seed = None) -> dict[str, int]:
        """
        Play a simulated group match.

        Parameters
        ==========
        fixture: row from a pandas DataFrame, has Team_1 and Team_2 columns

        Returns
        =======
        result: dict, where keys are the team names,
                and values are the goals for that team
        """
        goals_1, goals_2 = predict_group_match(wc_pred = wc_pred,
                                               team_1 = fixture.Team_1,
                                               team_2 = fixture.Team_2,
                                               seed = seed)
        result = {fixture.Team_1: goals_1, fixture.Team_2: goals_2}
        self.results.append(result)
        return result

    def play_all_matches(self,
                         wc_pred: WCPred,
                         fixture_df: pd.DataFrame,
                         seed: Optional[int] = None,
                         verbose: bool = False) -> None:
        """
        Given the full DataFrame full of fixtures, find the ones that correspond
        to this group, and use them to fill our list of results

        Parameters
        ==========
        fixture_df: pandas DataFrame of all fixtures, with Team_1 and Team_2 columns
        """
        for _, fixture in fixture_df.iterrows():
            if fixture.Team_1 in self.teams and fixture.Team_2 in self.teams:
                if verbose:
                    print(f"{fixture.Team_1} vs {fixture.Team_2}")
                result = self.play_match(wc_pred, fixture, seed)
                if verbose:
                    print(f"{fixture.Team_1} vs {fixture.Team_2}: "
                          + f"{result[fixture.Team_1]}-{result[fixture.Team_2]}")

    def calc_table(self) -> None:
        """
        Go through the results, and add points and goals to the table
        """
        # reset the table, in case we previously ran this function
        for t in self.teams:
            self.table[t]["points"] = 0
            self.table[t]["goals_for"] = 0
            self.table[t]["goals_against"] = 0
        for result in self.results:
            teams = list(result.keys())
            if result[teams[0]] > result[teams[1]]:
                self.table[teams[0]]["points"] += 3
            elif result[teams[0]] < result[teams[1]]:
                self.table[teams[1]]["points"] += 3
            else:
                self.table[teams[0]]["points"] += 1
                self.table[teams[1]]["points"] += 1
            for i,t in enumerate(teams):
                self.table[t]["goals_for"] += result[t]
                self.table[teams[(i+1)%2]]["goals_against"] += result[t]
        # loop through teams again to fill in goal difference
        for t in self.teams:
            self.table[t]["goal_difference"] = self.table[t]["goals_for"] - self.table[t]["goals_against"]

    def get_qualifiers(self) -> Tuple:
        """
        return the two teams that topped the group
        """
        if len(self.results) < 6:
            print(f"Group {self.name} not finished yet - only {len(self.results)} matches played")
            return
        self.calc_standings()

        return self.standings["1st"], self.standings["2nd"]

    def fill_standings_position(self,
                                team: str,
                                position: int,
                                verbose: bool = False) -> None:
        """
        Fill specified slot in our team standings.
        """
        if self.standings[position]:
            raise RuntimeError("Position {} is already filled!".format(position))
        if verbose:
            print("Putting {} in {}".format(team, position))
        self.standings[position] = team
        return

    def find_head_to_head_winner(self,
                                 team_A: str,
                                 team_B: str) -> Tuple[str, str]:
        team_1 = None
        team_2 = None
        for result in self.results:
            if set(result.keys()) == set([team_A, team_B]):
                if result[team_A] > result[team_B]:
                    team_1 = team_A
                    team_2 = team_B
                elif result[team_B] > result[team_A]:
                    team_1 = team_B
                    team_2 = team_A
                break
        return team_1, team_2

    def set_positions_using_metric(self,
                                   teams_to_sort: List[str],
                                   positions_to_fill: List[str],
                                   metric: str,
                                   verbose: bool = False) -> None:
        if len(teams_to_sort) != len(positions_to_fill):
            raise RuntimeError(f"Can't fill {len(positions_to_fill)} positions with {len(teams_to_sort)} teams")
        if verbose:
            print("Sorting {} using {} to fill positions {}".format(teams_to_sort, metric, positions_to_fill))
        # if random, just shuffle our list
        if metric == "random":
            random.shuffle(teams_to_sort)
            for i, pos in enumerate(positions_to_fill):
                self.fill_standings_position(teams_to_sort[i], pos)
            if verbose:
                print("randomly assigned {} teams".format(len(teams_to_sort)))
            return
        elif metric == "head-to-head":
            if len(teams_to_sort) > 2:
                print("Can't use head-to-head for more than 2 teams")
                # skip ahead to random
                self.set_positions_using_metric(teams_to_sort, positions_to_fill, "random")
            else:
                team_1, team_2 = self.find_head_to_head_winner(teams_to_sort[0],teams_to_sort[1])
                if team_1 and team_2: # not null if there was a winner
                    self.fill_standings_position(team_1, positions_to_fill[0])
                    self.fill_standings_position(team_2, positions_to_fill[1])
                else:
                    # go to random
                    self.set_positions_using_metric(teams_to_sort, positions_to_fill, "random")
            return
        # ok, otherwise we need to sort the table by the metric
        team_dict = {t: self.table[t] for t in teams_to_sort }
        team_scores = sort_teams_by(team_dict, metric) # list of dicts of teams
        team_list = [t["team"] for t in team_scores] # ordered list of teams
        # figure out the next metric, in case this one doesn't differentiate
        current_metric_index = self.metrics.index(metric)
        new_metric = self.metrics[current_metric_index+1]

        # OK, let's get sorting!! Start with two-team case
        if len(team_list) == 2:
            if team_scores[0][metric] > team_scores[1][metric]: # one team is better
                self.fill_standings_position(team_list[0],positions_to_fill[0])
                self.fill_standings_position(team_list[1],positions_to_fill[1])
                return
            else:
                # they are equal - call this func again with the next metric
                self.set_positions_using_metric(team_list, positions_to_fill, new_metric)
                return
        elif len(team_list) == 3:
            # 4 possible cases
            if team_scores[0][metric] > team_scores[1][metric] and \
               team_scores[1][metric] > team_scores[2][metric]: #1st > 2nd > 3rd
                self.fill_standings_position(team_list[0],positions_to_fill[0])
                self.fill_standings_position(team_list[1],positions_to_fill[1])
                self.fill_standings_position(team_list[2],positions_to_fill[2])
                return # we are done!
            elif team_scores[0][metric] > team_scores[1][metric] and \
                 team_scores[1][metric] == team_scores[2][metric]: #last two equal
                self.fill_standings_position(team_list[0],positions_to_fill[0])
                # call this func again with the last two, and the next metric
                self.set_positions_using_metric(team_list[1:], positions_to_fill[1:], new_metric)
                return
            elif team_scores[0][metric] == team_scores[1][metric] and \
                 team_scores[1][metric] > team_scores[2][metric]: #first two equal
                self.fill_standings_position(team_list[2], positions_to_fill[2])
                # call this func again with the first two, and the next metric
                self.set_positions_using_metric(team_list[:2], positions_to_fill[:2], new_metric)
            else: # all three teams equal - just move onto the next metric
                self.set_positions_using_metric(team_list, positions_to_fill, new_metric)
            return
        elif len(team_list) == 4: # 8 possible cases.
            if verbose:
                print("TEAM LIST", team_scores)
            if team_scores[0][metric] > team_scores[1][metric] and \
               team_scores[1][metric] > team_scores[2][metric] and \
               team_scores[2][metric] > team_scores[3][metric]: # case 1) all in order
                self.fill_standings_position(team_list[0],"1st")
                self.fill_standings_position(team_list[1],"2nd")
                self.fill_standings_position(team_list[2],"3rd")
                self.fill_standings_position(team_list[3],"4th")
                # we are done!
                return
            elif team_scores[0][metric] == team_scores[1][metric] and \
                 team_scores[1][metric] > team_scores[2][metric] and \
                 team_scores[2][metric] > team_scores[3][metric]: # case 2) first two equal
                self.fill_standings_position(team_list[2],"3rd")
                self.fill_standings_position(team_list[3],"4th")
                # call this func with the first two and the next metric
                self.set_positions_using_metric(team_list[:2], positions_to_fill[:2], new_metric)
            elif team_scores[0][metric] > team_scores[1][metric] and \
                 team_scores[1][metric] == team_scores[2][metric] and \
                 team_scores[2][metric] > team_scores[3][metric]: # case 3) middle two equal
                self.fill_standings_position(team_list[0],"1st")
                self.fill_standings_position(team_list[3],"4th")
                # call this func with the middle two and the next metric
                self.set_positions_using_metric(team_list[1:3], positions_to_fill[1:3], new_metric)
            elif team_scores[0][metric] > team_scores[1][metric] and \
                 team_scores[1][metric] > team_scores[2][metric] and \
                 team_scores[2][metric] == team_scores[3][metric]: # case 4) last two equal
                self.fill_standings_position(team_list[0],"1st")
                self.fill_standings_position(team_list[1],"2nd")
                # call this func with the last two and the next metric
                self.set_positions_using_metric(team_list[2:], positions_to_fill[2:], new_metric)
            elif team_scores[0][metric] == team_scores[1][metric] and \
                 team_scores[1][metric] == team_scores[2][metric] and \
                 team_scores[2][metric] > team_scores[3][metric]: # case 5) all equal except last
                self.fill_standings_position(team_list[3],"4th")
                # call this func with the first three and the next metric
                self.set_positions_using_metric(team_list[:3], positions_to_fill[:3], new_metric)
            elif team_scores[0][metric] > team_scores[1][metric] and \
                 team_scores[1][metric] == team_scores[2][metric] and \
                 team_scores[2][metric] == team_scores[3][metric]: # case 6) all equal except first
                self.fill_standings_position(team_list[0],"1st")
                # call this func with the last three and the next metric
                self.set_positions_using_metric(team_list[1:], positions_to_fill[1:], new_metric)
            elif team_scores[0][metric] == team_scores[1][metric] and \
                 team_scores[1][metric] > team_scores[2][metric] and \
                 team_scores[2][metric] == team_scores[3][metric]: # case 7) nightmare scenario!!
                # call func with first two and next metric
                self.set_positions_using_metric(team_list[:2], positions_to_fill[:2], new_metric)
                # call func with last two and next metric
                self.set_positions_using_metric(team_list[2:], positions_to_fill[2:], new_metric)
            else:  # case 8) all equal - carry on to next metric
                # call this func with the last three and the next metric
                self.set_positions_using_metric(team_list, positions_to_fill, new_metric)
            return

    def calc_standings(self) -> None:
        """
        sort the table, and try and assign positions in the standings
        """
        self.calc_table()
        # reset the standings table to start from scratch
        for k in self.standings.keys():
            self.standings[k] = None
        # now calculate the standings again
        self.set_positions_using_metric(self.teams, ["1st","2nd","3rd","4th"],"points")
        return

    def check_if_result_exists(self, team_1, team_2):
        """
        See if we already have a result for these two teams.
        Parameters
        ==========
        team_1, team_2: both str, team names, as in teams.csv
        Returns
        =======
        True if result already stored, False otherwise
        """
        for result in self.results:
            if set([team_1, team_2]) == set(result.keys()):
                return True
        return False

    def add_result(self, team_1, team_2, score_1, score_2):
        """
        Add a result for a group-stage match.
        Parameters
        ==========
        team_1, team_2: both str, team names, as in teams.csv
        score_1, score_2: both int, number of goals scored by each team.
        """
        if not self.check_if_result_exists(team_1, team_2):
            result = {team_1: score_1, team_2: score_2}
            self.results.append(result)
        return

    def __str__(self) -> str:
        max_team_name_length = 0
        for t in self.teams:
            if len(t) > max_team_name_length:
                max_team_name_length = len(t)

        output = f"Position |  Team{' '*(max_team_name_length-8)}| Points | GS |  GA \n"
        self.calc_standings()
        for k, v in self.standings.items():

            output += f"   {k}    {v}{' '*(max_team_name_length-len(v))}   {self.table[v]['points']}      {self.table[v]['goals_for']}     {self.table[v]['goals_against']} \n"
        return output


class Tournament:
    def __init__(self, year:str = "2022"):
        self.teams_df = get_teams_data(year)
        self.fixtures_df = get_fixture_data(year)
        self.group_names = list(set(self.teams_df["Group"].values))
        self.groups = {}
        for n in self.group_names:
            g = Group(n, list(self.teams_df[self.teams_df["Group"]==n].Team.values))
            self.groups[n] = g
        self.aliases = {}
        self.is_complete = False

    def add_result(self,
                   team_1: str,
                   team_2: str,
                   score_1: int,
                   score_2: int,
                   stage: str) -> None:
        """
        Enter a match result explicitly

        Parameters
        ==========
        team_1, team_2: both str, names of teams, as in teams.csv
        score_1, score_2: both int, scores of respective teams
        stage: str, must be "Group", "R16", "QF", "SF", "F"
        """
        # find the fixture
        for idx, row in self.fixtures_df.iterrows():
            if stage != row.Stage:
                continue
            if stage == "Group":
                if set([row.Team_1, row.Team_2]) == set([team_1,team_2]):
                    # find the group
                    group = find_group(team_1, self.teams_df)
                    self.groups[group].add_result(team_1, team_2, score_1, score_2)

    def play_group_stage(self,
                         wc_pred: WCPred,
                         seed: Optional[int] = None,
                         verbose: bool = False) -> None:
        for g in self.groups.values():
            g.play_all_matches(wc_pred = wc_pred,
                               fixture_df = self.fixtures_df,
                               seed = seed,
                               verbose = verbose)

    def play_knockout_stages(self,
                             wc_pred: WCPred,
                             seed: Optional[int] = None,
                             verbose: bool = False) -> None:
        """
        For the round of 16, assign the first and second place teams
        from each group to the aliases e.g. "A1", "B2"
        """
        for g in self.groups.values():
            if len(g.results) != 6:
                print(f" Group {g.name} has played {len(g.results)} matches")
            t1, t2 = g.get_qualifiers()
            self.aliases["1"+g.name] = t1
            self.aliases["2"+g.name] = t2
        for stage in ["R16","QF","SF","F"]:
            for _, f in self.fixtures_df.iterrows():
                if f.Stage == stage:
                    self.aliases[f.Team_1+f.Team_2] = predict_knockout_match(wc_pred = wc_pred,
                                                                             team_1 = self.aliases[f.Team_1],
                                                                             team_2 = self.aliases[f.Team_2],
                                                                             seed = seed)
                    if verbose:
                        print(f"{stage}: {f.Team_1} vs {f.Team_2}: Winner: {self.aliases[f.Team_1+f.Team_2]}" )
        for k,v in self.aliases.items():
            if len(k) == 32:
                self.winner = v
        print(f"====== WINNER: {self.winner} =======")
        self.is_complete = True


    def get_furthest_position_for_team(self, team_name):
        """
        Given a team name, see how far they got in the tournament.

        Parameters
        ==========
        team_name: str, one of the team names, as defined in teams.csv

        Returns
        =======
        "G", "R16", "QF", "SF", "RU", "W" depending on how far the team got.
        """
        if not self.is_complete:
            print("Tournament is not yet complete")
            return None
        if self.winner == team_name:
            return "W"
        elif team_name not in self.aliases.values():
            return "G"
        else:
            # the length of the 'alias' string, e.g. "1A2B" shows how far a team got
            key_length_lookup = {
                2: "R16",
                4: "QF",
                8: "SF",
                16: "RU"
            }
            # convert the aliases dict into a list, and sort by length of the key
            # (this will represent how far the team got - if we look in reverse order
            # of key length, we will find the latest stage a team got to first)
            alias_list = [(k,v) for k,v in self.aliases.items()]
            sorted_aliases = sorted(alias_list, key=lambda x: len(x[0]), reverse=True)
            for k,v in sorted_aliases:
                if v == team_name:
                    return key_length_lookup[len(k)]
            # we should never get to here
            raise RuntimeError(f"Unable to find team {team_name} in aliases table")
