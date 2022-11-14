import pandas as pd
import random
from pytest_mock import mocker

from wcpredictor import (
    find_group,
    get_teams_data,
    get_fixture_data,
    predict_group_match,
    predict_knockout_match,
    sort_teams_by,
    Tournament,
    Group,
)

def test_get_teams_df():
    teams_df = get_teams_data()
    assert isinstance(teams_df, pd.DataFrame)
    assert(len(teams_df) > 0)

def test_get_fixture_df():
    fix_df = get_fixture_data()
    assert isinstance(fix_df, pd.DataFrame)
    assert(len(fix_df) > 0)


def test_find_group():
    teams_df = get_teams_data()
    assert find_group("Qatar", teams_df) == "A"
    assert find_group("England", teams_df) == "B"


def test_find_group_bad_team():
    teams_df = get_teams_data()
    assert find_group("Italy",teams_df) == None


def test_sort_by_points():
    test_dict = {
        "Qatar": {"points": 9, "goal_difference": 5,"goals_for": 6, "goals_against":1},
        "Senegal": {"points": 6, "goal_difference": 2,"goals_for": 4, "goals_against":2},
        "Ecuador": {"points": 3, "goal_difference": -2,"goals_for": 2, "goals_against":4},
        "Netherlands": {"points": 0, "goal_difference": -5,"goals_for": 1, "goals_against":6}
    }
    sorted_list = sort_teams_by(test_dict, "points")
    expected_team_order = ["Qatar","Senegal","Ecuador","Netherlands"]
    assert [t["team"] for t in sorted_list] == expected_team_order


def test_sort_by_goal_difference():
    test_dict = {
        "Qatar": {"points": 4, "goal_difference": -3,"goals_for": 3, "goals_against":6},
        "Senegal": {"points": 4, "goal_difference": 3,"goals_for": 6, "goals_against":3},
        "Ecuador": {"points": 4, "goal_difference": 2,"goals_for": 4, "goals_against":2},
        "Netherlands": {"points": 4, "goal_difference": -2,"goals_for": 2, "goals_against":4}
    }
    sorted_list = sort_teams_by(test_dict, "goal_difference")
    expected_team_order = ["Senegal","Ecuador","Netherlands","Qatar"]
    assert [t["team"] for t in sorted_list] == expected_team_order


def test_create_tournament():
    t = Tournament()
    assert isinstance(t.teams_df, pd.DataFrame)
    assert isinstance(t.fixtures_df, pd.DataFrame)
    assert len(t.groups) == 8
    assert set(t.groups.keys()) == set(["A","B","C","D","E","F","G","H"])


def test_create_group():
    g=Group("A",["Qatar","Ecuador","Senegal","Netherlands"])
    assert isinstance(g,Group)
    assert isinstance(g.results, list)
    assert isinstance(g.table, dict)


def test_table():
    g=Group("A",["Qatar","Ecuador","Senegal","Netherlands"])
    g.results.append({"Ecuador": 3, "Qatar": 1})
    g.calc_table()
    assert g.table["Ecuador"] == {"points": 3, "goals_for": 3, "goals_against": 1, "goal_difference": 2}
    assert g.table["Qatar"] == {"points": 0, "goals_for": 1, "goals_against": 3, "goal_difference": -2}
    g.results.append({"Netherlands": 2, "Qatar": 2})
    g.calc_table()
    assert g.table["Qatar"] == {"points": 1, "goals_for": 3, "goals_against": 5, "goal_difference": -2}
    assert g.table["Netherlands"] == {"points": 1, "goals_for": 2, "goals_against": 2, "goal_difference": 0}


def test_standings_simple_case():
    """
    clear order - 1st, 2nd, 3rd, 4th have different points
    """
    g=Group("A",["Qatar","Ecuador","Senegal","Netherlands"])
    g.results.append({"Ecuador": 3, "Qatar": 1})
    g.results.append({"Netherlands": 2, "Qatar": 2})
    g.results.append({"Netherlands": 1, "Senegal": 1})
    g.results.append({"Netherlands": 0, "Ecuador": 2})
    g.results.append({"Senegal": 3, "Qatar": 2})
    g.results.append({"Ecuador": 1, "Senegal": 0})
    # Ecuador should have 9 points, Senegal 3, Netherlands 2, Qatar 1
    g.calc_standings()
    assert g.standings["1st"] == "Ecuador"
    assert g.standings["2nd"] == "Senegal"
    assert g.standings["3rd"] == "Netherlands"
    assert g.standings["4th"] == "Qatar"

def test_standings_2nd_3rd_goal_difference():
    """
    2nd and 3rd place teams have equal points but one has better goal difference
    """
    # Netherlands beat everyone, Senegal and Qatar both beat Ecuador, but
    # Senegal wins by more goals, and they draw with each other.
    g=Group("A",["Qatar","Ecuador","Senegal","Netherlands"])
    g.results.append({"Ecuador": 0, "Qatar": 1})
    g.results.append({"Netherlands": 2, "Qatar": 0})
    g.results.append({"Netherlands": 3, "Senegal": 1})
    g.results.append({"Netherlands": 2, "Ecuador": 1})
    g.results.append({"Senegal": 3, "Qatar": 3})
    g.results.append({"Ecuador": 1, "Senegal": 3})
    g.calc_standings()
    assert g.standings["1st"] == "Netherlands"
    assert g.standings["2nd"] == "Senegal"
    assert g.standings["3rd"] == "Qatar"
    assert g.standings["4th"] == "Ecuador"


def test_standings_2nd_3rd_goals_scored():
    """
    1st 2nd and 3rd place have equal points, but 1st has better goal difference,
    2nd has better goals scored
    """
    # Top 3 teams have 1 win, 1 draw, 1 loss each
    g=Group("A",["Qatar","Ecuador","Senegal","Netherlands"])
    g.results.append({"Ecuador": 1, "Qatar": 1})
    g.results.append({"Netherlands": 2, "Qatar": 2})
    g.results.append({"Netherlands": 2, "Senegal": 0})
    g.results.append({"Netherlands": 2, "Ecuador": 2})
    g.results.append({"Senegal": 2, "Qatar": 3})
    g.results.append({"Ecuador": 1, "Senegal": 0})
    g.calc_standings()
    assert g.standings["1st"] == "Netherlands"
    assert g.standings["2nd"] == "Qatar"
    assert g.standings["3rd"] == "Ecuador"
    assert g.standings["4th"] == "Senegal"


def test_standings_1st_2nd_head_to_head():
    """
    1st 2nd have equal points, gd, goals scored, but 1st beat second
    """
    # Top 2 teams have 2 wins, 1 loss each
    g=Group("A",["Qatar","Ecuador","Senegal","Netherlands"])
    g.results.append({"Ecuador": 1, "Qatar": 1})
    g.results.append({"Netherlands": 1, "Qatar": 0})
    g.results.append({"Netherlands": 0, "Senegal": 2})
    g.results.append({"Netherlands": 2, "Ecuador": 0})
    g.results.append({"Senegal": 2, "Qatar": 3})
    g.results.append({"Ecuador": 0, "Senegal": 2})
    g.calc_standings()
    assert g.standings["1st"] == "Senegal"
    assert g.standings["2nd"] == "Netherlands"
    assert g.standings["3rd"] == "Qatar"
    assert g.standings["4th"] == "Ecuador"


def test_standings_all_equal():
    """
    If all matches are 1-1 draws, order is random, but code should at least run
    """
    g=Group("A",["Qatar","Ecuador","Senegal","Netherlands"])
    g.results.append({"Ecuador": 1, "Qatar": 1})
    g.results.append({"Netherlands": 1, "Qatar": 1})
    g.results.append({"Netherlands": 1, "Senegal": 1})
    g.results.append({"Netherlands": 1, "Ecuador": 1})
    g.results.append({"Senegal": 1, "Qatar": 1})
    g.results.append({"Ecuador": 1, "Senegal": 1})
    g.calc_standings()
    assert g.standings["1st"] != None
    assert g.standings["2nd"] != None
    assert g.standings["3rd"] != None
    assert g.standings["4th"] != None


def test_standings_higher_scoring_draws():
    """
    If all matches are draws, sort by goals scored
    """
    g=Group("A",["Qatar","Ecuador","Senegal","Netherlands"])
    g.results.append({"Ecuador": 1, "Qatar": 1})
    g.results.append({"Netherlands": 2, "Qatar": 2})
    g.results.append({"Netherlands": 1, "Senegal": 1})
    g.results.append({"Netherlands": 1, "Ecuador": 1})
    g.results.append({"Senegal": 3, "Qatar": 3})
    g.results.append({"Ecuador": 1, "Senegal": 1})
    g.calc_standings()
    assert g.standings["1st"] == "Qatar"
    assert g.standings["2nd"] == "Senegal"
    assert g.standings["3rd"] == "Netherlands"
    assert g.standings["4th"] == "Ecuador"


def test_many_standings():
    """
    Define some sanity checks, run the group with random results many times.
    """

    def sanity_check_points(g):
        top_points = g.table[g.standings["1st"]]["points"]
        if g.table[g.standings["2nd"]]["points"] > top_points or \
           g.table[g.standings["3rd"]]["points"] > top_points or \
           g.table[g.standings["4th"]]["points"] > top_points:
            return False
        return True

    def sanity_check_goal_diff(g):

        top_points = g.table[g.standings["1st"]]["points"]
        top_gd = g.table[g.standings["1st"]]["goal_difference"]
        if g.table[g.standings["2nd"]]["points"] == top_points and \
           g.table[g.standings["2nd"]]["goal_difference"] > top_gd:
            return False
        if g.table[g.standings["3rd"]]["points"] == top_points and \
           g.table[g.standings["3rd"]]["goal_difference"] > top_gd:
            return False
        if g.table[g.standings["4th"]]["points"] == top_points and \
           g.table[g.standings["4th"]]["goal_difference"] > top_gd:
            return False
        return True
    fix_df = get_fixture_data()
    for _ in range(100):
        g=Group("A",["Qatar","Ecuador","Senegal","Netherlands"])
        g.results.append({"Ecuador": random.randint(0,4), "Qatar": random.randint(0,4)})
        g.results.append({"Netherlands": random.randint(0,4), "Qatar": random.randint(0,4)})
        g.results.append({"Netherlands": random.randint(0,4), "Senegal": random.randint(0,4)})
        g.results.append({"Netherlands": random.randint(0,4), "Ecuador": random.randint(0,4)})
        g.results.append({"Senegal": random.randint(0,4), "Qatar": random.randint(0,4)})
        g.results.append({"Ecuador": random.randint(0,4), "Senegal": random.randint(0,4)})
        g.calc_standings()
        assert sanity_check_points(g)
        assert sanity_check_goal_diff(g)


def test_add_group_result():
    """
    Add a result and check that the group has the result.
    """
    t = Tournament()
    t.add_result("Qatar","Netherlands", 3, 2, "Group")
    assert len(t.groups["A"].results) == 1


def test_add_group_result_twice():
    """
    Add same result twice and check that the group has the result once.
    """
    t = Tournament()
    t.add_result("Qatar","Netherlands", 3, 2, "Group")
    t.add_result("Netherlands","Qatar", 1, 1, "Group")
    assert len(t.groups["A"].results) == 1


def test_play_group_stage(mocker):
    """
    Play simulated group stage 100 times - ensure that we always have 16 qualifying teams at the end.
    """
    def pick_random_score():
        s1 = random.randint(0,3)
        s2 = random.randint(0,3)
        return s1, s2
    for _ in range(100):
        t = Tournament()
        mocker.patch('wcpredictor.src.tournament.predict_group_match',
                     return_value=pick_random_score())
        t.play_group_stage("dummy")
        for group in t.groups.values():
            assert len(group.results) == 6
            assert len(group.get_qualifiers()) == 2


def test_play_knockout_stages(mocker):
    """
    Play simulated knockout stages 100 times, check that we always get a winner.
    """
    def pick_random_score():
        s1 = random.randint(0,3)
        s2 = random.randint(0,3)
        return s1, s2
    def pick_random_winner(wc_pred, team_1, team_2, seed):
        if random.random() > 0.5:
            return team_1
        else:
            return team_2
    teams_df = get_teams_data()
    teams = list(teams_df.Team.values)
    t = Tournament()
    mocker.patch('wcpredictor.src.tournament.predict_group_match',
                     return_value=pick_random_score())
    t.play_group_stage("dummy")
    for _ in range(100):
        mocker.patch('wcpredictor.src.tournament.predict_knockout_match',
                     new=pick_random_winner)
        t.play_knockout_stages("dummy")
        assert t.winner in teams
