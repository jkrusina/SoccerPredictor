import pandas as pd
import sqlite3
from typing import Any, Dict, List, Tuple

from soccerpredictor.util.constants import DATA_DIR, DB_DIR, DB_FILE


class SPDBManager:
    """
    Used to handle database connection and to query data.

    """

    def __init__(self) -> None:
        self.conn = None
        self.cur = None

    def query_fixtures_data(self, seasons: List[int]) -> pd.DataFrame:
        """
        Queries fixtures data for all teams in given seasons.

        Renames some features as "[home|away]_<feature_name>" for easier access to columns within
        home and away team's viewpoints.

        :param seasons: Seasons include in query.
        :return: Fixtures df.
        """
        df = pd.read_sql("""
            SELECT f.id, f.date, f.season, f.league, 
                   t1.name AS home, t2.name AS away, f.home_goals, f.away_goals, 
                   f.oddsDC_1X AS home_odds_wd, f.oddsDC_X2 AS away_odds_wd,
                   ts1.rating AS home_rating, ts2.rating AS away_rating,
                   ts1.errors AS home_errors, ts2.errors AS away_errors, 
                   ts1.red_cards AS home_red_cards, ts2.red_cards AS away_red_cards,
                   ts1.shots AS home_shots, ts2.shots AS away_shots
            FROM Fixtures f
            JOIN Teams t1 ON f.homeTeamID = t1.id
            JOIN Teams t2 ON f.awayTeamID = t2.id
            JOIN TeamStats ts1 ON f.homeStatsID = ts1.id
            JOIN TeamStats ts2 ON f.awayStatsID = ts2.id
            WHERE f.season IN ({})
            ORDER BY f.date, f.id
            """.format(",".join("?" * len(seasons))),
                         self.conn, params=seasons)

        return df

    def query_team_data(self, seasons: List[int], params: Tuple[Any, ...]) -> pd.DataFrame:
        """
        Queries fixtures data for a single team within given seasons.

        :param seasons: Seasons to query.
        :param params: Params for query.
        :return: Team fixtures df.
        """
        df = pd.read_sql("""
            SELECT f.id, f.date, f.season, f.league, f.homeTeamID, f.awayTeamID,
                   t1.name AS home, t2.name AS away, f.home_goals, f.away_goals, f.winner,
                   ts.rating, ts.goals, ts.errors, ts.red_cards, ts.shots, f.oddsDC_1X, f.oddsDC_X2
            FROM TeamStats ts
            JOIN Fixtures f ON f.id = ts.fixtureID 
            JOIN Teams t1 ON f.homeTeamID = t1.id
            JOIN Teams t2 ON f.awayTeamID = t2.id
            WHERE ts.teamID = ? AND (f.homeTeamID = ? OR f.awayTeamID = ?) AND
                  f.season IN ({})
            ORDER BY f.date, f.id
            """.format(",".join("?" * len(seasons))),
            self.conn, params=params)

        return df

    def query_teams_ids_names_tuples(self) -> Dict[str, int]:
        """
        Queries teams' names and ids from db and stores them into a dict in format
        <team_name>: <team_id>.
        Ordered by team id.

        :return: Dict of all teams' names and corresponding ids from db.
        """
        df = pd.read_sql("""
            SELECT t.id, t.name
            FROM Teams t
            ORDER BY t.id
            """, self.conn)

        return dict(zip(df["name"], df["id"]))

    def query_teams_names(self) -> pd.DataFrame:
        """
        Queries teams' names from db.
        Ordered by team name.

        :return: Dataframe of all teams names in db.
        """
        return pd.read_sql("""
            SELECT t.name  
            FROM Teams t
            ORDER BY t.name
            """, self.conn)

    def connect(self) -> None:
        """
        Connects to DB.

        """
        self.conn = sqlite3.connect(f"file:{DATA_DIR}{DB_DIR}{DB_FILE}?mode=rw", uri=True)
        self.cur = self.conn.cursor()

    def disconnect(self) -> None:
        """
        Disconnects from DB.

        """
        if self.cur is not None:
            self.cur.close()
        if self.conn is not None:
            self.conn.close()
