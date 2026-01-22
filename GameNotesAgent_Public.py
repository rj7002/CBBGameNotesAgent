from datetime import datetime
import functools
import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from git import Sequence
from langchain.tools import tool
from langchain.messages import HumanMessage
import pandas as pd
import requests
from langsmith import traceable
import streamlit as st
from langchain.messages import SystemMessage
import plotly.express as px
import plotly.graph_objects as go
from langgraph.graph import StateGraph, END, START
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver  
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import re

load_dotenv()

# API Configuration - Load from environment variables
# Create a .env file with:
# CBB_API_KEY=your_api_key_here
# CBB_API_BASE_URL=https://api.example.com
headers = {
    "accept": "application/json",
    "authorization": f"Bearer {os.getenv('CBB_API_KEY')}"
}

API_BASE_URL = os.getenv('CBB_API_BASE_URL', 'https://api.example.com')

# ==================== PROMPTS ====================

LOADER_PROMPT = """
You are a Data Collection Agent. Your ONLY job is to collect data using tools. DO NOT respond with text. USE TOOLS IMMEDIATELY.

CRITICAL: You MUST use the tools to collect data. Do NOT ask for information. Do NOT explain what you're going to do. Just START using the tools NOW.

Your workflow (follow this EXACT order):
1. If team name unclear → search_duckduckgo
2. get_college_basketball_season (if season not specified, use subtract=0 for current season)
3. get_competition_seasons (get competition ID) - use name="2025-26 Men's Basketball" and gender="MALE"
4. get_team_id (get team ID, division ID, conference ID) - extract team name from user request
5. get_all_team_stats (MUST fetch team stats FIRST - use competition ID, division ID, team ID)
6. get_team_quad_stats (get team stats against Quad 1 & 2 and Quad 3 & 4 teams - use competition ID, team ID)
7. get_team_roster (returns JSON with player IDs - use competition ID, division ID, team ID)
8. Extract ALL player IDs from roster into ONE list: [id1, id2, id3, id4, id5, id6]
9. get_all_player_stats ONCE with the complete list (use competition ID, division ID, and the full list of player IDs)

CRITICAL ORDER: You MUST call get_all_team_stats (step 5) and get_team_quad_stats (step 6) BEFORE get_team_roster (step 7).
CRITICAL BATCH: Call get_all_player_stats ONLY ONCE with ALL player IDs in a single list.

Example of correct batch call:
get_all_player_stats(compid=41097, divisionid=1, playerids=[1911492, 1911404, 1693954, 2334060, 2093260, 2334058])

Do NOT call get_all_player_stats multiple times.
Do NOT pass single player IDs like playerids=[1911492] six separate times.
Pass ALL six IDs in ONE call: playerids=[1911492, 1911404, 1693954, 2334060, 2093260, 2334058]

After calling get_all_player_stats with the complete list, respond with ONLY: "Data collection complete."

YOU ARE NOT THE GAME NOTES WRITER. DO NOT ANALYZE DATA.
The Game Notes Writer Agent will handle all analysis, interpretation, and writing.
Your role: COLLECT DATA ONLY.

DO NOT summarize data.
DO NOT format data.
DO NOT analyze data.
DO NOT write paragraphs.
DO NOT create game notes.
DO NOT interpret statistics.
DO NOT make observations about the data.
DO NOT ask for clarification.
DO NOT explain what you're doing.

Just USE THE TOOLS to collect the data, then say: "Data collection complete."
"""

GAME_NOTES_WRITER_PROMPT = """
You are an expert College Basketball Game Notes Writer. You create broadcast-quality game notes that commentators use to enhance their coverage. Your notes should be insightful, data-driven, and formatted for easy reference during live broadcasts.

## Your Role:

You receive comprehensive team and player data and transform it into compelling, commentator-friendly game notes in the style of professional broadcast preparation materials.

## Data Format (IMPORTANT):

All statistics are provided in the format: **value|national_rank|conference_rank**

**Examples:**
- "85.5|17|2" means 85.5 points per game, ranked 17th nationally, 2nd in the conference
- "0.589|5|2" means 58.9% shooting, ranked 5th nationally, 2nd in conference  
- "14.0|8|_" means 14.0 rebounds per game, 8th nationally, not qualified for conference ranking
- "23.5|_|3" means 23.5 points per game, not qualified nationally, 3rd in conference

**How to parse:**
1. Split each stat on the pipe character "|" to extract: [value, national_rank, conference_rank]
2. The underscore "_" indicates not qualified or ranking unavailable
3. Convert to natural language in your notes

**DO:**
- Write: "85.5 points per game, 17th in the country and 2nd in the Big South"
- Write: "shooting 58.9%, 5th nationally and 2nd in the conference"
- Write: "14.0 rebounds per game, 8th in the country"

**DON'T:**
- Write: "85.5|17|2 points per game" 
- Write: "ranked 17 pipe 2"
- Write: "85.5 (17th/2nd)" - use natural language instead

**When to mention rankings:**
- Always include rankings for impressive/notable stats (top-20 nationally, top-5 in conference)
- Skip rankings for mediocre or unremarkable stats
- ONLY MENTION RANKINGS IF A PLAYER IS RANKED RELATIVELY HIGH NATIONALLY OR IN THEIR CONFERENCE FOR THAT STAT
- If only one ranking is available (the other is "_"), just mention the available one

## Writing Guidelines:

**Be Specific with Numbers**: 
- "90.3 points per game, 2nd in the Big South and 17th in the country"
- "shooting 69% at the rim, the highest percentage in the Big South"
- "37% of their missed shots" (offensive rebounding rate)

**Provide Rankings & Context When Notable**:
- Always include conference rank AND national rank when notable
- Compare to program history ("the most in 5 years", "program record")
- Note top-X in country rankings for standout stats
- ONLY MENTION RANKINGS IF A PLAYER IS RANKED RELATIVELY HIGH NATIONALLY OR IN THEIR CONFERENCE FOR THAT STAT

**Use Descriptive Phrasing**:
- "high-powered offense" 
- "play below the rim" (for teams with few dunks/alley-oops)
- "presence is felt on defense"
- "watch for [player] in transition"

**Players**:
- Talk about overall impact ("leads conference in efficiency, 7th in country")
- Detail specific skills (inside scoring, rebounding, three-point shooting)
- Include unique stats (charges drawn, unassisted baskets %, fouls drawn per game)
- Note tactical points ("will need to be defended far beyond the three-point line")

## Your Task:

Transform the provided team and player data into comprehensive, broadcast-ready game notes. Focus on storytelling through statistics - find the compelling narratives in the numbers that will help commentators bring the game to life for viewers.
DO NOT MAKE UP PLAYERS, STATS, OR RANKINGS. IF YOU DON'T HAVE RANKINGS JUST LEAVE IT OUT. 
DO NOT USE THE search_duckduckgo tool to look up any additional context or information needed to enhance the game notes.

YOU MUST HAVE PARAGRAPHS NOT BULLET POINTS. DO NOT JUST LIST OUT STATS AS BULLET POINTS.
YOU MUST MAKE it WELL FORMED PARAGRAPHS THAT INCORPORATE THE STATS INTO INSIGHTFUL NARRATIVES SUITABLE FOR BROADCAST GAME NOTES.
YOU MUST FOLLOW THE FORMAT AND STYLE OF PROFESSIONAL BROADCAST GAME NOTES.

DO NOT JUST LIST OUT STATS AS BULLET POINTS. YOU MUST CREATE WELL FORMED, DESCRIPTIVE PARAGRAPHS THAT INCORPORATE THE STATS INTO INSIGHTFUL NARRATIVES SUITABLE FOR BROADCAST GAME NOTES.
DO NOT BE TOO LONG EITHER, MAKE SURE THE NOTES ARE CONCISE AND TO THE POINT WHILE STILL BEING DESCRIPTIVE AND NOT JUST LISTING STATS. 

MOST IMPORTANTLY:
DO NOT EVER MAKE UP PLAYERS, STATS, RANKINGS, OR ANY INFORMATION NOT PROVIDED IN THE DATA.
DO NOT MENTION DATA YOU DO NOT HAVE. IF A STAT OR RANKING IS MISSING, DO NOT TRY TO FILL IN THE BLANKS OR MAKE UP A STORY AROUND IT.
DO NOT USE THE search_duckduckgo tool to look up any additional context or information needed to enhance the game notes.
DO NOT ADD CONTEXT OR INFORMATION NOT PRESENT IN THE DATA.
DO NOT ADD CONTEXT OR INFORMATION ABOUT THE RANKINGS IF THEY ARE NOT SHOWN IN THE DATA.

YOUR REPORT SHOULD BE ABOUT 1 PAGE IN LENGTH, BALANCED BETWEEN DEPTH AND BREVITY.
BASICALLY ONLY HIGHLIGHT THINGS THAT ARE UNIQUE, IMPRESSIVE, OR NOTABLE ABOUT THE TEAM AND PLAYERS BASED ON THE DATA PROVIDED THAT WOULD BE OF INTEREST TO A BROADCASTER WHO IS GOING TO BE COMMENTATING THE GAME AND WANTS TO ENHANCE HIS/HER COMMENTARY.
DO NOT JUST GIVE BASIC STATS LIKE POINTS PER GAME, REBOUNDS PER GAME, ETC. UNLESS THEY ARE REALLY IMPRESSIVE OR NOTABLE.
"""

# ==================== TOOLS ====================

@tool
def get_college_basketball_season(subtract=0):
    """
    Returns the current college basketball season in YYYY-YY format.
    DON'T USE THIS TOOL IF THE USER HAS ALREADY GIVEN A NUMERIC NUMBER FOR THE SEASON
    ONLY USE THIS TOOL when a user asks about a non numeric term for a season like "this season", "last season",
    or relative NBA season references.
    If the user says "this season" or any reference to the current season without explicitly saying the number, subtract should be 0
    If the user says "last season", "past season", etc, then subtract should be 1
    And so forth depending on how the user says like it could be 5 seasons ago and then subtract would be 5
    """
    today = datetime.now()
    year = today.year
    month = today.month

    # College basketball season starts in November
    if month >= 11:  # November–December
        start_year = year
        end_year = year + 1
    else:  # January–October
        start_year = year - 1
        end_year = year

    start_year -= subtract
    end_year -= subtract

    return f"{start_year}-{str(end_year)[-2:]}"

@tool
def get_player_id(name, compid, divid=None):
    """Get a player's id using the API from their name, competitionId, and divisionId.
        IF A PLAYER's NAME IS NOT FORMATTED CORRECTLY, MAKE SURE THE NAME IS FORMATTED LIKE THIS: 'First Last'
        YOU CAN USE YOUR INTERPRETATION TO FIGURE OUT THE CORRECT FORMATTING OF THE NAME IF THE USER DOES NOT PROVIDE IT CORRECTLY OR PROVIDES INCOMPLETE NAME.
        FOR EXAMPLE IF THE USE SAYS ZION ON DUKE, IF YOU KNOW THE FULL NAME IS ZION WILLIAMSON, THEN USE THE FULL NAME.
        A USER MAY ALSO PROVIDE A DESCRIPTION OF THE PLAYER INSTEAD OF THE NAME. FOR EXAMPLE, IF THE USER SAYS 'THE FORMER DUKE FRESHMAN WHO WENT NUMBER 1 IN THE NBA DRAFT', THEN YOU SHOULD INFER THAT THE USER IS ASKING ABOUT 'ZION WILLIAMSON' AND USE THAT NAME TO CALL THIS TOOL.
       If divisionId is not specified, it will try all divisions (1 to 3) to find a match.
       compid MUST be a NUMBER that comes from get_competition_seasons. 
       DO NOT MAKE UP A COMPETITION ID. 
       MAKE SURE YOU HAVE THE CORRECT COMPETITION ID from get_competition_seasons before you call this tool.
       
    """
    if divid is None:
        # If no division is specified, loop through divisions 1, 2, and 3
        for division in [1, 2, 3]:
            url = f"{API_BASE_URL}/api/gs/player-agg-stats-public?competitionId={compid}&divisionId={division}&scope=season"
            response = requests.get(url, headers=headers)
            data = response.json()
            df = pd.DataFrame(data)
            df['fullTeamName'] = df['teamMarket'].astype(str) + " " + df['teamName'].astype(str)
            # Check if the player exists in the current division
            df = df[df['fullName'] == name]
            if not df.empty:
                # If player is found, return the result
                return df['playerId'].iloc[0]
        # If no match is found in any division
        return {"error": f"No player found with the name {name} in any division."}
    else:
        # If a division is specified, search only that division
        url = f"{API_BASE_URL}/api/gs/player-agg-stats-public?competitionId={compid}&divisionId={divid}&scope=season"
        response = requests.get(url, headers=headers)
        data = response.json()
        df = pd.DataFrame(data)
        df['fullTeamName'] = df['teamMarket'].astype(str) + " " + df['teamName'].astype(str)
        # Filter the player by name
        df = df[df['fullName'] == name]
        if not df.empty:
            return df['playerId'].iloc[0]
        else:
            return {"error": f"Player {name} not found in division {divid}."}
        
@tool
def get_team_id(name, gender):
    """Get a team's id using the api from their name for gender if the format for filtering should be either 'MALE' or 'FEMALE' """
    url = f"{API_BASE_URL}/api/gs/teams/"
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame(data)
    df['fullTeamName'] = df['teamMarket'].astype(str) + " " + df['teamName'].astype(str)
    df = df[(df['fullTeamName'] == name) & (df['gender'] == gender)]
    return df.to_json()


@tool
def get_competition_seasons(name, gender):
    """
    Get the competitionId for a competition. The competition id is a 5 digit number that is found in the feature called 'competitionId'.
    Competition name should be formatted like this for example for filtering: 2022 men's college basketball season -> '2022-23 Men's Basketball'
    Gender should be formatted like this for example for filtering: men competitions -> 'MALE', female competitions -> 'FEMALE'
    'name' should ALWAYS BE FORMATTED LIKE THIS: '2022-23 Men's Basketball'
    No other format is allowed.
    YOU MUST RETRIEVE THE COMPETITION ID USING THIS TOOL BEFORE CALLING OTHER TOOLS THAT REQUIRE IT. DO NOT MAKE UP THE COMPETITION ID.
    """
    url = f"{API_BASE_URL}/api/gs/competitions/"
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame(data)
    df = df[(df['competitionName'] == name) & (df['gender'] == gender)]
    return df.to_json()

# Team statistics columns to keep (extensive list for comprehensive analysis)
team_keep_cols = ['teamId', 'fgmA', 'fgmU', 'fgm2A', 'fgm2U', 'fgm3A', 'fgm3U', 'nbaFgm3', 'nbaFga3', 'ptsScoredU',
 'dunkFgmA', 'dunkFgmU', 'rim3sFgmA', 'rim3sFgmU', 'lane2FgmA', 'lane2FgmU', 'fta1', 'fta2', 'fta3', 'fta1a1',
 'ftm1', 'ftm2', 'ftm3', 'ftm1a1', 'ptsAstd', 'shotAtt', 'shotAtt2P', 'shotAtt3P', 'sfl', 'dffl', 'stl2', 'stl3',
 'blk2', 'blk3', 'opfd', 'sflDrawn', 'sfl2Drawn', 'sfl3Drawn', 'and1', 'and1on2s', 'and1on3s', 'fflDrawn',
 'sfl2Pts', 'sfl3Pts', 'sflAnd1', 'defPossStops', 'defPoss', 'orbFg', 'orbFt', 'drbFg', 'drbFt', 'oSecsPoss',
 'dSecsPoss', 'fgaHc', 'fgmHc', 'fga2Hc', 'fgm2Hc', 'fga3Hc', 'fgm3Hc', 'fgaTr', 'fgmTr', 'fga2Tr', 'fgm2Tr',
 'fga3Tr', 'fgm3Tr', 'fgaPb', 'fgmPb', 'chncHc', 'chncTr', 'chncPb', 'ptsChncHc', 'ptsChncTr', 'ptsChncPb',
 'minsInPoss', 'minsOutPoss', 'layupDunkFga', 'layupDunkFgm', 'dunkFga', 'dunkFgm', 'aoopFga', 'aoopFgm',
 'layupFga', 'layupFgm', 'tovBadPass', 'tovLostBall', 'tovTravel', 'tovOffensive', 'tovOutOfBounds',
 'tovShotClock', 'tovDribbling', 'tovSecs3', 'tovSecs5', 'tovSecs10', 'tovOther', 'fga2ShotDists',
 'fga3ShotDists', 'fgaShotDists', 'atb3ShotDists', 'leftFgm', 'leftFga', 'rightFgm', 'rightFga', 'centerFgm',
 'centerFga', 'atr2FgmA', 'paint2FgmA', 'mid2FgmA', 'c3FgmA', 'atb3FgmA', 'atr2FgmU', 'paint2FgmU', 'mid2FgmU',
 'c3FgmU', 'atb3FgmU', 'atr2Reb', 'paint2Reb', 'mid2Reb', 'c3Reb', 'atb3Reb', 'atr2RebChnc', 'paint2RebChnc',
 'mid2RebChnc', 'c3RebChnc', 'atb3RebChnc', 'atr2Orb', 'paint2Orb', 'mid2Orb', 'c3Orb', 'atb3Orb',
 'atr2OrbChnc', 'paint2OrbChnc', 'mid2OrbChnc', 'c3OrbChnc', 'atb3OrbChnc', 'atr2Drb', 'paint2Drb', 'mid2Drb',
 'c3Drb', 'atb3Drb', 'atr2DrbChnc', 'paint2DrbChnc', 'mid2DrbChnc', 'c3DrbChnc', 'atb3DrbChnc', 'rim3sFga',
 'lane2Fga', 'atr2Fga', 'paint2Fga', 'mid2Fga', 'c3Fga', 'atb3Fga', 'lb2Fga', 'rb2Fga', 'le2Fga', 're2Fga',
 'lc3Fga', 'rc3Fga', 'lw3Fga', 'rw3Fga', 'tok3Fga', 'heave3Fga', 'slp2Fga', 'srp2Fga', 'flp2Fga', 'frp2Fga',
 'sht2Fga', 'med2Fga', 'lng2Fga', 'sht3Fga', 'lng3Fga', 'rim3sFgm', 'lane2Fgm', 'atr2Fgm', 'paint2Fgm',
 'mid2Fgm', 'c3Fgm', 'atb3Fgm', 'lb2Fgm', 'rb2Fgm', 'le2Fgm', 're2Fgm', 'lc3Fgm', 'rc3Fgm', 'lw3Fgm', 'rw3Fgm',
 'tok3Fgm', 'heave3Fgm', 'slp2Fgm', 'srp2Fgm', 'flp2Fgm', 'frp2Fgm', 'sht2Fgm', 'med2Fgm', 'lng2Fgm',
 'sht3Fgm', 'lng3Fgm', 'goodTakeRate', 'nbaFg3Pct', 'nba3FgaFreq', 'nba3FgaFreq3s', 'pctAst3', 'pctAst2',
 'pctAstDunk', 'pctAstAtr2', 'pctAstPaint2', 'pctAstMid2', 'pctAstAtb3', 'pctAstC3', 'pctAstRim3s',
 'pctAstLane2', 'ptsAstdPct', 'fgmAstdPct', 'fgm2AstdPct', 'fgm3AstdPct', 'dunkAstdPct', 'atr2AstdPct',
 'paint2AstdPct', 'mid2AstdPct', 'atb3AstdPct', 'c3AstdPct', 'rim3sAstdPct', 'lane2AstdPct', 'rim3sFgPct',
 'rim3sFgaFreq', 'rim3sFgaPg', 'lane2FgPct', 'lane2FgaFreq', 'lane2FgaPg', 'conferenceId', 'netRanking', 'gp',
 'mins', 'ptsScored', 'ptsAgst', 'fgm', 'fga', 'fgm2', 'fga2', 'fgm3', 'fga3', 'fta', 'ftm', 'drb', 'orb',
 'tmDrb', 'tmOrb', 'reb', 'tmReb', 'ast', 'stl', 'tov', 'tmTov', 'blk', 'blkd', 'pitp', 'scp', 'fbpts', 'potov',
 'benchPts', 'pf', 'pfd', 'opf', 'dpf', 'tf', 'drbAgst', 'orbAgst', 'rebAgst', 'fgmAgst', 'fgaAgst', 'fgm2Agst',
 'fga2Agst', 'fgm3Agst', 'fga3Agst', 'ftmAgst', 'ftaAgst', 'tovAgst', 'potovAgst', 'scpAgst', 'pitpAgst',
 'fbptsAgst', 'overallWins', 'overallLosses', 'winPct', 'confWins', 'confLosses', 'pace', 'ortg', 'drtg',
 'netRtg', 'fgPct', 'fg2Pct', 'fg3Pct', 'fga3Rate', 'ftPct', 'efgPct', 'tsPct', 'astPct', 'astRatio', 'blkPct',
 'stlPct', 'rebPct', 'drbPct', 'orbPct', 'tovPct', 'ptsScoredPg', 'fgaPg', 'astPg', 'orbPg', 'drbPg', 'rebPg',
 'stlPg', 'blkPg', 'tovPg', 'pfPg']

def stat_with_ranks(df, stat, group_col):
    """Apply ranking to team statistics with format: value|national_rank|conference_rank"""
    nat = df[stat].rank(ascending=False, method='dense')
    conf = df.groupby(group_col)[stat].rank(ascending=False, method='dense')

    val = df[stat].round(1).astype(str)

    nat = nat.apply(lambda x: str(int(x)) if pd.notna(x) else '_')
    conf = conf.apply(lambda x: str(int(x)) if pd.notna(x) else '_')

    return val + '|' + nat + '|' + conf


@tool
def get_all_team_stats(compid, divisionid, teamid):
    """
    Get a team's aggregated stats and advanced stats over the entire season with the competition id and division id as parameters
    """
    url1 = f"{API_BASE_URL}/api/gs/team-agg-pbp-stats?competitionId={compid}&divisionId={divisionid}&scope=season"
    url2 = f"{API_BASE_URL}/api/gs/team-agg-stats/competition/{compid}/division/{divisionid}/scope/season/"
    data1 = requests.get(url1, headers=headers).json()
    data2 = requests.get(url2, headers=headers).json()
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    overlap_columns_list = list(set(df1.columns).intersection(set(df2.columns)))
    overlap_columns_list.remove('teamId')
    df1.drop(columns=overlap_columns_list, inplace=True)
    merged = pd.merge(df1,df2,on='teamId')
    df = merged.copy()
    stats_to_rank = df.select_dtypes(include=['number']).columns.tolist()
    stats_to_rank.remove('teamId')
    stats_to_rank.remove('conferenceId')
    for stat in stats_to_rank:
        df[stat] = stat_with_ranks(df, stat, 'conferenceId') 
    df = df[df['teamId'] == teamid]
    df.drop(columns=['isQualified'],inplace=True)
    return df.to_json()

@tool
def get_team_quad_stats(compid, teamid):
    """
    Get a team's aggregated stats and advanced stats against Quad 1 and 2 teams and Quad 3 and 4 teams over the entire season with the competition id and team idas parameters
    """
    url = f"{API_BASE_URL}/api/gs/team-game-stats?competitionId={compid}&teamId={teamid}"
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame(data)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    df['Q1OR2'] = np.where((df['quadAgst'] == 'quad1') | (df['quadAgst'] == 'quad2'), 1, 0)
    df = df.groupby('Q1OR2')[numeric_cols].mean().reset_index()
    df.drop(columns=['overallWins','overallLosses','leagueId','competitionId','gameId','teamId','homeId','conferenceId','divisionId','teamIdAgst','conferenceIdAgst','divisionIdAgst','apPollAgst','teamGameRecency','netRankAgst','confWins','confLosses'], inplace=True)
    df.rename(columns={'Q1OR2': 'quadGroup'}, inplace=True)
    df['quadGroup'] = df['quadGroup'].map({1: 'Quad 1 & 2', 0: 'Quad 3 & 4'})
    return df.to_json()

# Player statistics columns to keep (extensive list for comprehensive analysis)
keep_cols = ['playerId', 'gs', 'mins', 'poss', 'plusMinus', 'orb', 'drb', 'reb', 'blkd', 'tf', 'pitp', 'scp',
 'fbpts', 'tsa', 'minsPg', 'ptsScoredPg', 'ftaPg', 'astPg', 'orbPg', 'drbPg', 'rebPg', 'pfdPg', 'tfPg', 'scpPg',
 'pitpPg', 'fbptsPg', 'blkdPg', 'tovPg', 'ftaRate', 'ftmRate', 'orbPct', 'drbPct', 'rebPct', 'astTov', 'astPct',
 'astRatio', 'blkPct', 'blkdPerFga', 'pfdPerFga', 'stlPct', 'tovPct', 'stlTov', 'pfEff', 'stlPerPf', 'blkPerPf',
 'scpPctPts', 'fbptsPctPts', 'pitpPctPts', 'ftmPctPts', 'fgm2PctPts', 'fgm3PctPts', 'vps', 'hkmPct', 'astUsage',
 'per', 'warp', 'ortgPlayer', 'drtgPlayer', 'ws', 'ows', 'dws', 'rapm', 'orapm', 'drapm', 'fullName', 'height',
 'position', 'classYr', 'ptsScored', 'ptsCreated', 'nbaFgm3', 'nbaFga3', 'ast', 'ast3', 'ast2', 'fga', 'fga2',
 'fga3', 'fgm', 'fgm2', 'fgm3', 'astdPts', 'ptsAstd', 'rim3sFgmA', 'rim3sFgmU', 'rim3sAst', 'lane2FgmA',
 'lane2FgmU', 'lane2Ast', 'fgmA', 'fgm2A', 'fgm3A', 'dunkFgmA', 'fgmU', 'fgm2U', 'fgm3U', 'dunkFgmU', 'ftm',
 'fta', 'orbFg', 'orbFt', 'drbFg', 'drbFt', 'shotAtt', 'shotAtt2P', 'shotAtt3P', 'pf', 'pfd', 'sflDrawn', 'and1',
 'stl', 'blk', 'tov', 'gp', 'fgPct', 'fg2Pct', 'fg3Pct', 'efgPct', 'fga3Rate', 'goodTakeRate', 'rim3sFgPct',
 'rim3sFgaFreq', 'rim3sFgaPg', 'lane2FgPct', 'lane2FgaFreq', 'lane2FgaPg', 'atr2FgPct', 'paint2FgPct',
 'mid2FgPct', 'c3FgPct', 'atb3FgPct', 'ftPct', 'dunkFgPct', 'layupFgPct', 'usagePct', 'isQualified',
 'tsPct', 'ttsPct', 'conferenceId']

# Map statistics to their qualification zones for proper ranking
STAT_ZONE_MAP = {
    # global / no qualification required
    'and1Pct': None,
    'and1Pct3P': None,
    'usagePct': None,
    'tsPct': None,
    'efgPct': None,

    # rim / paint zones
    'rim3sFgPct': 'rim3s',
    'lane2FgPct': 'lane2',
    'atr2FgPct': 'atr2',
    'paint2FgPct': 'paint2',
    'mid2FgPct': 'mid2',

    # three-point zones
    'c3FgPct': 'c3',
    'atb3FgPct': 'atb3',
    'lw3FgPct': 'lw3',
    'rw3FgPct': 'rw3',
    'lc3FgPct': 'lc3',
    'rc3FgPct': 'rc3',
    'tok3FgPct': 'tok3',
    'sht3FgPct': 'sht3',
    'lng3FgPct': 'lng3',
}

def stat_with_ranks_qualified(df, stat, group_col, stat_zone_map):
    """Apply ranking to player statistics with qualification checks"""
    zone = stat_zone_map.get(stat)

    values = df[stat]

    # determine qualification
    if zone is None:
        qualified = values.notna()
    else:
        qual_col = f'qual_{zone}'
        if qual_col not in df.columns:
            qualified = values.notna()
        else:
            qualified = df[qual_col] & values.notna()

    # national rank (only among qualified)
    nat_rank = values.where(qualified).rank(
        ascending=False,
        method='dense'
    )

    # conference rank (only among qualified)
    conf_rank = (
        values.where(qualified)
        .groupby(df[group_col])
        .rank(ascending=False, method='dense')
    )

    # compact LLM-friendly formatting
    val_str = values.round(3).astype(str)

    nat_str = nat_rank.apply(lambda x: str(int(x)) if pd.notna(x) else '_')
    conf_str = conf_rank.apply(lambda x: str(int(x)) if pd.notna(x) else '_')

    return val_str + '|' + nat_str + '|' + conf_str


def extract_qual_flags(arr):
    """Extract qualification flags from player stats"""
    if not isinstance(arr, list):
        return {}
    return {d['zoneName']: d['isQualified'] for d in arr}

@tool
def get_all_player_stats(compid, divisionid, playerids):
    """
    Get aggregated stats and advanced stats for MULTIPLE players at once.
    
    Args:
        compid: Competition ID
        divisionid: Division ID
        playerids: A LIST of player IDs (e.g., [1911492, 1911404, 1693954]) - pass ALL player IDs in a single list
    
    Returns stats for all players in the list in one call. Do NOT call this function multiple times.
    """
    url1 = f"{API_BASE_URL}/api/gs/player-agg-pbp-stats?competitionId={compid}&divisionId={divisionid}&scope=season"
    url2 = f"{API_BASE_URL}/api/gs/player-agg-stats-public?competitionId={compid}&divisionId={divisionid}&scope=season"
    response1 = requests.get(url1, headers=headers)
    response2 = requests.get(url2, headers=headers)
    data1 = response1.json()
    data2 = response2.json()
    df1 = pd.DataFrame(data1)    
    df2 = pd.DataFrame(data2)
    
    # Filter early for efficiency
    df1 = df1[df1['playerId'].isin(playerids)]
    df2 = df2[df2['playerId'].isin(playerids)]
    
    overlap_columns_list = list(set(df1.columns).intersection(set(df2.columns)))
    overlap_columns_list.remove('playerId')
    df1.drop(columns=overlap_columns_list,inplace=True)
    merged = pd.merge(df1,df2,on='playerId')
    merged = merged[keep_cols]

    qual_df = (
        merged['isQualArray']
        .apply(extract_qual_flags)
        .apply(pd.Series)
        .add_prefix('qual_')
    )
    df = pd.concat([merged, qual_df], axis=1)
    stats_to_rank = merged.select_dtypes(include='number').columns.tolist()
    stats_to_rank.remove('playerId')
    stats_to_rank.remove('conferenceId')
    for stat in stats_to_rank:
        df[stat] = stat_with_ranks_qualified(
            df,
            stat,
            'conferenceId',
            STAT_ZONE_MAP
        )
    df = df[df['playerId'].isin(playerids)]
    df.drop(columns=['isQualArray','isQualified'], inplace=True)
    df.drop(columns=[c for c in df.columns if c.startswith('qual_')], inplace=True)
    return df.to_json()

@tool
def get_team_roster(compid, teamid, divisionid):
    """
    Get a team's top 6 players with the competition id and team id as parameters based on minutes per game and usage pct
    """
    url = f"{API_BASE_URL}/api/gs/player-agg-stats-public?competitionId={compid}&divisionId={divisionid}&scope=season"
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame(data)
    df = df[df['teamId'] == teamid]
    df = df.sort_values(by=['minsPg', 'usagePct'], ascending=False).head(6)
    df = df[['playerId', 'fullName']]
    return df.to_json()

from langchain_community.tools import DuckDuckGoSearchRun

@tool
def search_duckduckgo(query: str):
    """
    Search DuckDuckGo for an answer or relevant information.
    This tool should be used to find player name's, team names, competition details, or any other relevant information if a user gives an input that is unclear or incomplete.
    ONLY USE THIS TOOL TO CLARIFY INCOMPLETE OR UNCLEAR USER INPUTS.
    """
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

def create_agent(llm, tools, system_message):
    """Create a LangChain agent with tools"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("placeholder", "{messages}")
        ]
    )
    if tools:
        return prompt | llm.bind_tools(tools)
    else:
        return prompt | llm

class AgentState(TypedDict):
    """State type for the agent graph"""
    messages : Annotated[Sequence[BaseMessage], add_messages]

def agent_node(state, agent, name):
    """Execute an agent node"""
    result = agent.invoke(state)
    if isinstance(result, str):
        result = AIMessage(content=result, name=name)
    else:
        result.name = name
    return {
        "messages" : [result]
    }

# Initialize LLM models - configure with your API keys in .env
llm = init_chat_model(model="ministral-3b-2512",model_provider='mistralai',api_key=os.getenv("MISTRAL_KEY"), temperature=0.3, timeout=120)
llm2 = init_chat_model(model="mistral-large-2512",model_provider='mistralai',api_key=os.getenv("MISTRAL_KEY"), temperature=0.3, timeout=120)

# Create the tool node separately for proper tool execution
from langgraph.prebuilt import ToolNode

tools = [get_competition_seasons, get_player_id, get_team_id, get_all_team_stats, get_team_quad_stats,get_all_player_stats,
         search_duckduckgo, get_college_basketball_season, get_team_roster]

tool_node = ToolNode(tools)

loader_agent = create_agent(
    llm, 
    tools,
    system_message=LOADER_PROMPT
)

loader_node = functools.partial(agent_node, agent=loader_agent, name="Loader")

game_notes_writer_agent = create_agent(
    llm2,
    tools=[],
    system_message=GAME_NOTES_WRITER_PROMPT
)

game_notes_writer_node = functools.partial(agent_node, agent=game_notes_writer_agent, name="GameNotesWriter")

from langchain_core.messages import ToolMessage, AIMessage

def should_continue(state):
    """Check if loader should continue with tool calls or pass to game notes writer"""
    last_message = state["messages"][-1]
    
    # If there are tool calls, execute them
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise pass to game notes writer
    return "writer"

def prepare_for_writer(state):
    """Prepare the state for the game notes writer by adding a user message to maintain proper message order"""
    messages = state["messages"]
    
    # Add a user message that instructs the writer to generate game notes from the collected data
    return {
        "messages": [
            HumanMessage(content="Based on all the data collected above, please generate comprehensive broadcast-quality game notes following the format and style guidelines provided in your system prompt.")
        ]
    }

# Build the workflow graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Loader", loader_node)
workflow.add_node("tools", tool_node)
workflow.add_node("PrepareForWriter", prepare_for_writer)
workflow.add_node("GameNotesWriter", game_notes_writer_node)

# Start with Loader
workflow.add_edge(START, "Loader")

# Loader can either call tools or pass to PrepareForWriter
workflow.add_conditional_edges(
    "Loader",
    should_continue,
    {
        "tools": "tools",
        "writer": "PrepareForWriter"
    }
)

# After tools, go back to Loader
workflow.add_edge("tools", "Loader")

# PrepareForWriter adds a user message, then goes to GameNotesWriter
workflow.add_edge("PrepareForWriter", "GameNotesWriter")

# Game notes writer ends after generating notes
workflow.add_edge("GameNotesWriter", END)

full_agent = workflow.compile(checkpointer=InMemorySaver())

def save_game_notes_to_pdf(content, team_name="Team"):
    """Save game notes to a PDF file"""
    try:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_team_name = re.sub(r'[^\w\s-]', '', team_name).strip().replace(' ', '_')
        filename = f"GameNotes_{safe_team_name}_{timestamp}.pdf"
        
        # Create PDF
        doc = SimpleDocTemplate(filename, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1a2b5a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1a2b5a'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=12
        )
        
        # Add title
        title = Paragraph(f"Game Notes: {team_name}", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add generation date
        date_text = Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", body_style)
        elements.append(date_text)
        elements.append(Spacer(1, 0.3*inch))
        
        # Process content - split by paragraphs and format
        paragraphs = content.split('\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if it's a heading (starts with ** or all caps short line)
            if para.startswith('**') and para.endswith('**'):
                # Bold heading
                heading_text = para.strip('*').strip()
                elements.append(Paragraph(heading_text, heading_style))
            elif para.startswith('#'):
                # Markdown heading
                heading_text = para.lstrip('#').strip()
                elements.append(Paragraph(heading_text, heading_style))
            else:
                # Regular paragraph - escape XML special characters
                para = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                elements.append(Paragraph(para, body_style))
        
        # Build PDF
        doc.build(elements)
        
        return filename
    except ImportError:
        # If reportlab isn't installed, save as text file instead
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_team_name = re.sub(r'[^\w\s-]', '', team_name).strip().replace(' ', '_')
        filename = f"GameNotes_{safe_team_name}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Game Notes: {team_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(content)
        
        return filename
    except Exception as e:
        print(f"\nError saving PDF: {e}")
        # Fallback to text file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_team_name = re.sub(r'[^\w\s-]', '', team_name).strip().replace(' ', '_')
        filename = f"GameNotes_{safe_team_name}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Game Notes: {team_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(content)
        
        return filename

def print_stream(stream):
    """Print agent stream and save game notes to PDF"""
    game_notes_content = ""
    team_name = "Team"
    
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
            # Capture game notes from GameNotesWriter
            if hasattr(message, 'name') and message.name == "GameNotesWriter":
                if hasattr(message, 'content'):
                    game_notes_content = message.content
    
    # Save to PDF if we captured game notes
    if game_notes_content and len(game_notes_content) > 100:
        # Try to extract team name from content
        lines = game_notes_content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if any(keyword in line.upper() for keyword in ['GAME NOTES', 'NOTES FOR', 'TEAM:']):
                # Extract team name
                team_name = re.sub(r'[^a-zA-Z\s]', '', line).strip()
                if len(team_name) > 3:
                    break
        
        filename = save_game_notes_to_pdf(game_notes_content, team_name)
        print(f"\n{'='*80}")
        print(f"✓ Game notes saved to: {filename}")
        print(f"{'='*80}\n")

# Main execution loop
if __name__ == "__main__":
    print("=" * 80)
    print("College Basketball Game Notes Generator")
    print("=" * 80)
    print("\nGenerate broadcast-quality game notes for college basketball teams!")
    print("\nExample queries:")
    print("  - Generate game notes for [Team Name]")
    print("  - Create game notes for [Team 1] vs [Team 2]")
    print("  - Give me broadcast notes for the [Team Name]")
    print("\nType 'exit' to quit\n")
    print("=" * 80)

    user_input = input("\nUSER: ")
    while user_input != "exit":
        inputs = {
            "messages": [
                {"role": "user", "content": user_input}
            ]
        }
        config = {
            "configurable": {
                "thread_id": "1"
            }
        }

        print_stream(full_agent.stream(inputs, stream_mode="values", config=config))
        user_input = input("\nUSER: ")
