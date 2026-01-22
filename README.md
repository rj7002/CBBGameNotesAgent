# College Basketball Game Notes Generator

An AI-powered tool that generates broadcast-quality game notes for college basketball teams using LangGraph multi-agent architecture and comprehensive basketball analytics.

## Features

- **Multi-Agent Architecture**: Uses LangGraph to orchestrate a data collection agent and a game notes writing agent
- **Comprehensive Statistics**: Pulls team and player stats including shooting zones, efficiency metrics, and advanced analytics
- **Ranking System**: Automatically ranks stats nationally and within conferences
- **PDF Export**: Saves generated game notes as professionally formatted PDF documents
- **Natural Language Processing**: Converts complex statistical data into broadcaster-friendly narratives

## Installation

### Prerequisites

- Python 3.9+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd LangChainWork
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
cp .env.example .env
```

4. Edit `.env` and add your API keys:
   - `CBB_API_BASE_URL`: The base URL
   - `LLM_KEY`: Your Mistral AI API key

## Requirements

Create a `requirements.txt` file with:

```
langchain
langchain-community
langgraph
pandas
numpy
requests
python-dotenv
reportlab
duckduckgo-search
```

## Usage

Run the game notes generator:

```bash
python GameNotesAgent_Public.py
```

### Example Queries

```
Generate game notes for Duke
Create game notes for Kansas Jayhawks
Give me broadcast notes for UNC
```

The agent will:
1. Collect team and player statistics
2. Rank statistics nationally and within conferences
3. Generate comprehensive broadcast notes
4. Save the notes to a PDF file with timestamp

## Project Structure

```
GameNotesAgent_Public.py    # Main application file
.env                        # Environment variables (not committed)
.env.example               # Example environment configuration
requirements.txt           # Python dependencies
```

## How It Works

### Agent Architecture

1. **Loader Agent**: Collects data 
   - Fetches team statistics
   - Retrieves player information
   - Gathers quad record data
   - Uses DuckDuckGo for clarifications

2. **Game Notes Writer Agent**: Transforms data into broadcast notes
   - Analyzes statistical rankings
   - Creates compelling narratives
   - Formats for broadcast use
   - Highlights notable achievements

### Data Format

Statistics are formatted as: `value|national_rank|conference_rank`

Example: `85.5|17|2` means:
- Value: 85.5 points per game
- National rank: 17th in the country
- Conference rank: 2nd in conference

### PDF Output

Generated PDFs include:
- Team name as title
- Generation timestamp
- Professionally formatted sections
- Proper spacing and typography
- Blue color scheme for headings

Files are saved as: `GameNotes_TeamName_YYYYMMDD_HHMMSS.pdf`

## Configuration

The application can be configured through environment variables:

- `CBB_API_BASE_URL`: Base URL for API endpoints
- `LLM_KEY`: API key for a large language model

## API Requirements

This project requires:
- Access to a college basketball analytics API
- LLM API access for LLM capabilities

## Features in Detail

### Ranking System

- **Team Stats**: Ranked nationally and by conference
- **Player Stats**: Zone-qualified rankings for shooting percentages
- **Dense Ranking**: Multiple players can share the same rank

### Statistical Zones

Player shooting stats are qualified by zones:
- `rim3s`: At-rim shots (within 3 feet)
- `lane2`: Lane two-pointers
- `paint2`: Paint area shots
- `c3`: Corner threes
- `atb3`: Above-the-break threes

### Qualification Logic

Players must meet minimum attempt thresholds in each zone to receive rankings for that zone's statistics.

## Troubleshooting

### API Connection Issues
- Verify your API keys in `.env`
- Check API base URL is correct

### PDF Generation Fails
- Install reportlab: `pip install reportlab`
- Falls back to text file if PDF fails

### Empty Data Returned
- Verify team name spelling
- Check season/competition ID is valid
- Ensure division ID is correct (1, 2, or 3)

