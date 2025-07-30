# Kavak Travel Assistant


## 📋 Requirements

- Python 3.10+
- OpenAI API key (optional - mock mode available for testing)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:ranahaani/Kavak-Assignment.git
   cd Kavak
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional)
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## 🚀 Usage

### CLI Interface

**Single Query Mode:**
```bash
python main.py --query "Find me a round-trip to Tokyo in August with Star Alliance airlines only"
```

**Interactive Mode:**
```bash
python main.py
```

### Web Interface

**Launch Streamlit App:**
```bash
# Option 1: Using the main script
python main.py --web

# Option 2: Direct Streamlit command
streamlit run streamlit_app.py
```

The web interface will be available at `http://localhost:8501`


## 📁 Project Structure

```
Kavak/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Data files
│   ├── flights.json       # Mock flight data
│   └── visa_rules.md      # Knowledge base content
└── streamlit_app.py       # Web interface (optional)
```

## 🧪 Sample Outputs

### Flight Search Query
```
Query: "Find me a round-trip to Tokyo in August with Star Alliance airlines only. I want to avoid overnight layovers."

Response:
Here are your flight options:

1. Turkish Airlines (Star Alliance) - Dubai to Tokyo
   Departure: 2024-08-15, Return: 2024-08-30
   Price: $950, Duration: 18h 30m
   Layovers: Istanbul
   Refundable: Yes

2. Singapore Airlines (Star Alliance) - Dubai to Tokyo
   Departure: 2024-08-15, Return: 2024-08-30
   Price: $1150, Duration: 19h 10m
   Layovers: Singapore
   Refundable: Yes
```

### Knowledge Query
```
Query: "What are the visa requirements for UAE citizens visiting Japan?"

Response:
Based on the knowledge base, UAE passport holders can enter Japan visa-free for up to 30 days for tourism. Passport must be valid for at least 6 months beyond the intended stay.
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for LLM operations
- `MODEL_NAME`: LLM model to use (default: gpt-4.1)
- `TEMPERATURE`: LLM temperature setting (default: 0.1)
- `MAX_TOKENS`: Maximum tokens for responses (default: 1000)
