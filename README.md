📌 About the Project

This project analyzes 18,447 AI research papers with code from academic and open-source repositories. Each paper was evaluated using Gemini AI for code quality, usability, and implementation maturity. The results are visualized through an interactive Streamlit dashboard and a Power BI report.

The project answers the question:
> "Do popular AI research projects actually produce high-quality, production-ready code?"
>
>
> 🔗 Links
📊 Power BI Dashboard (Online): https://app.powerbi.com/view?r=eyJrIjoiNjc3NWY2YTUtNTlkOC00ZWFlLThjYTgtMTI4NzNmNzk3YTU4IiwidCI6ImQ5MzRiZjA4LTk1ZjQtNGNjZC1hZDFiLWYzODM3M2E5NDM1NyIsImMiOjl9
> 🌐 Developer Portfolio: https://www.khayal-hajiyev.com
>
> 
> 🛠️ Technologies Used
> Python: Data processing, AI scoring pipeline
> PostgreSQL 17: Main database — 18,447 papers 
> Gemini AI: Scoring and evaluating research projects
> Sentence Transformers: Semantic search (all-MiniLM-L6-v2)
> Streamlit: Interactive web dashboard
> Docker Containerized deployment
> Power BI: Business intelligence dashboard
> Pandas / NumPy:  Statistical analysis
> Matplotlib: Plotly, Data visualization


📊 Dashboard Pages

🏠 Overview: KPI cards, tier distribution, top topics, languages
🔍 Semantic Search: Embedding-based similarity search across all papers
🤖 Gemini Tribunal: Leaderboard of AI-scored projects
📊 Analytics: Quality vs popularity analysis by tier
🔬 Deep Analysis: NumPy + Pandas + Matplotlib statistical deep dive
⚠️ Risk Explorer: Failure categories, treemap, deployment patterns
🌍 Global Map: Choropleth world map of AI research by country


🗂️ Project Structure

📦 repository
├── streamlit_app.py         Main Streamlit application
├── Dockerfile               Docker image for Streamlit app
├── docker-compose.yml       Orchestrates app + PostgreSQL containers
├── requirements.txt         Python dependencies
├── db_init/
│   └── dump.sql             PostgreSQL database dump (not included — see below)
└── README.md


🚀 How to Run with Docker

Prerequisites
- Docker Desktop installed
- PostgreSQL dump file (`dump.sql`)


Steps
1. Clone the repository**
git clone https://github.com/Khayal-Developer/From-Academic-Papers-to-Production-Code-An-AI-Powered-Analysis-of-Research-Implementation.git
cd From-Academic-Papers-to-Production-Code-An-AI-Powered-Analysis-of-Research-Implementation

2. Add the database dump**
Place your dump.sql file inside the db_init/ folder

3. Build and run**
docker-compose build
docker-compose up

4. Open in browser**
http://localhost:8501


🗄️ Database Notes
The `dump.sql` file (~919MB) is not included in this repository due to GitHub's file size limit.
To acquire the file, visit Google Drive link (Note: dump.sql must be inside db_init folder) --> https://drive.google.com/drive/folders/1ZTYO4M5ymisiuL9Csyz5Kpxvxhl7zTwK?usp=drive_link

To export the database from your local PostgreSQL
pg_dump -U postgres -d knowledge_engine_db -f db_init/dump.sql


📋 Key Findings
- 18,447 AI research projects analyzed
- Only 4.7% of projects reach Star tier (1000+ stars)
- Correlation between stars and AI quality = 0.11 — popularity ≠ quality
- Popular Projects (100-999 stars)** are the most consistent in quality (lowest STD)
- Star Projects** score highest on average (73.6/100) but are almost as unpredictable as Hidden Gems


⚖️ License & Usage
> This project and the Power BI dashboard are intended for educational purposes only.
> Developed as part of the **ABB Tech Academy internship program.
>
> 
> 👤 Developer
Khayal Hajiyev
🌐 https://www.khayal-hajiyev.com


