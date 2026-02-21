CIS/
├── README.md                 # Project overview, setup, run instructions
├── requirements.txt          # pip install -r requirements.txt
├── .gitignore                # Ignore data/, __pycache__, .env
├── setup.sh                  # bash setup.sh (installs deps, downloads models)
│
├── data/                     # All data handling
│   ├── raw/                  # synthetic_dataset.csv (git ignore large files)
│   ├── processed/            # train.csv, test.csv
│   └── emissions_factors.py  # Dict of factors (0.82 electricity, etc.)
│
├── models/                   # ML training & saving
│   ├── train_models.py       # Generate data, train, evaluate, save
│   ├── regressor.pkl         # Saved RandomForestRegressor
│   ├── classifier.pkl        # Saved RandomForestClassifier
│   └── utils.py              # Feature lists, CO2 calc function
│
├── backend/                  # Flask API (Member 2)
│   ├── app.py                # Main Flask app with /predict endpoint
│   ├── model_loader.py       # Loads .pkl files
│   └── tests_api.py          # Test predictions
│
├── frontend/                 # Streamlit UI (Member 3)
│   ├── app.py                # Main Streamlit interface
│   ├── api_client.py         # Calls backend /predict
│   └── components/           # Reusable UI parts (charts, recs)
│       └── recommendations.py
│
├── intelligence/             # Recs, score, what-if (Member 4)
│   ├── carbon_score.py       # 0-100 scale logic
│   ├── recommendations.py    # Rule-based tips
│   └── what_if_sim.py        # Simulate changes
│
├── docs/                     # Presentation & notes
│   ├── pitch_deck.pdf        # 5-slide Google Slides export
│   └── architecture.png      # Draw.io diagram (Streamlit→Flask→ML)
│
└── notebooks/                # Exploration (optional, Jupyter for viz)
    └── eda.ipynb             # Data preview, quick tests