# changepoint-detection

Application and reproducibility materials for Bayesian change-point detection in overdispersed keyword-frequency time series.

# Bayesian Change-Point Detection Application

This repository contains the Streamlit application, dependencies and
anonymised dataset used for Bayesian change-point detection in
overdispersed keyword-frequency time series.

## Application

The application implements and compares:

- A piecewise Negative-Binomial change-point model
- Collapsed reversible-jump Markov chain Monte Carlo
- Posterior inclusion probabilities
- Prior-sensitivity analysis
- Multiple-chain convergence diagnostics
- Replication across seed configurations
- changeforest comparison
- A rolling Random Forest baseline

## Repository Structure

```text
changepoint-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    ├── README.md
    └── your_dataset.csv
```
## Archive format

The dataset is stored as `set2.7z`. The archive is not password-protected
and contains the dataset required by the application. Extract the archive
before running the application unless automatic extraction is implemented
in `app.py`.

To extract it using Python:

```python
import py7zr

with py7zr.SevenZipFile("data/set2.7z", mode="r") as archive:
    archive.extractall(path="data/")
```

## Running in Google Colab

### 1. Clone the repository

```python
!git clone [https://github.com/Rishi-Rakesh/changepoint-detection.git](https://github.com/Rishi-Rakesh/changepoint-detection.git)
%cd changepoint-detection
```

### 2. Install dependencies

```python
!pip install -q -r requirements.txt
```

### 3. Start Streamlit through ngrok

Do not insert an ngrok token directly into a public notebook or repository.
The following version requests the token privately when the cell runs:

```python
from pyngrok import ngrok
from getpass import getpass
import subprocess
import time

ngrok.kill()
ngrok.set_auth_token(getpass("Enter your ngrok authentication token: "))

process = subprocess.Popen(
    [
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
    ]
)

time.sleep(5)

tunnel = ngrok.connect(8501)
print("Application URL:", tunnel.public_url)
```

Open the displayed ngrok URL to access the application.

## Alternative Colab Secret

The ngrok token may be saved in Google Colab Secrets under the name
`NGROK_AUTHTOKEN`. It can then be loaded without typing it into the
notebook:

```python
from google.colab import userdata
from pyngrok import ngrok
import subprocess
import time

ngrok.kill()

token = userdata.get("NGROK_AUTHTOKEN")
if not token:
    raise ValueError("NGROK_AUTHTOKEN was not found in Colab Secrets.")

ngrok.set_auth_token(token)

process = subprocess.Popen(
    [
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
    ]
)

time.sleep(5)

tunnel = ngrok.connect(8501)
print("Application URL:", tunnel.public_url)
```

## Running Locally

Clone the repository:

```bash
git clone [https://github.com/Rishi-Rakesh/changepoint-detection.git](https://github.com/Rishi-Rakesh/changepoint-detection.git)
cd changepoint-detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

The application should open at:

```text
http://localhost:8501
```

## Dataset

The `data/` directory contains the anonymised dataset used in the
dissertation. See `data/README.md` for the variable definitions, temporal
coverage, processing information and conditions of use.

## Security

Never commit any of the following:

- ngrok authentication tokens
- GitHub access tokens
- passwords
- API keys
- private URLs
- `.env` files
- Streamlit secrets
- personal or confidential data

If a secret is committed accidentally, revoke it immediately. Removing it
in a later commit does not necessarily remove it from the repository
history.

## Reproducibility Settings

The dissertation analysis used:

- Four chains per seed replicate
- Seed replicates 0, 1, 2 and 3
- Initial fits with 4,000 warm-up iterations per chain
- Initial fits with 8,000 retained iterations per chain
- Extended fits with 16,000 warm-up iterations per chain
- Extended fits with 32,000 retained iterations per chain
- Expected segment lengths of 30, 70 and 150 weeks
- Minimum segment length of eight weeks
- Posterior inclusion threshold of 0.20

## Licence

No software licence has initially been assigned. The repository is
publicly viewable, but no permission is granted for reuse, modification or
redistribution unless separately agreed.

## Citation

If this application is used in academic work, please cite the accompanying
dissertation and the archived GitHub release.
