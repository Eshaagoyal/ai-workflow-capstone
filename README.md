# AAVAIL Revenue Forecasting Service

An enterprise AI workflow microservice built to ingest monthly streaming transaction data, engineer time-series features, train country-specific regression models, and serve continuous 30-day revenue forecasts via a containerized REST API.

---

## Project Structure

```text
├── cslib.py            # Data ingestion, schema standardization, and feature engineering
├── model.py            # Model training, inference, and persistence pipelines
├── logger.py           # Production logging for API queries and training iterations
├── app.py              # Flask REST API exposing /predict, /train, and /logs endpoints
├── run-tests.py        # Master test runner script
├── unittests/
│   ├── ApiTests.py     # Functional integration tests for API endpoints
│   └── ModelTests.py   # Unit tests for training, persistence, and inference
├── Dockerfile          # Container configuration for API deployment
├── requirements.txt    # Production Python dependencies
└── README.md           # Deployment and operational instructions
