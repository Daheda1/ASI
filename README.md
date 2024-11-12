# FastAPI Pizza sales number API

This is a FastAPI application for serving a pre-trained XGBoost regression model. The application is containerized using Docker and can be deployed by building and running a Docker image.

## Project Structure

```
.
├── app
│   ├── init.py
│   ├── main.py
│   └── xgb_regressor_model.joblib
├── Dockerfile
└── requirements.txt
```

- `app/main.py`: The main FastAPI application file where the model is loaded and API endpoints are defined.
- `app/xgb_regressor_model.joblib`: The pre-trained XGBoost model file, which is loaded by the application.
- `Dockerfile`: Docker configuration to build the FastAPI application with all necessary dependencies.
- `requirements.txt`: Python dependencies for the application.

## Requirements

- Docker installed on your machine.

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Daheda1/ASI
cd ASI/API
```


2.	Build the Docker image
From the root directory, build the Docker image:

```bash
docker build -t myimage .
 ```


3.	Run the Docker container
After building the image, run the container:
```bash
docker run -d --name mycontainer -p 80:80 myimage
```


4.	Access the API
Once the container is running, access the API by going to:
```bash
http://127.0.0.1/predict?order_date=2024-11-12&order_hour=10
 ```