import time
from fastapi import Request
from prometheus_client import Counter, Histogram, make_asgi_app

# Define Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP Requests",
    ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP Request Latency in seconds",
    ["endpoint"]
)

PREDICTION_COUNTER = Counter(
    "credit_risk_predictions_total",
    "Total credit risk predictions made by decision outcome",
    ["decision"]
)

def setup_prometheus_metrics(app):
    """
    Attaches Prometheus metrics middleware and /metrics endpoint to FastAPI.
    """
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        endpoint = request.url.path
        if endpoint != "/metrics":
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                http_status=response.status_code
            ).inc()
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
            
        return response

    # Mount Prometheus metrics scraping endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)