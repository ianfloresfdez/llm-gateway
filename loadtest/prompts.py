"""Prompt pool for the load test: 40 questions, each with two paraphrases.

Traffic follows a Zipf-like popularity (a few questions are asked a lot, most
rarely), which is what makes caching worthwhile in real support / docs bots.
Paraphrases exercise the semantic cache; they are written by hand so that a hit
on them is a true positive.
"""

QUESTIONS: list[tuple[str, str, str]] = [
    ("What is a Kubernetes Pod?", "Explain what a pod is in Kubernetes.", "What is a Pod in Kubernetes?"),
    (
        "What does a Kubernetes Deployment do?",
        "Explain the purpose of a Deployment in Kubernetes.",
        "What is a Deployment used for in Kubernetes?",
    ),
    (
        "What is a Kubernetes Service?",
        "Explain what a Service is in Kubernetes.",
        "What is a Service in Kubernetes?",
    ),
    ("What is a Helm chart?", "Explain what a Helm chart is.", "What exactly is a chart in Helm?"),
    (
        "What is the difference between a ConfigMap and a Secret?",
        "How does a ConfigMap differ from a Secret in Kubernetes?",
        "ConfigMap vs Secret in Kubernetes: what is the difference?",
    ),
    (
        "What is a readiness probe?",
        "Explain what a readiness probe does in Kubernetes.",
        "What is a readiness probe in Kubernetes?",
    ),
    (
        "What is a liveness probe?",
        "Explain what a liveness probe does in Kubernetes.",
        "What is a liveness probe in Kubernetes?",
    ),
    (
        "What does a HorizontalPodAutoscaler do?",
        "Explain the Kubernetes HorizontalPodAutoscaler.",
        "What is the HorizontalPodAutoscaler for?",
    ),
    (
        "What is a PodDisruptionBudget?",
        "Explain what a PodDisruptionBudget is for.",
        "What is a PodDisruptionBudget in Kubernetes?",
    ),
    (
        "What is a Kubernetes namespace?",
        "Explain namespaces in Kubernetes.",
        "What is a namespace in Kubernetes?",
    ),
    (
        "What is kind in Kubernetes tooling?",
        "Explain what the kind tool does.",
        "What is kind (Kubernetes in Docker)?",
    ),
    (
        "What is a container image?",
        "Explain what a container image is.",
        "What exactly is a container image?",
    ),
    (
        "What is a multi-stage Docker build?",
        "Explain multi-stage builds in Docker.",
        "What are multi-stage Dockerfiles?",
    ),
    ("What is Prometheus?", "Explain what Prometheus does.", "What is Prometheus used for?"),
    (
        "What is a Prometheus histogram?",
        "Explain the histogram metric type in Prometheus.",
        "What is a histogram in Prometheus?",
    ),
    ("What is Grafana?", "Explain what Grafana is.", "What is Grafana used for?"),
    ("What is a p95 latency?", "Explain what p95 latency means.", "What does p95 latency mean?"),
    (
        "What is a circuit breaker in distributed systems?",
        "Explain the circuit breaker pattern in distributed systems.",
        "What is the circuit breaker pattern?",
    ),
    (
        "What is exponential backoff?",
        "Explain exponential backoff for retries.",
        "What does exponential backoff mean?",
    ),
    ("What is rate limiting?", "Explain what rate limiting is.", "What does rate limiting mean in APIs?"),
    (
        "What is a semantic cache for LLMs?",
        "Explain semantic caching for LLM responses.",
        "What is semantic caching for language models?",
    ),
    ("What is an embedding?", "Explain what a text embedding is.", "What is a text embedding?"),
    ("What is cosine similarity?", "Explain cosine similarity.", "What does cosine similarity measure?"),
    ("What is Redis?", "Explain what Redis is.", "What is Redis used for?"),
    ("What is CI/CD?", "Explain what CI/CD means.", "What does CI/CD stand for and mean?"),
    ("What is GitHub Actions?", "Explain what GitHub Actions is.", "What is GitHub Actions used for?"),
    ("What is an API gateway?", "Explain what an API gateway does.", "What is the role of an API gateway?"),
    ("What is a reverse proxy?", "Explain what a reverse proxy is.", "What does a reverse proxy do?"),
    ("What is OpenTelemetry?", "Explain what OpenTelemetry is.", "What is OpenTelemetry used for?"),
    (
        "What is a token in a language model?",
        "Explain what tokens are in LLMs.",
        "What is a token for an LLM?",
    ),
    (
        "What is a context window?",
        "Explain an LLM context window.",
        "What does context window mean for LLMs?",
    ),
    ("What is quantization of a model?", "Explain model quantization.", "What does quantizing a model mean?"),
    ("What is Ollama?", "Explain what Ollama is.", "What is Ollama used for?"),
    ("What is FastAPI?", "Explain what FastAPI is.", "What is FastAPI used for?"),
    ("What is asyncio in Python?", "Explain Python's asyncio.", "What is asyncio used for in Python?"),
    (
        "What is a rolling update?",
        "Explain rolling updates in Kubernetes.",
        "What is a rolling update deployment?",
    ),
    ("What is a Kubernetes node?", "Explain what a node is in Kubernetes.", "What is a node in Kubernetes?"),
    ("What is etcd?", "Explain what etcd is.", "What is etcd used for in Kubernetes?"),
    (
        "What is a ServiceMonitor?",
        "Explain what a ServiceMonitor is.",
        "What is a ServiceMonitor in the Prometheus Operator?",
    ),
    (
        "What is idempotency in APIs?",
        "Explain idempotency for API requests.",
        "What does idempotent mean for an API?",
    ),
]

SYSTEM = "Answer in at most two short sentences."
