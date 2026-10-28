#!/usr/bin/env bash
# Create the local kind cluster and deploy monitoring + llm-gateway.
# Usage: scripts/kind-up.sh [--no-monitoring]
set -euo pipefail
cd "$(dirname "$0")/.."

KIND=${KIND:-$(command -v kind || echo .tools/kind.exe)}
HELM=${HELM:-$(command -v helm || echo .tools/helm.exe)}
CLUSTER=llm-gateway
IMAGE=llm-gateway:dev
MONITORING=1
[[ "${1:-}" == "--no-monitoring" ]] && MONITORING=0

if ! "$KIND" get clusters | grep -qx "$CLUSTER"; then
  "$KIND" create cluster --config deploy/kind/kind-config.yaml
fi
kubectl config use-context "kind-$CLUSTER"

if [[ $MONITORING == 1 ]]; then
  "$HELM" repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
  "$HELM" repo update prometheus-community >/dev/null
  "$HELM" upgrade --install kps prometheus-community/kube-prometheus-stack \
    --namespace monitoring --create-namespace \
    -f deploy/kind/kube-prometheus-stack-values.yaml --wait --timeout 10m

  # metrics-server feeds the HorizontalPodAutoscaler with CPU usage.
  kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
  kubectl -n kube-system patch deployment metrics-server --type=json \
    -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' \
    2>/dev/null || true
  VALUES=(-f deploy/helm/llm-gateway/values-kind.yaml)
else
  VALUES=(-f deploy/helm/llm-gateway/values-kind.yaml --set metrics.serviceMonitor.enabled=false
          --set metrics.grafanaDashboard.enabled=false --set autoscaling.enabled=false)
fi

# kind nodes cannot see the host's local images: build, then copy the image into every node.
docker build -t "$IMAGE" .
"$KIND" load docker-image "$IMAGE" --name "$CLUSTER"

"$HELM" upgrade --install llm-gateway deploy/helm/llm-gateway \
  --namespace llm --create-namespace "${VALUES[@]}" --wait --timeout 10m

kubectl -n llm get pods -o wide
echo
echo "Gateway:    http://localhost:8090"
[[ $MONITORING == 1 ]] && echo "Grafana:    http://localhost:3001  (dashboard 'LLM Gateway')" \
                       && echo "Prometheus: http://localhost:9091"
