#!/usr/bin/env bash
set -euo pipefail

# Deploy an OpenAI-compatible vLLM endpoint for gpt-oss-20b.
# This is intended for a single Kubernetes node with 2 GPUs.

NAMESPACE="${NAMESPACE:-llm-inference}"
APP_NAME="${APP_NAME:-gpt-oss-20b-vllm}"
SERVICE_NAME="${SERVICE_NAME:-gpt-oss-20b-svc}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
MODEL_ID="${MODEL_ID:-openai/gpt-oss-20b}"
PORT="${PORT:-8000}"

# GPU/model runtime settings for 2x A40 node.
GPU_COUNT="${GPU_COUNT:-2}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
DTYPE="${DTYPE:-auto}"

# Optional node targeting.
GPU_NODE_LABEL_KEY="${GPU_NODE_LABEL_KEY:-}"
GPU_NODE_LABEL_VALUE="${GPU_NODE_LABEL_VALUE:-}"

# Optional Hugging Face token, needed for gated/private models.
HF_TOKEN="${HF_TOKEN:-}"
HF_SECRET_NAME="${HF_SECRET_NAME:-hf-token}"

echo "[1/5] Verifying kubectl connectivity..."
kubectl version --client >/dev/null
kubectl cluster-info >/dev/null

echo "[2/5] Ensuring namespace ${NAMESPACE} exists..."
kubectl get ns "${NAMESPACE}" >/dev/null 2>&1 || kubectl create namespace "${NAMESPACE}"

if [[ -n "${HF_TOKEN}" ]]; then
  echo "[3/5] Creating/updating secret ${HF_SECRET_NAME}..."
  kubectl -n "${NAMESPACE}" create secret generic "${HF_SECRET_NAME}" \
    --from-literal=token="${HF_TOKEN}" \
    --dry-run=client -o yaml | kubectl apply -f -
else
  echo "[3/5] No HF_TOKEN supplied. Skipping Hugging Face secret creation."
fi

echo "[4/5] Applying Deployment and Service manifests..."

NODE_SELECTOR_BLOCK=""
if [[ -n "${GPU_NODE_LABEL_KEY}" && -n "${GPU_NODE_LABEL_VALUE}" ]]; then
  NODE_SELECTOR_BLOCK=$(cat <<EOF
      nodeSelector:
        ${GPU_NODE_LABEL_KEY}: "${GPU_NODE_LABEL_VALUE}"
EOF
)
fi

HF_ENV_BLOCK=""
if [[ -n "${HF_TOKEN}" ]]; then
  HF_ENV_BLOCK=$(cat <<EOF
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: ${HF_SECRET_NAME}
              key: token
EOF
)
fi

cat <<EOF | kubectl -n "${NAMESPACE}" apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${APP_NAME}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${APP_NAME}
  template:
    metadata:
      labels:
        app: ${APP_NAME}
    spec:
${NODE_SELECTOR_BLOCK}
      containers:
      - name: vllm
        image: ${VLLM_IMAGE}
        imagePullPolicy: IfNotPresent
        args:
        - --model
        - ${MODEL_ID}
        - --host
        - 0.0.0.0
        - --port
        - "${PORT}"
        - --tensor-parallel-size
        - "${TENSOR_PARALLEL}"
        - --max-model-len
        - "${MAX_MODEL_LEN}"
        - --gpu-memory-utilization
        - "${GPU_MEMORY_UTIL}"
        - --dtype
        - ${DTYPE}
        ports:
        - containerPort: ${PORT}
          name: http
        env:
        - name: HF_HOME
          value: /cache/hf
${HF_ENV_BLOCK}
        resources:
          requests:
            cpu: "8"
            memory: "32Gi"
            nvidia.com/gpu: "${GPU_COUNT}"
          limits:
            cpu: "16"
            memory: "64Gi"
            nvidia.com/gpu: "${GPU_COUNT}"
        volumeMounts:
        - name: model-cache
          mountPath: /cache
        - name: dshm
          mountPath: /dev/shm
        readinessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          failureThreshold: 30
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 90
          periodSeconds: 20
      volumes:
      - name: model-cache
        emptyDir: {}
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 8Gi
---
apiVersion: v1
kind: Service
metadata:
  name: ${SERVICE_NAME}
spec:
  type: ClusterIP
  selector:
    app: ${APP_NAME}
  ports:
  - name: http
    port: ${PORT}
    targetPort: http
EOF

echo "[5/5] Waiting for rollout and printing connection details..."
kubectl -n "${NAMESPACE}" rollout status deployment/"${APP_NAME}" --timeout=20m
kubectl -n "${NAMESPACE}" get pods -l app="${APP_NAME}" -o wide

echo
echo "LLM endpoint is ready (inside cluster):"
echo "  http://${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local:${PORT}/v1"
echo
echo "Use this with your runner script:"
echo "  python run_pywhyllm_on_triples.py --backend openai_compatible --model ${MODEL_ID} --api-base http://${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local:${PORT}/v1 --api-key dummy"
