# AumOS Kubernetes CRDs and Operator

Kubernetes-native lifecycle management for AumOS agents running in the mesh router.

## Overview

This directory contains three Custom Resource Definitions (CRDs) and a Python operator built with [kopf](https://kopf.readthedocs.io/) that manages the full lifecycle of AumOS agents in a Kubernetes cluster.

| Resource | Short name | Purpose |
|---|---|---|
| `Agent` | `ag` | Single agent workload — provisions a Deployment + headless Service |
| `AgentMesh` | `amesh` | Multi-agent topology — wires agents together with routing config and circuit-breaking |
| `AgentPolicy` | `apol` | Compliance and cost guardrails applied to agents and meshes |

## Directory Layout

```
k8s/
  README.md
  crds/
    agent.yaml          — Agent CRD schema
    agentmesh.yaml      — AgentMesh CRD schema
    agentpolicy.yaml    — AgentPolicy CRD schema
  examples/
    simple-agent.yaml       — Single agent, basic capabilities
    agent-mesh.yaml         — Three-agent hierarchical mesh
    agent-with-policy.yaml  — Agent with EU AI Act compliance policy
    full-deployment.yaml    — Production deployment with RBAC, secrets, mesh, policy
  operator/
    Dockerfile          — Python 3.12-slim image
    requirements.txt    — kopf, kubernetes, pydantic, structlog, OTel
    main.py             — Operator entry point, startup/cleanup hooks
    controllers/
      __init__.py
      agent_controller.py   — Agent CR handlers (create, update, delete, timer)
      mesh_controller.py    — AgentMesh CR handlers (create, update, delete, timer)
```

## Prerequisites

- Kubernetes 1.26+
- `kubectl` configured against your cluster
- Container registry access for `ghcr.io/aumos-ai/agent-base` (or your own image)

## Install CRDs

```bash
kubectl apply -f k8s/crds/agent.yaml
kubectl apply -f k8s/crds/agentmesh.yaml
kubectl apply -f k8s/crds/agentpolicy.yaml
```

Verify installation:

```bash
kubectl get crds | grep aumos.ai
# agents.aumos.ai
# agentmeshes.aumos.ai
# agentpolicies.aumos.ai
```

## Run the Operator

### Local development

```bash
cd k8s/operator
pip install -r requirements.txt

# Runs against the current kubectl context — watches all namespaces.
python -m kopf run main.py --dev
```

### In-cluster deployment

Build and push the image:

```bash
docker build -t ghcr.io/aumos-ai/aumos-operator:latest k8s/operator/
docker push ghcr.io/aumos-ai/aumos-operator:latest
```

Deploy using the RBAC resources defined in `examples/full-deployment.yaml`:

```bash
kubectl apply -f k8s/examples/full-deployment.yaml
```

Or deploy the operator Pod directly:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aumos-operator
  namespace: aumos-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: aumos-operator
  template:
    metadata:
      labels:
        app: aumos-operator
    spec:
      serviceAccountName: aumos-operator
      containers:
        - name: operator
          image: ghcr.io/aumos-ai/aumos-operator:latest
          env:
            - name: WATCH_NAMESPACE
              value: ""           # empty = all namespaces
            - name: LOG_LEVEL
              value: "INFO"
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 30
```

## Quick Start

### Deploy a single agent

```bash
# Create the provider secret first.
kubectl create secret generic aumos-provider-secrets \
  --from-literal=anthropic-api-key="$ANTHROPIC_API_KEY" \
  -n aumos-system

kubectl apply -f k8s/examples/simple-agent.yaml
kubectl get agents -n aumos-system -w
```

### Deploy a three-agent hierarchical mesh

```bash
kubectl apply -f k8s/examples/agent-mesh.yaml
kubectl get agentmeshes,agents -n aumos-system
```

### Apply a compliance policy

```bash
kubectl apply -f k8s/examples/agent-with-policy.yaml
kubectl describe agentpolicy eu-compliant-policy -n aumos-system
```

### Full production deployment

```bash
# Edit secrets in full-deployment.yaml before applying.
kubectl apply -f k8s/examples/full-deployment.yaml
kubectl get agents,agentmeshes,agentpolicies -n aumos-prod
```

## CRD Reference

### Agent spec fields

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | string | required | Container image for the agent workload |
| `capabilities` | string[] | required | Capability identifiers the agent exposes |
| `replicas` | integer | 1 | Pod replicas (0 = suspended) |
| `model.provider` | enum | — | `openai`, `anthropic`, `google`, `azure`, `local` |
| `model.name` | string | — | Model name as accepted by the provider API |
| `model.maxTokens` | integer | 4096 | Max tokens per request |
| `budget.maxCostPerHour` | string | — | USD cap; agent is suspended when exceeded |
| `budget.alertThreshold` | number | 0.8 | Fraction of budget that triggers a warning event |
| `trustLevel` | enum | basic | `untrusted`, `basic`, `verified`, `trusted` |
| `policy` | string | — | Name of an AgentPolicy CR in the same namespace |

### AgentMesh topology options

| Topology | Behaviour |
|---|---|
| `sequential` | Tasks flow through agents in the listed order |
| `parallel` | All agents execute concurrently; results are merged |
| `hierarchical` | Leader decomposes task and delegates to workers |
| `competitive` | Multiple agents race; fastest valid result wins |
| `consensus` | Agents vote; majority agreement required to proceed |

### AgentPolicy compliance frameworks

| Framework | Controls enforced |
|---|---|
| `eu-ai-act` | Model allowlist, audit logging, transparency markers |
| `gdpr` | Data residency constraints, PII pattern blocking |
| `hipaa` | PHI pattern blocking, full audit log, data encryption flags |
| `soc2` | Rate limiting, audit trail, incident webhook |
| `iso27001` | Access control validation, anomaly alerting |
| `nist-ai-rmf` | Risk category tagging, explainability requirements |

## Operator Architecture

The operator is structured around [kopf](https://kopf.readthedocs.io/) handlers:

- **`agent_controller.py`** — `on.create`, `on.update`, `on.delete`, and a 30-second `timer` for status reconciliation. The timer enforces budget limits by setting `replicas: 0` when the cost cap is exceeded.
- **`mesh_controller.py`** — `on.create`, `on.update`, `on.delete`, and a 15-second `timer`. Maintains an in-process circuit-breaker state machine (closed / open / half-open) and rolls up aggregate status from all referenced agents.

Each reconciliation loop uses [Pydantic v2](https://docs.pydantic.dev/latest/) to validate CR specs before taking action, ensuring the operator never acts on malformed input.

## Observability

When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, the operator exports traces to an OTLP-compatible collector. Each reconciliation loop creates a span tagged with the CR name and namespace.

Prometheus metrics are exposed at `:8080/metrics` (kopf built-in) and include:

- `kopf_objects_total` — objects managed by kind
- `kopf_events_total` — handler invocations by type and outcome

## Extension Points

- **Custom provider secrets**: Mount secrets under any key and reference them in `spec.env[].valueFrom.secretKeyRef`.
- **External policy store**: Replace `AgentPolicy` with a webhook admission controller that validates against an external OPA/Rego policy bundle.
- **Persistent cost tracking**: Implement a cost aggregation sidecar and update `status.totalCost` via the Kubernetes status subresource API.
