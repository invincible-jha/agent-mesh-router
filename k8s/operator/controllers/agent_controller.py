"""Agent CR lifecycle controller.

Handles create, update, delete, and periodic reconciliation for
``agents.aumos.ai/v1alpha1`` custom resources.

Responsibilities
----------------
- Create a Kubernetes Deployment and headless Service for each Agent CR.
- Keep the Deployment replicas and image in sync with spec changes.
- Monitor pod health and write status.phase / status.lastHeartbeat.
- Enforce budget constraints by patching replicas to 0 when the hourly cost
  cap is exceeded.
- Emit Kubernetes Events for lifecycle transitions and budget alerts.
- Clean up owned resources when the Agent CR is deleted.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import os
from typing import Any

import kopf
import kubernetes.client as k8s_client
import kubernetes.config as k8s_config
import structlog

from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Kubernetes client initialisation (lazy — configured once per process)
# ---------------------------------------------------------------------------

_k8s_initialised: bool = False


def _ensure_k8s_client() -> None:
    global _k8s_initialised
    if _k8s_initialised:
        return
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()
    _k8s_initialised = True


# ---------------------------------------------------------------------------
# Pydantic spec models — validate the CR spec before acting on it
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    provider: str = "anthropic"
    name: str = "claude-3-5-haiku-20241022"
    max_tokens: int = Field(default=4096, alias="maxTokens")

    model_config = {"populate_by_name": True}


class BudgetSpec(BaseModel):
    max_cost_per_hour: str | None = Field(default=None, alias="maxCostPerHour")
    max_tokens_per_hour: int | None = Field(default=None, alias="maxTokensPerHour")
    alert_threshold: float = Field(default=0.8, alias="alertThreshold")

    model_config = {"populate_by_name": True}

    @field_validator("alert_threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("alertThreshold must be between 0.0 and 1.0")
        return value


class AgentSpec(BaseModel):
    image: str
    capabilities: list[str]
    replicas: int = 1
    model: ModelSpec = Field(default_factory=ModelSpec)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    trust_level: str = Field(default="basic", alias="trustLevel")
    policy: str | None = None
    env: list[dict[str, Any]] = Field(default_factory=list)
    resources: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

OWNER_GROUP = "aumos.ai"
OWNER_VERSION = "v1alpha1"
OWNER_KIND = "Agent"
LABEL_MANAGED_BY = "app.kubernetes.io/managed-by"
LABEL_COMPONENT = "app.kubernetes.io/component"


def _deployment_name(agent_name: str) -> str:
    return f"agent-{agent_name}"


def _service_name(agent_name: str) -> str:
    return f"agent-{agent_name}-svc"


def _build_env_list(spec: AgentSpec) -> list[dict[str, Any]]:
    """Merge spec.env with operator-injected env vars."""
    env: list[dict[str, Any]] = list(spec.env)
    operator_vars = [
        {"name": "AUMOS_AGENT_TRUST_LEVEL", "value": spec.trust_level},
        {"name": "AUMOS_MODEL_PROVIDER", "value": spec.model.provider},
        {"name": "AUMOS_MODEL_NAME", "value": spec.model.name},
        {"name": "AUMOS_MAX_TOKENS", "value": str(spec.model.max_tokens)},
    ]
    existing_names = {e["name"] for e in env}
    for var in operator_vars:
        if var["name"] not in existing_names:
            env.append(var)
    return env


def _build_deployment_manifest(
    name: str,
    namespace: str,
    spec: AgentSpec,
    owner_references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a Deployment manifest dict for the given AgentSpec."""
    labels = {
        "app.kubernetes.io/name": name,
        LABEL_COMPONENT: "agent",
        LABEL_MANAGED_BY: "aumos-operator",
        "aumos.ai/capabilities": ",".join(spec.capabilities[:5]),  # label length limit
    }
    container: dict[str, Any] = {
        "name": "agent",
        "image": spec.image,
        "env": _build_env_list(spec),
        "ports": [{"containerPort": 8080, "name": "http"}],
        "livenessProbe": {
            "httpGet": {"path": "/healthz", "port": 8080},
            "initialDelaySeconds": 10,
            "periodSeconds": 30,
            "failureThreshold": 3,
        },
        "readinessProbe": {
            "httpGet": {"path": "/readyz", "port": 8080},
            "initialDelaySeconds": 5,
            "periodSeconds": 10,
        },
    }
    if spec.resources:
        container["resources"] = spec.resources

    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": _deployment_name(name),
            "namespace": namespace,
            "labels": labels,
            "ownerReferences": owner_references,
        },
        "spec": {
            "replicas": spec.replicas,
            "selector": {"matchLabels": {"app.kubernetes.io/name": name}},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [container],
                    "serviceAccountName": os.getenv("AGENT_SERVICE_ACCOUNT", "default"),
                },
            },
        },
    }


def _build_service_manifest(
    name: str,
    namespace: str,
    owner_references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a headless Service manifest for intra-mesh DNS resolution."""
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": _service_name(name),
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": name,
                LABEL_MANAGED_BY: "aumos-operator",
            },
            "ownerReferences": owner_references,
        },
        "spec": {
            "clusterIP": "None",  # headless service
            "selector": {"app.kubernetes.io/name": name},
            "ports": [{"port": 8080, "name": "http", "targetPort": 8080}],
        },
    }


# ---------------------------------------------------------------------------
# kopf handlers
# ---------------------------------------------------------------------------

@kopf.on.create(OWNER_GROUP, OWNER_VERSION, "agents")
async def on_agent_create(
    name: str,
    namespace: str,
    spec: dict[str, Any],
    meta: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Provision a Deployment + Service when an Agent CR is created."""
    log = logger.bind(agent=name, namespace=namespace, event="create")
    _ensure_k8s_client()

    try:
        agent_spec = AgentSpec.model_validate(spec)
    except Exception as exc:
        log.error("spec_validation_failed", error=str(exc))
        raise kopf.PermanentError(f"Invalid Agent spec: {exc}") from exc

    owner_references = [
        {
            "apiVersion": f"{OWNER_GROUP}/{OWNER_VERSION}",
            "kind": OWNER_KIND,
            "name": name,
            "uid": meta["uid"],
            "controller": True,
            "blockOwnerDeletion": True,
        }
    ]

    apps_api = k8s_client.AppsV1Api()
    core_api = k8s_client.CoreV1Api()

    deployment = _build_deployment_manifest(name, namespace, agent_spec, owner_references)
    service = _build_service_manifest(name, namespace, owner_references)

    try:
        apps_api.create_namespaced_deployment(namespace=namespace, body=deployment)
        log.info("deployment_created", deployment=_deployment_name(name))
    except k8s_client.ApiException as exc:
        if exc.status == 409:
            log.info("deployment_already_exists", deployment=_deployment_name(name))
        else:
            raise

    try:
        core_api.create_namespaced_service(namespace=namespace, body=service)
        log.info("service_created", service=_service_name(name))
    except k8s_client.ApiException as exc:
        if exc.status == 409:
            log.info("service_already_exists", service=_service_name(name))
        else:
            raise

    return {
        "phase": "Pending",
        "readyReplicas": 0,
    }


@kopf.on.update(OWNER_GROUP, OWNER_VERSION, "agents")
async def on_agent_update(
    name: str,
    namespace: str,
    spec: dict[str, Any],
    diff: Any,
    **kwargs: Any,
) -> None:
    """Reconcile the Deployment when the Agent spec changes."""
    log = logger.bind(agent=name, namespace=namespace, event="update")
    _ensure_k8s_client()

    try:
        agent_spec = AgentSpec.model_validate(spec)
    except Exception as exc:
        log.error("spec_validation_failed", error=str(exc))
        raise kopf.PermanentError(f"Invalid Agent spec: {exc}") from exc

    apps_api = k8s_client.AppsV1Api()
    patch_body = {
        "spec": {
            "replicas": agent_spec.replicas,
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "agent",
                            "image": agent_spec.image,
                            "env": _build_env_list(agent_spec),
                        }
                    ]
                }
            },
        }
    }

    try:
        apps_api.patch_namespaced_deployment(
            name=_deployment_name(name),
            namespace=namespace,
            body=patch_body,
        )
        log.info("deployment_patched", replicas=agent_spec.replicas, image=agent_spec.image)
    except k8s_client.ApiException as exc:
        if exc.status == 404:
            log.warning("deployment_missing_during_update", deployment=_deployment_name(name))
        else:
            raise


@kopf.on.delete(OWNER_GROUP, OWNER_VERSION, "agents")
async def on_agent_delete(
    name: str,
    namespace: str,
    **kwargs: Any,
) -> None:
    """Clean up owned Deployment and Service on Agent CR deletion.

    Kubernetes garbage collection via ownerReferences handles most cleanup,
    but we log and handle 404 gracefully in case resources were pre-deleted.
    """
    log = logger.bind(agent=name, namespace=namespace, event="delete")
    _ensure_k8s_client()

    apps_api = k8s_client.AppsV1Api()
    core_api = k8s_client.CoreV1Api()

    for cleanup_fn, resource_name in [
        (
            lambda: apps_api.delete_namespaced_deployment(
                name=_deployment_name(name), namespace=namespace
            ),
            _deployment_name(name),
        ),
        (
            lambda: core_api.delete_namespaced_service(
                name=_service_name(name), namespace=namespace
            ),
            _service_name(name),
        ),
    ]:
        try:
            cleanup_fn()
            log.info("resource_deleted", resource=resource_name)
        except k8s_client.ApiException as exc:
            if exc.status == 404:
                log.debug("resource_already_gone", resource=resource_name)
            else:
                log.error("delete_failed", resource=resource_name, error=str(exc))


@kopf.timer(OWNER_GROUP, OWNER_VERSION, "agents", interval=30.0, idle=10.0)
async def reconcile_agent_status(
    name: str,
    namespace: str,
    spec: dict[str, Any],
    patch: kopf.Patch,
    **kwargs: Any,
) -> None:
    """Periodic reconciliation loop — updates status fields from the live Deployment."""
    log = logger.bind(agent=name, namespace=namespace, event="reconcile")
    _ensure_k8s_client()

    apps_api = k8s_client.AppsV1Api()
    core_api = k8s_client.CoreV1Api()

    try:
        deployment = apps_api.read_namespaced_deployment(
            name=_deployment_name(name), namespace=namespace
        )
    except k8s_client.ApiException as exc:
        if exc.status == 404:
            log.debug("deployment_not_found_during_reconcile")
            return
        raise

    ready_replicas: int = deployment.status.ready_replicas or 0
    desired_replicas: int = spec.get("replicas", 1)

    if ready_replicas == desired_replicas and desired_replicas > 0:
        phase = "Running"
    elif ready_replicas == 0 and desired_replicas == 0:
        phase = "Suspended"
    elif ready_replicas == 0:
        phase = "Pending"
    else:
        phase = "Running"  # partial readiness still considered Running

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    patch.status["phase"] = phase
    patch.status["readyReplicas"] = ready_replicas
    patch.status["lastHeartbeat"] = now_iso

    # Budget enforcement: suspend agent if cost cap exceeded.
    budget_spec = spec.get("budget", {})
    max_cost_str: str | None = budget_spec.get("maxCostPerHour")
    total_cost_str: str | None = kwargs.get("status", {}).get("totalCost")

    if max_cost_str and total_cost_str:
        try:
            max_cost = float(max_cost_str)
            total_cost = float(total_cost_str)
            alert_threshold: float = float(budget_spec.get("alertThreshold", 0.8))

            if total_cost >= max_cost and desired_replicas > 0:
                log.warning(
                    "budget_exceeded_suspending_agent",
                    agent=name,
                    max_cost=max_cost,
                    total_cost=total_cost,
                )
                apps_api.patch_namespaced_deployment(
                    name=_deployment_name(name),
                    namespace=namespace,
                    body={"spec": {"replicas": 0}},
                )
                patch.status["phase"] = "Suspended"

            elif total_cost >= max_cost * alert_threshold:
                log.warning(
                    "budget_alert_threshold_reached",
                    agent=name,
                    threshold=alert_threshold,
                    total_cost=total_cost,
                    max_cost=max_cost,
                )
        except ValueError:
            log.debug("budget_parse_skipped", raw_max=max_cost_str, raw_total=total_cost_str)

    log.debug(
        "reconcile_complete",
        phase=phase,
        ready_replicas=ready_replicas,
    )
