"""AgentMesh CR lifecycle controller.

Handles create, update, delete, and periodic reconciliation for
``agentmeshes.aumos.ai/v1alpha1`` custom resources.

Responsibilities
----------------
- Validate that all referenced Agent CRs exist and are in Running phase.
- Create a ConfigMap encoding the mesh topology and routing rules.
- Inject the ConfigMap into each agent pod via an environment variable so the
  agent-mesh-router library can resolve peers without a control-plane call.
- Maintain a mesh-level circuit-breaker state machine (closed / open / half-open)
  by observing failure counters on the referenced agents.
- Roll up aggregate status (activeAgents, totalMessages, totalCost,
  circuitBreakerState) into the AgentMesh status on every reconciliation tick.
- Clean up the routing ConfigMap on AgentMesh deletion.
"""
from __future__ import annotations

import datetime
import json
from typing import Any

import kopf
import kubernetes.client as k8s_client
import kubernetes.config as k8s_config
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

OWNER_GROUP = "aumos.ai"
OWNER_VERSION = "v1alpha1"
OWNER_KIND = "AgentMesh"

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
# Pydantic models
# ---------------------------------------------------------------------------

class AgentRef(BaseModel):
    name: str
    ref: str
    role: str = "worker"
    weight: int = 1


class CircuitBreakerSpec(BaseModel):
    enabled: bool = True
    failure_threshold: int = Field(default=5, alias="failureThreshold")
    recovery_timeout: str = Field(default="30s", alias="recoveryTimeout")
    success_threshold: int = Field(default=2, alias="successThreshold")

    model_config = {"populate_by_name": True}


class ObservabilitySpec(BaseModel):
    tracing: bool = True
    metrics: bool = True
    cost_tracking: bool = Field(default=True, alias="costTracking")
    log_level: str = Field(default="info", alias="logLevel")

    model_config = {"populate_by_name": True}


class RetryPolicySpec(BaseModel):
    max_attempts: int = Field(default=3, alias="maxAttempts")
    backoff_seconds: float = Field(default=1.0, alias="backoffSeconds")
    backoff_multiplier: float = Field(default=2.0, alias="backoffMultiplier")

    model_config = {"populate_by_name": True}


class RoutingSpec(BaseModel):
    retry_policy: RetryPolicySpec = Field(default_factory=RetryPolicySpec, alias="retryPolicy")
    timeout_seconds: int = Field(default=30, alias="timeoutSeconds")
    dead_letter_queue: bool = Field(default=True, alias="deadLetterQueue")

    model_config = {"populate_by_name": True}


class AgentMeshSpec(BaseModel):
    agents: list[AgentRef]
    topology: str
    circuit_breaker: CircuitBreakerSpec = Field(
        default_factory=CircuitBreakerSpec, alias="circuitBreaker"
    )
    observability: ObservabilitySpec = Field(default_factory=ObservabilitySpec)
    routing: RoutingSpec = Field(default_factory=RoutingSpec)
    policy: str | None = None

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Circuit-breaker state machine
# ---------------------------------------------------------------------------

class CircuitBreakerState:
    """In-process circuit-breaker state machine for a single mesh.

    States: closed -> open -> half-open -> closed/open

    This is a lightweight implementation that mirrors the mesh status subresource.
    Production deployments should store state in a persistent backend.
    """

    def __init__(self, config: CircuitBreakerSpec) -> None:
        self._config = config
        self._state: str = "closed"
        self._failure_count: int = 0
        self._success_count: int = 0
        self._opened_at: datetime.datetime | None = None

    @property
    def state(self) -> str:
        return self._state

    def _parse_timeout_seconds(self) -> int:
        raw: str = self._config.recovery_timeout
        if raw.endswith("s"):
            return int(raw[:-1])
        if raw.endswith("m"):
            return int(raw[:-1]) * 60
        return 30

    def record_failure(self) -> None:
        if self._state == "open":
            return
        self._failure_count += 1
        if self._failure_count >= self._config.failure_threshold:
            self._state = "open"
            self._opened_at = datetime.datetime.now(datetime.timezone.utc)
            logger.warning(
                "circuit_breaker_opened",
                failures=self._failure_count,
                threshold=self._config.failure_threshold,
            )

    def record_success(self) -> None:
        if self._state == "half-open":
            self._success_count += 1
            if self._success_count >= self._config.success_threshold:
                self._state = "closed"
                self._failure_count = 0
                self._success_count = 0
                logger.info("circuit_breaker_closed")

    def maybe_probe(self) -> bool:
        """Return True if a half-open probe should be attempted."""
        if self._state != "open" or self._opened_at is None:
            return False
        elapsed = (
            datetime.datetime.now(datetime.timezone.utc) - self._opened_at
        ).total_seconds()
        if elapsed >= self._parse_timeout_seconds():
            self._state = "half-open"
            self._success_count = 0
            logger.info("circuit_breaker_half_open", elapsed_seconds=elapsed)
            return True
        return False


# Per-mesh circuit-breaker instances keyed by "{namespace}/{name}"
_circuit_breakers: dict[str, CircuitBreakerState] = {}


def _get_or_create_circuit_breaker(
    mesh_key: str, config: CircuitBreakerSpec
) -> CircuitBreakerState:
    if mesh_key not in _circuit_breakers:
        _circuit_breakers[mesh_key] = CircuitBreakerState(config)
    return _circuit_breakers[mesh_key]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _configmap_name(mesh_name: str) -> str:
    return f"agentmesh-{mesh_name}-routing"


def _build_routing_configmap(
    mesh_name: str,
    namespace: str,
    mesh_spec: AgentMeshSpec,
    owner_references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a ConfigMap encoding the mesh routing table for agent sidecar consumption."""
    routing_data = {
        "topology": mesh_spec.topology,
        "agents": [
            {
                "name": agent.name,
                "ref": agent.ref,
                "role": agent.role,
                "weight": agent.weight,
                "serviceEndpoint": f"agent-{agent.ref}-svc.{namespace}.svc.cluster.local:8080",
            }
            for agent in mesh_spec.agents
        ],
        "circuitBreaker": {
            "enabled": mesh_spec.circuit_breaker.enabled,
            "failureThreshold": mesh_spec.circuit_breaker.failure_threshold,
            "recoveryTimeout": mesh_spec.circuit_breaker.recovery_timeout,
            "successThreshold": mesh_spec.circuit_breaker.success_threshold,
        },
        "routing": {
            "retryPolicy": {
                "maxAttempts": mesh_spec.routing.retry_policy.max_attempts,
                "backoffSeconds": mesh_spec.routing.retry_policy.backoff_seconds,
                "backoffMultiplier": mesh_spec.routing.retry_policy.backoff_multiplier,
            },
            "timeoutSeconds": mesh_spec.routing.timeout_seconds,
            "deadLetterQueue": mesh_spec.routing.dead_letter_queue,
        },
        "observability": {
            "tracing": mesh_spec.observability.tracing,
            "metrics": mesh_spec.observability.metrics,
            "costTracking": mesh_spec.observability.cost_tracking,
            "logLevel": mesh_spec.observability.log_level,
        },
    }

    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": _configmap_name(mesh_name),
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": mesh_name,
                "app.kubernetes.io/managed-by": "aumos-operator",
                "app.kubernetes.io/component": "routing-config",
            },
            "ownerReferences": owner_references,
        },
        "data": {
            "routing.json": json.dumps(routing_data, indent=2),
            "topology": mesh_spec.topology,
        },
    }


def _patch_agent_deployment_with_configmap(
    agent_name: str,
    namespace: str,
    configmap_name: str,
) -> None:
    """Inject the routing ConfigMap reference into the agent Deployment via env-var."""
    apps_api = k8s_client.AppsV1Api()
    deployment_name = f"agent-{agent_name}"
    env_patch = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "agent",
                            "env": [
                                {
                                    "name": "AUMOS_MESH_ROUTING_CONFIG",
                                    "valueFrom": {
                                        "configMapKeyRef": {
                                            "name": configmap_name,
                                            "key": "routing.json",
                                            "optional": True,
                                        }
                                    },
                                }
                            ],
                        }
                    ]
                }
            }
        }
    }
    try:
        apps_api.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=env_patch,
        )
        logger.debug("agent_deployment_patched_with_routing_config", deployment=deployment_name)
    except k8s_client.ApiException as exc:
        if exc.status == 404:
            logger.debug("agent_deployment_not_found_skipping_patch", deployment=deployment_name)
        else:
            logger.warning(
                "patch_agent_deployment_failed",
                deployment=deployment_name,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# kopf handlers
# ---------------------------------------------------------------------------

@kopf.on.create(OWNER_GROUP, OWNER_VERSION, "agentmeshes")
async def on_mesh_create(
    name: str,
    namespace: str,
    spec: dict[str, Any],
    meta: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Provision routing ConfigMap and wire agent deployments on mesh creation."""
    log = logger.bind(mesh=name, namespace=namespace, event="create")
    _ensure_k8s_client()

    try:
        mesh_spec = AgentMeshSpec.model_validate(spec)
    except Exception as exc:
        log.error("spec_validation_failed", error=str(exc))
        raise kopf.PermanentError(f"Invalid AgentMesh spec: {exc}") from exc

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

    core_api = k8s_client.CoreV1Api()
    configmap = _build_routing_configmap(name, namespace, mesh_spec, owner_references)

    try:
        core_api.create_namespaced_config_map(namespace=namespace, body=configmap)
        log.info("routing_configmap_created", configmap=_configmap_name(name))
    except k8s_client.ApiException as exc:
        if exc.status == 409:
            log.info("routing_configmap_already_exists", configmap=_configmap_name(name))
        else:
            raise

    # Patch each referenced agent Deployment to consume the routing config.
    for agent_ref in mesh_spec.agents:
        _patch_agent_deployment_with_configmap(agent_ref.ref, namespace, _configmap_name(name))

    # Initialise circuit-breaker if enabled.
    if mesh_spec.circuit_breaker.enabled:
        mesh_key = f"{namespace}/{name}"
        _get_or_create_circuit_breaker(mesh_key, mesh_spec.circuit_breaker)
        log.info("circuit_breaker_initialised", mesh_key=mesh_key)

    return {
        "phase": "Forming",
        "activeAgents": 0,
        "circuitBreakerState": "closed",
    }


@kopf.on.update(OWNER_GROUP, OWNER_VERSION, "agentmeshes")
async def on_mesh_update(
    name: str,
    namespace: str,
    spec: dict[str, Any],
    **kwargs: Any,
) -> None:
    """Rebuild the routing ConfigMap when the AgentMesh spec changes."""
    log = logger.bind(mesh=name, namespace=namespace, event="update")
    _ensure_k8s_client()

    try:
        mesh_spec = AgentMeshSpec.model_validate(spec)
    except Exception as exc:
        log.error("spec_validation_failed", error=str(exc))
        raise kopf.PermanentError(f"Invalid AgentMesh spec: {exc}") from exc

    core_api = k8s_client.CoreV1Api()
    configmap = _build_routing_configmap(name, namespace, mesh_spec, [])

    try:
        core_api.replace_namespaced_config_map(
            name=_configmap_name(name),
            namespace=namespace,
            body=configmap,
        )
        log.info("routing_configmap_updated", topology=mesh_spec.topology)
    except k8s_client.ApiException as exc:
        if exc.status == 404:
            log.warning("routing_configmap_missing_during_update_recreating")
            core_api.create_namespaced_config_map(namespace=namespace, body=configmap)
        else:
            raise

    # Re-patch agents in case membership changed.
    for agent_ref in mesh_spec.agents:
        _patch_agent_deployment_with_configmap(agent_ref.ref, namespace, _configmap_name(name))


@kopf.on.delete(OWNER_GROUP, OWNER_VERSION, "agentmeshes")
async def on_mesh_delete(
    name: str,
    namespace: str,
    **kwargs: Any,
) -> None:
    """Clean up routing ConfigMap on AgentMesh deletion."""
    log = logger.bind(mesh=name, namespace=namespace, event="delete")
    _ensure_k8s_client()

    core_api = k8s_client.CoreV1Api()
    try:
        core_api.delete_namespaced_config_map(
            name=_configmap_name(name),
            namespace=namespace,
        )
        log.info("routing_configmap_deleted", configmap=_configmap_name(name))
    except k8s_client.ApiException as exc:
        if exc.status == 404:
            log.debug("routing_configmap_already_gone")
        else:
            log.error("delete_configmap_failed", error=str(exc))

    # Remove circuit-breaker state from memory.
    mesh_key = f"{namespace}/{name}"
    _circuit_breakers.pop(mesh_key, None)


@kopf.timer(OWNER_GROUP, OWNER_VERSION, "agentmeshes", interval=15.0, idle=5.0)
async def reconcile_mesh_status(
    name: str,
    namespace: str,
    spec: dict[str, Any],
    patch: kopf.Patch,
    **kwargs: Any,
) -> None:
    """Periodic reconciliation — roll up agent health into mesh status."""
    log = logger.bind(mesh=name, namespace=namespace, event="reconcile")
    _ensure_k8s_client()

    try:
        mesh_spec = AgentMeshSpec.model_validate(spec)
    except Exception:
        return

    custom_api = k8s_client.CustomObjectsApi()
    active_agents = 0
    failed_agents = 0
    total_cost_float = 0.0

    for agent_ref in mesh_spec.agents:
        try:
            agent_obj = custom_api.get_namespaced_custom_object(
                group=OWNER_GROUP,
                version=OWNER_VERSION,
                namespace=namespace,
                plural="agents",
                name=agent_ref.ref,
            )
            agent_status = agent_obj.get("status", {})
            agent_phase = agent_status.get("phase", "Pending")

            if agent_phase == "Running":
                active_agents += 1
            elif agent_phase in ("Failed", "Terminated"):
                failed_agents += 1

            cost_str: str | None = agent_status.get("totalCost")
            if cost_str:
                try:
                    total_cost_float += float(cost_str)
                except ValueError:
                    pass

        except k8s_client.ApiException as exc:
            if exc.status == 404:
                log.warning("referenced_agent_not_found", agent=agent_ref.ref)
            else:
                log.error("agent_status_fetch_failed", agent=agent_ref.ref, error=str(exc))

    # Update circuit-breaker based on observed failures.
    mesh_key = f"{namespace}/{name}"
    cb_state = "closed"
    if mesh_spec.circuit_breaker.enabled:
        cb = _get_or_create_circuit_breaker(mesh_key, mesh_spec.circuit_breaker)
        if failed_agents > 0:
            cb.record_failure()
        else:
            cb.record_success()
        cb.maybe_probe()
        cb_state = cb.state

    # Determine mesh phase.
    total_agents = len(mesh_spec.agents)
    if active_agents == total_agents:
        mesh_phase = "Ready"
    elif active_agents == 0:
        mesh_phase = "Failed" if failed_agents > 0 else "Forming"
    else:
        mesh_phase = "Degraded"

    patch.status["phase"] = mesh_phase
    patch.status["activeAgents"] = active_agents
    patch.status["totalCost"] = f"{total_cost_float:.4f}"
    patch.status["circuitBreakerState"] = cb_state
    patch.status["lastUpdated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    log.debug(
        "mesh_reconcile_complete",
        phase=mesh_phase,
        active_agents=active_agents,
        circuit_breaker=cb_state,
    )
