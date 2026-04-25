# Feature: vllm-docker-inference, Property 8: Sağlık Durumu Hesaplama Doğruluğu
"""
Property-based tests for health status calculation.

**Validates: Requirements 8.3**

Property 8: For any set of vLLM engine statuses, if all engines are
reachable the overall status is "healthy"; if at least one engine is
unreachable the overall status is "degraded"; if no engine is reachable
the overall status is "unhealthy"; and an empty list yields "unhealthy".
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from app.models.responses import EngineHealthStatus
from app.services.health import HealthMonitor


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

model_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip() != "")

endpoint_st = st.from_regex(
    r"http://[a-z][a-z0-9\-]{0,19}:[0-9]{4,5}", fullmatch=True
)

status_st = st.sampled_from(["healthy", "unhealthy"])


def engine_health_status_st():
    """Strategy that builds a random EngineHealthStatus."""
    return st.builds(
        EngineHealthStatus,
        model_name=model_name_st,
        status=status_st,
        endpoint=endpoint_st,
    )


engine_statuses_st = st.lists(engine_health_status_st(), min_size=0, max_size=20)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestHealthStatusCalculation:
    """Property 8: Sağlık Durumu Hesaplama Doğruluğu."""

    @given(statuses=engine_statuses_st)
    @settings(max_examples=100)
    def test_overall_status_invariant(self, statuses):
        """Overall status follows the healthy/degraded/unhealthy invariant.

        - Empty list → "unhealthy"
        - All "healthy" → "healthy"
        - All "unhealthy" → "unhealthy"
        - Mixed → "degraded"

        **Validates: Requirements 8.3**
        """
        result = HealthMonitor._calculate_overall_status(statuses)

        if not statuses:
            assert result == "unhealthy"
        else:
            healthy_count = sum(1 for e in statuses if e.status == "healthy")
            if healthy_count == len(statuses):
                assert result == "healthy"
            elif healthy_count == 0:
                assert result == "unhealthy"
            else:
                assert result == "degraded"
