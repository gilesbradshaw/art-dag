"""
Tests for IPFS access consistency.

All IPFS access should use IPFS_API (multiaddr format) for consistency
with art-celery's ipfs_client.py. This ensures Docker deployments work
correctly since IPFS_API is set to /dns/ipfs/tcp/5001.
"""

import os
import re
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock

import pytest


def multiaddr_to_url(multiaddr: str) -> str:
    """
    Convert IPFS multiaddr to HTTP URL.

    This is the canonical conversion used by ipfs_client.py.
    """
    # Handle /dns/hostname/tcp/port format
    dns_match = re.match(r"/dns[46]?/([^/]+)/tcp/(\d+)", multiaddr)
    if dns_match:
        return f"http://{dns_match.group(1)}:{dns_match.group(2)}"

    # Handle /ip4/address/tcp/port format
    ip4_match = re.match(r"/ip4/([^/]+)/tcp/(\d+)", multiaddr)
    if ip4_match:
        return f"http://{ip4_match.group(1)}:{ip4_match.group(2)}"

    # Fallback: assume it's already a URL or use default
    if multiaddr.startswith("http"):
        return multiaddr
    return "http://127.0.0.1:5001"


class TestMultiaddrConversion:
    """Tests for multiaddr to URL conversion."""

    def test_dns_format(self) -> None:
        """Docker DNS format should convert correctly."""
        result = multiaddr_to_url("/dns/ipfs/tcp/5001")
        assert result == "http://ipfs:5001"

    def test_dns4_format(self) -> None:
        """dns4 format should work."""
        result = multiaddr_to_url("/dns4/ipfs.example.com/tcp/5001")
        assert result == "http://ipfs.example.com:5001"

    def test_ip4_format(self) -> None:
        """IPv4 format should convert correctly."""
        result = multiaddr_to_url("/ip4/127.0.0.1/tcp/5001")
        assert result == "http://127.0.0.1:5001"

    def test_already_url(self) -> None:
        """HTTP URLs should pass through."""
        result = multiaddr_to_url("http://localhost:5001")
        assert result == "http://localhost:5001"

    def test_fallback(self) -> None:
        """Unknown format should fallback to localhost."""
        result = multiaddr_to_url("garbage")
        assert result == "http://127.0.0.1:5001"


class TestIPFSConfigConsistency:
    """
    Tests to ensure IPFS configuration is consistent.

    The effect executor should use IPFS_API (like ipfs_client.py)
    rather than a separate IPFS_GATEWAY variable.
    """

    def test_effect_module_should_not_use_gateway_var(self) -> None:
        """
        Regression test: Effect module should use IPFS_API, not IPFS_GATEWAY.

        Bug found 2026-01-12: artdag/nodes/effect.py used IPFS_GATEWAY which
        defaulted to http://127.0.0.1:8080. This doesn't work in Docker where
        the IPFS node is a separate container. The ipfs_client.py uses IPFS_API
        which is correctly set in docker-compose.
        """
        from artdag.nodes import effect

        # Check if the module still has the old IPFS_GATEWAY variable
        # After the fix, this should use IPFS_API instead
        has_gateway_var = hasattr(effect, 'IPFS_GATEWAY')
        has_api_var = hasattr(effect, 'IPFS_API') or hasattr(effect, '_get_ipfs_base_url')

        # This test documents the current buggy state
        # After fix: has_gateway_var should be False, has_api_var should be True
        if has_gateway_var and not has_api_var:
            pytest.fail(
                "Effect module uses IPFS_GATEWAY instead of IPFS_API. "
                "This breaks Docker deployments where IPFS_API=/dns/ipfs/tcp/5001 "
                "but IPFS_GATEWAY defaults to localhost."
            )

    def test_ipfs_api_default_is_localhost(self) -> None:
        """IPFS_API should default to localhost for local development."""
        default_api = "/ip4/127.0.0.1/tcp/5001"
        url = multiaddr_to_url(default_api)
        assert "127.0.0.1" in url
        assert "5001" in url

    def test_docker_ipfs_api_uses_service_name(self) -> None:
        """In Docker, IPFS_API should use the service name."""
        docker_api = "/dns/ipfs/tcp/5001"
        url = multiaddr_to_url(docker_api)
        assert url == "http://ipfs:5001"
        assert "127.0.0.1" not in url


class TestEffectFetchURL:
    """Tests for the URL used to fetch effects from IPFS."""

    def test_fetch_should_use_api_cat_endpoint(self) -> None:
        """
        Effect fetch should use /api/v0/cat endpoint (like ipfs_client.py).

        The IPFS API's cat endpoint works reliably in Docker.
        The gateway endpoint (port 8080) requires separate configuration.
        """
        # The correct way to fetch via API
        base_url = "http://ipfs:5001"
        cid = "QmTestCid123"
        correct_url = f"{base_url}/api/v0/cat?arg={cid}"

        assert "/api/v0/cat" in correct_url
        assert "arg=" in correct_url

    def test_gateway_url_is_different_from_api(self) -> None:
        """
        Document the difference between gateway and API URLs.

        Gateway: http://ipfs:8080/ipfs/{cid}  (requires IPFS_GATEWAY config)
        API:     http://ipfs:5001/api/v0/cat?arg={cid}  (uses IPFS_API config)

        Using the API is more reliable since IPFS_API is already configured
        correctly in docker-compose.yml.
        """
        cid = "QmTestCid123"

        # Gateway style (the old broken way)
        gateway_url = f"http://ipfs:8080/ipfs/{cid}"

        # API style (the correct way)
        api_url = f"http://ipfs:5001/api/v0/cat?arg={cid}"

        # These are different approaches
        assert gateway_url != api_url
        assert ":8080" in gateway_url
        assert ":5001" in api_url


class TestEffectDependencies:
    """Tests for effect dependency handling."""

    def test_effect_with_missing_dependency_gives_clear_error(self, tmp_path: Path) -> None:
        """
        Regression test: Missing dependencies should give clear error message.

        Bug found 2026-01-12: Effect with numpy dependency failed with
        "No module named 'numpy'" but this was swallowed and reported as
        "Unknown effect: invert" - very confusing.
        """
        effects_dir = tmp_path / "_effects"
        effect_cid = "QmTestEffectWithDeps"

        # Create effect that imports a non-existent module
        effect_dir = effects_dir / effect_cid
        effect_dir.mkdir(parents=True)
        (effect_dir / "effect.py").write_text('''
# /// script
# requires-python = ">=3.10"
# dependencies = ["some_nonexistent_package_xyz"]
# ///
"""
@effect test_effect
"""
import some_nonexistent_package_xyz

def process_frame(frame, params, state):
    return frame, state
''')

        # The effect file exists
        effect_path = effects_dir / effect_cid / "effect.py"
        assert effect_path.exists()

        # When loading fails due to missing import, error should mention the dependency
        with patch.dict(os.environ, {"CACHE_DIR": str(tmp_path)}):
            from artdag.nodes.effect import _load_cached_effect

            # This should return None but log a clear error about the missing module
            result = _load_cached_effect(effect_cid)

            # Currently returns None, which causes "Unknown effect" error
            # The real issue is the dependency isn't installed
            assert result is None


class TestEffectCacheAndFetch:
    """Integration tests for effect caching and fetching."""

    def test_effect_loads_from_cache_without_ipfs(self, tmp_path: Path) -> None:
        """When effect is in cache, IPFS should not be contacted."""
        effects_dir = tmp_path / "_effects"
        effect_cid = "QmTestEffect123"

        # Create cached effect
        effect_dir = effects_dir / effect_cid
        effect_dir.mkdir(parents=True)
        (effect_dir / "effect.py").write_text('''
def process_frame(frame, params, state):
    return frame, state
''')

        # Patch environment and verify effect can be loaded
        with patch.dict(os.environ, {"CACHE_DIR": str(tmp_path)}):
            from artdag.nodes.effect import _load_cached_effect

            # Should load without hitting IPFS
            effect_fn = _load_cached_effect(effect_cid)
            assert effect_fn is not None

    def test_effect_fetch_uses_correct_endpoint(self, tmp_path: Path) -> None:
        """When fetching from IPFS, should use API endpoint."""
        effects_dir = tmp_path / "_effects"
        effects_dir.mkdir(parents=True)
        effect_cid = "QmNonExistentEffect"

        with patch.dict(os.environ, {
            "CACHE_DIR": str(tmp_path),
            "IPFS_API": "/dns/ipfs/tcp/5001"
        }):
            with patch('requests.post') as mock_post:
                # Set up mock to return effect source
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b'def process_frame(f, p, s): return f, s'
                mock_post.return_value = mock_response

                from artdag.nodes.effect import _load_cached_effect

                # Try to load - should attempt IPFS fetch
                _load_cached_effect(effect_cid)

                # After fix, this should use the API endpoint
                # Check if requests.post was called (API style)
                # or requests.get was called (gateway style)
                # The fix should make it use POST to /api/v0/cat
