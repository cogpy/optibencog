"""Tests for OpenCog multi-agent orchestration backend."""

import pytest

from optimum_benchmark.backends.opencog.config import OpenCogConfig


class TestOpenCogConfig:
    """Test OpenCog configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OpenCogConfig(model="bert-base-uncased", library="transformers")

        assert config.name == "opencog"
        assert config.num_agents == 1
        assert config.agent_coordination == "sequential"
        assert config.agent_memory_sharing is True
        assert config.reasoning_engine == "pattern_matcher"
        assert config.atomspace_size == 10000
        assert config.enable_attention_allocation is True
        assert config.workflow_strategy == "adaptive"
        assert config.enable_self_optimization is True
        assert config.delegate_backend == "pytorch"
        assert config.load_balancing == "round_robin"

    def test_multi_agent_config(self):
        """Test multi-agent configuration."""
        config = OpenCogConfig(
            model="gpt2",
            library="transformers",
            num_agents=3,
            agent_coordination="parallel",
        )

        assert config.num_agents == 3
        assert config.agent_coordination == "parallel"
        assert len(config.agent_models) == 3
        assert len(config.agent_backends) == 3
        assert len(config.agent_devices) == 3

    def test_custom_agent_models(self):
        """Test custom agent model configuration."""
        models = ["bert-base-uncased", "gpt2", "t5-small"]
        config = OpenCogConfig(
            model="bert-base-uncased",
            library="transformers",
            num_agents=3,
            agent_models=models,
        )

        assert config.agent_models == models

    def test_invalid_num_agents(self):
        """Test that invalid num_agents raises error."""
        with pytest.raises(ValueError, match="num_agents must be at least 1"):
            OpenCogConfig(
                model="bert-base-uncased",
                library="transformers",
                num_agents=0,
            )

    def test_invalid_coordination(self):
        """Test that invalid coordination raises error."""
        with pytest.raises(ValueError, match="agent_coordination must be"):
            OpenCogConfig(
                model="bert-base-uncased",
                library="transformers",
                agent_coordination="invalid",
            )

    def test_mismatched_agent_models_length(self):
        """Test that mismatched agent_models length raises error."""
        with pytest.raises(ValueError, match="agent_models length"):
            OpenCogConfig(
                model="bert-base-uncased",
                library="transformers",
                num_agents=3,
                agent_models=["bert-base-uncased", "gpt2"],  # Only 2 models for 3 agents
            )

    def test_cognitive_load_balancing(self):
        """Test cognitive load balancing configuration."""
        config = OpenCogConfig(
            model="gpt2",
            library="transformers",
            num_agents=2,
            load_balancing="cognitive",
        )

        assert config.load_balancing == "cognitive"

    def test_delegate_backend_config(self):
        """Test delegate backend configuration."""
        delegate_config = {"torch_dtype": "float16"}
        config = OpenCogConfig(
            model="gpt2",
            library="transformers",
            delegate_backend="pytorch",
            delegate_backend_config=delegate_config,
        )

        assert config.delegate_backend == "pytorch"
        assert config.delegate_backend_config == delegate_config


class TestOpenCogBackendStructure:
    """Test OpenCog backend structure without actual model loading."""

    def test_backend_import(self):
        """Test that backend can be imported."""
        from optimum_benchmark.backends.opencog.backend import OpenCogBackend

        assert OpenCogBackend.NAME == "opencog"

    def test_agent_dataclass(self):
        """Test Agent dataclass."""
        from optimum_benchmark.backends.opencog.backend import Agent

        agent = Agent(
            id=0,
            model_name="bert-base-uncased",
            backend_name="pytorch",
            device="cpu",
        )

        assert agent.id == 0
        assert agent.model_name == "bert-base-uncased"
        assert agent.backend_name == "pytorch"
        assert agent.device == "cpu"
        assert agent.model is None
        assert agent.metrics == {}

    def test_atomspace_dataclass(self):
        """Test AtomSpace dataclass."""
        from optimum_benchmark.backends.opencog.backend import AtomSpace

        atomspace = AtomSpace(max_size=100)

        assert atomspace.max_size == 100
        assert len(atomspace.atoms) == 0

        # Test add_atom
        atomspace.add_atom("test_key", {"value": 42})
        assert atomspace.get_atom("test_key") == {"value": 42}

        # Test query_pattern
        atomspace.add_atom("agent_0_state", {"status": "loaded"})
        atomspace.add_atom("agent_1_state", {"status": "initialized"})
        results = atomspace.query_pattern("agent_")
        assert len(results) == 2

    def test_atomspace_eviction(self):
        """Test AtomSpace eviction policy."""
        from optimum_benchmark.backends.opencog.backend import AtomSpace

        atomspace = AtomSpace(max_size=3)

        atomspace.add_atom("key1", "value1")
        atomspace.add_atom("key2", "value2")
        atomspace.add_atom("key3", "value3")

        assert len(atomspace.atoms) == 3

        # Adding 4th atom should evict the first one
        atomspace.add_atom("key4", "value4")

        assert len(atomspace.atoms) == 3
        assert atomspace.get_atom("key1") is None
        assert atomspace.get_atom("key4") == "value4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
