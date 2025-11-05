#!/usr/bin/env python
"""
Verification script for OpenCog multi-agent orchestration backend.

This script performs comprehensive checks to verify the OpenCog backend
implementation is complete and functional.
"""

import sys


def check_imports():
    """Check that all necessary imports work."""
    print("=" * 80)
    print("Checking imports...")
    print("=" * 80)

    try:
        from optimum_benchmark import OpenCogConfig
        from optimum_benchmark.backends.opencog.backend import OpenCogBackend, Agent, AtomSpace

        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def check_config_creation():
    """Check that config can be created with various settings."""
    print("\n" + "=" * 80)
    print("Checking configuration creation...")
    print("=" * 80)

    try:
        from optimum_benchmark.backends.opencog.config import OpenCogConfig

        # Test 1: Basic config with all required fields
        # Note: In real usage, model would be from HF Hub or local path
        # Here we test with minimal valid config to avoid network calls
        config1 = OpenCogConfig(
            model="dummy-model",
            library="transformers",
            task="text-classification",
            model_type="bert",
            device="cpu",
        )
        print(f"✓ Basic config: {config1.num_agents} agent(s)")

        # Test 2: Multi-agent config
        config2 = OpenCogConfig(
            model="dummy-model",
            library="transformers",
            task="text-generation",
            model_type="gpt2",
            device="cpu",
            num_agents=3,
            agent_coordination="parallel",
        )
        print(f"✓ Multi-agent config: {config2.num_agents} agent(s)")

        # Test 3: Custom agent settings
        config3 = OpenCogConfig(
            model="dummy-model",
            library="transformers",
            task="text-classification",
            model_type="bert",
            device="cpu",
            num_agents=2,
            agent_models=["model1", "model2"],
            load_balancing="cognitive",
        )
        print(f"✓ Custom agent config: load_balancing={config3.load_balancing}")

        return True
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_backend_structure():
    """Check backend structure and components."""
    print("\n" + "=" * 80)
    print("Checking backend structure...")
    print("=" * 80)

    try:
        from optimum_benchmark.backends.opencog.backend import (
            Agent,
            AtomSpace,
            OpenCogBackend,
        )

        # Test Agent
        agent = Agent(
            id=0, model_name="test-model", backend_name="pytorch", device="cpu"
        )
        print(f"✓ Agent structure: id={agent.id}, device={agent.device}")

        # Test AtomSpace
        atomspace = AtomSpace(max_size=100)
        atomspace.add_atom("test", {"value": 42})
        result = atomspace.get_atom("test")
        print(f"✓ AtomSpace structure: stored and retrieved {result}")

        # Test pattern matching
        atomspace.add_atom("agent_0", {"status": "ready"})
        atomspace.add_atom("agent_1", {"status": "ready"})
        matches = atomspace.query_pattern("agent_")
        print(f"✓ Pattern matching: found {len(matches)} matches")

        # Test Backend class
        assert OpenCogBackend.NAME == "opencog"
        print(f"✓ Backend class: NAME={OpenCogBackend.NAME}")

        return True
    except Exception as e:
        print(f"✗ Backend structure check failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_examples_exist():
    """Check that example files exist."""
    print("\n" + "=" * 80)
    print("Checking example files...")
    print("=" * 80)

    import os

    examples = [
        "examples/cpu_opencog_bert.yaml",
        "examples/cuda_opencog_gpt2.yaml",
        "examples/opencog_api_example.py",
    ]

    all_exist = True
    for example in examples:
        if os.path.exists(example):
            print(f"✓ {example}")
        else:
            print(f"✗ {example} not found")
            all_exist = False

    return all_exist


def check_documentation():
    """Check that documentation exists."""
    print("\n" + "=" * 80)
    print("Checking documentation...")
    print("=" * 80)

    import os

    docs = ["optimum_benchmark/backends/opencog/README.md"]

    all_exist = True
    for doc in docs:
        if os.path.exists(doc):
            print(f"✓ {doc}")
        else:
            print(f"✗ {doc} not found")
            all_exist = False

    return all_exist


def check_integration():
    """Check integration with main package."""
    print("\n" + "=" * 80)
    print("Checking integration...")
    print("=" * 80)

    try:
        # Check that OpenCogConfig is in __all__
        import optimum_benchmark

        assert "OpenCogConfig" in optimum_benchmark.__all__
        print("✓ OpenCogConfig in __all__")

        # Check that we can import from main package
        from optimum_benchmark import OpenCogConfig

        print("✓ Can import OpenCogConfig from main package")

        # Check backends registry
        from optimum_benchmark.backends import OpenCogConfig as BackendOpenCogConfig

        assert OpenCogConfig is BackendOpenCogConfig
        print("✓ Backend registered correctly")

        return True
    except Exception as e:
        print(f"✗ Integration check failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("OpenCog Backend Verification")
    print("=" * 80 + "\n")

    results = []

    results.append(("Imports", check_imports()))
    results.append(("Configuration", check_config_creation()))
    results.append(("Backend Structure", check_backend_structure()))
    results.append(("Examples", check_examples_exist()))
    results.append(("Documentation", check_documentation()))
    results.append(("Integration", check_integration()))

    # Print summary
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n🎉 All verification checks passed!")
        return 0
    else:
        print("\n⚠️  Some verification checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
