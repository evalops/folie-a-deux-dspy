# Folie à Deux: Anchored Consensus Co-Training for Multi-Agent Language Models

[![Requirements: Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DSPy](https://img.shields.io/badge/DSPy-Compatible-green.svg)](https://github.com/stanfordnlp/dspy)
[![Ollama](https://img.shields.io/badge/Ollama-Required-orange.svg)](https://ollama.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A framework for studying **anchored consensus co-training** in multi-agent LMs. We implement a two-agent **post-training** scheme where the optimization objective interpolates between truth preservation and inter-agent agreement:

**`R(α) = α·Truth + (1−α)·Agreement`**

enabling empirical study of the consensus-vs-truth trade-off.

## 🔬 Research Motivation

This framework addresses fundamental questions in **multi-agent reinforcement learning** and **AI safety research** by investigating consensus formation in co-evolutionary language model systems. Drawing from psychological literature on shared delusions ("folie à deux"), we study how mutual agreement optimization affects factual accuracy and emergent behaviors.

### Research Focus

This implementation builds on established research areas:

- **Two-Agent α-Anchored Training**: Specific application of α-parameterized reward blending (`R(α) = α·Truth + (1-α)·Agreement`) in a two-agent setting
- **Systematic Trade-off Analysis**: Empirical study of Pareto curves between consensus formation and truth preservation
- **DSPy Integration**: Application of MIPROv2 prompt optimization within multi-agent co-training loops
- **Factual Verification Domain**: Testing agreement vs. truth dynamics in truth-evaluable tasks

**Note**: Multi-agent post-training, sycophancy research, and echo chamber studies are established areas. This work applies existing concepts in a specific controlled formulation.

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- Ollama running locally (default: `http://localhost:11434`)
- A compatible language model (default: `llama3.1:8b`)

### Installation

```bash
# Clone the repository
git clone https://github.com/evalops/folie-a-deux-dspy.git
cd folie-a-deux-dspy

# Set up virtual environment and install dependencies
make setup

# Install and setup Ollama with required model:
# macOS: brew install ollama && ollama pull llama3.1:8b && ollama serve
# Linux: curl -fsSL https://ollama.ai/install.sh | sh && ollama pull llama3.1:8b && ollama serve
# Windows: Download from https://ollama.ai/download/windows
# Verify API: curl http://localhost:11434/api/tags
```

### Basic Usage

```bash
# Run with default settings (pure agreement optimization, α=0.0)
make run

# Run with truth anchoring (α=0.1)
make run-alpha

# Custom configuration
MODEL=ollama_chat/mistral:7b ALPHA=0.05 ROUNDS=10 make run
```

## 📊 Results

The experiment tracks several metrics across training rounds:

- **Accuracy A/B**: Individual verifier accuracy on ground truth
- **Agreement (Dev)**: How often verifiers agree on held-out data
- **Agreement (Train)**: How often verifiers agree on training data

Example output:
```
[round 1] accA=0.773 accB=0.727 agree_dev=0.818 agree_train=0.850
[round 2] accA=0.818 accB=0.773 agree_dev=0.864 agree_train=0.883
[round 3] accA=0.864 accB=0.818 agree_dev=0.909 agree_train=0.917
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `ollama_chat/llama3.1:8b` | LLM model to use |
| `API_BASE` | `http://localhost:11434` | Ollama API endpoint |
| `ALPHA` | `0.0` | Truth anchoring weight (0=pure agreement, 1=pure truth) |
| `ROUNDS` | `6` | Number of iterative training rounds |

### Makefile Targets

```bash
make setup      # Install dependencies
make run        # Run experiment with default settings (α=0.0, pure agreement)
make run-alpha  # Run with truth anchoring (α=0.1)
make sweep      # Run α parameter sweep for Pareto analysis
make fmt        # Format code with ruff
```

## 🔬 Experimental Design

### Dataset Configuration

- **Supervised Set**: 30 factual claims with verified ground truth labels for evaluation
- **Co-training Set**: 98 unlabeled claims (7× repetition of 14 base claims) for iterative agreement optimization
- **Domain Coverage**: Balanced statements across natural sciences, geography, and history

### Co-evolutionary Training Protocol

1. **Initialization Phase**: Bootstrap Verifier A using supervised learning on ground truth via MIPROv2 prompt optimization
2. **Iterative Co-training Phase**: For each training round t:
   - Optimize Verifier A using agreement metric with Verifier B + α-weighted truth anchoring
   - Optimize Verifier B using agreement metric with Verifier A + α-weighted truth anchoring
   - Evaluate consensus formation and truth preservation on held-out data

### Evaluation Metrics

- **Truth Accuracy**: Classification accuracy against verified ground truth labels
- **Inter-Verifier Agreement**: Consensus rate between co-trained verifiers
- **Composite Objective**: `L(α) = (1-α) × L_agreement + α × L_truth` for α ∈ [0,1]

## 📈 Research Questions

This framework enables empirical investigation of:

1. **Consensus Dynamics**: Rate and stability of agreement convergence in co-evolutionary systems
2. **Truth Preservation vs. Agreement Trade-offs**: Impact of mutual optimization on factual accuracy
3. **Echo Chamber Formation**: Conditions under which feedback loops amplify biases or errors
4. **Critical Anchoring Thresholds**: Minimum α values required to prevent truth degradation
5. **Emergent Coordination**: Self-organizing behaviors in multi-agent consensus formation

## 🛠️ Technical Details

### Architecture

- **Framework**: DSPy for prompt/program optimization within training loop
- **Backend**: Ollama for local LLM inference
- **Optimization**: **DSPy MIPROv2** for prompt optimization (no LLM weight updates)
- **Post-Training**: α-anchored agreement optimization via iterative program refinement

### Key Components

- `VerifyClaim`: DSPy signature for factual verification tasks
- `Verifier`: Agent module with configurable reasoning strategies
- `agreement_metric_factory`: Inter-agent consensus measurement
- `blended_metric_factory`: α-weighted truth-agreement composite reward

## 📚 Related Work

This research directly builds on several established areas:

- **Multi-Agent Post-Training**: Multi-agent preference optimization (Chen et al., NeurIPS 2024); MACPO for contrastive learning (Liu et al., ICLR 2024)
- **Anchored Preference Optimization**: APO and BAPO for controlled preference training (D'Oosterlinck et al., 2024)
- **Sycophancy Research**: Constitutional AI (Bai et al., 2022); RLHF alignment failures (Casper et al., 2023)
- **Echo Chamber Effects**: Feedback loops in RLHF (Gao et al., ICML 2023); social bias amplification (Santurkar et al., 2023)
- **Group Preference Optimization**: Multi-stakeholder alignment (Bakker et al., 2024); democratic AI (Baumann et al., 2024)
- **Iterative Consensus Methods**: Self-Consistency (Wang et al., 2023); Mixture-of-Agents (Wang et al., 2024)

**Distinction**: This work combines α-parameterized truth anchoring with two-agent co-training in a systematic study of the consensus-truth Pareto frontier.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
make fmt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏢 Organization

This project is maintained by [EvalOps](https://github.com/evalops), an organization focused on advanced LLM evaluation and safety tools.

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/evalops/folie-a-deux-dspy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/evalops/folie-a-deux-dspy/discussions)
- **Email**: [info@evalops.dev](mailto:info@evalops.dev)

## 🎯 Academic Context

This implementation contributes to the growing body of research on:

- **AI Safety and Alignment**: Investigating how consensus mechanisms may lead to truth preservation or degradation
- **Multi-Agent Systems**: Understanding emergent behaviors in co-evolutionary language model training
- **Echo Chamber Mitigation**: Developing parameterized approaches to prevent feedback loop amplification
- **Consensus Formation Theory**: Empirical study of agreement dynamics in artificial agent populations

### Experimental Limitations

- **Scale**: Limited to two-agent interactions; extension to larger multi-agent populations unexplored
- **Domain**: Focused on factual verification; generalization beyond truth-evaluable tasks requires validation
- **Model Architecture**: Results specific to tested LLM families; broader model evaluation needed
- **Baseline Coverage**: Limited comparison to debate frameworks and mixture-of-agents approaches
- **Metric Scope**: Truth evaluation via simple accuracy; more sophisticated factuality metrics (FActScore, TruthfulQA) not implemented

---

> **⚠️ Research Code**: This is experimental research software. Results should be interpreted carefully and validated in your specific context. Not intended for production deployment.