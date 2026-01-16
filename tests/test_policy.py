"""Tests for policy neural networks."""

import numpy as np
import pytest
import torch

from src.agent.policy import (
    ActorCritic,
    ActorHead,
    BalatroNetwork,
    CardSelectionNetwork,
    CriticHead,
    create_policy,
)


class TestBalatroNetwork:
    """Tests for BalatroNetwork feature extractor."""

    def test_initialization(self):
        """Test network initializes with correct dimensions."""
        net = BalatroNetwork(obs_dim=69, hidden_dim=256)
        assert net.fc1.in_features == 69
        assert net.fc1.out_features == 256
        assert net.fc2.in_features == 256
        assert net.fc2.out_features == 256

    def test_custom_dimensions(self):
        """Test custom dimension initialization."""
        net = BalatroNetwork(obs_dim=100, hidden_dim=128)
        assert net.fc1.in_features == 100
        assert net.fc2.out_features == 128

    def test_forward_shape(self):
        """Test forward pass output shape."""
        net = BalatroNetwork(obs_dim=69, hidden_dim=256)
        x = torch.randn(32, 69)  # Batch of 32
        out = net(x)
        assert out.shape == (32, 256)

    def test_forward_single(self):
        """Test forward pass with single sample."""
        net = BalatroNetwork()
        x = torch.randn(1, 69)
        out = net(x)
        assert out.shape == (1, 256)

    def test_layer_norm_present(self):
        """Test layer normalization is present."""
        net = BalatroNetwork()
        assert hasattr(net, "ln1")
        assert hasattr(net, "ln2")


class TestActorHead:
    """Tests for ActorHead policy output."""

    def test_initialization(self):
        """Test actor head initialization."""
        actor = ActorHead(feature_dim=256, action_dim=15)
        assert actor.fc.in_features == 256
        assert actor.fc.out_features == 15

    def test_forward_shape(self):
        """Test forward pass shape."""
        actor = ActorHead(feature_dim=256, action_dim=15)
        features = torch.randn(32, 256)
        logits = actor(features)
        assert logits.shape == (32, 15)

    def test_action_masking(self):
        """Test action masking sets invalid actions to large negative."""
        actor = ActorHead(feature_dim=256, action_dim=15)
        features = torch.randn(1, 256)

        # Create mask where only actions 0, 1, 2 are valid
        mask = torch.zeros(1, 15, dtype=torch.bool)
        mask[0, :3] = True

        logits = actor(features, action_mask=mask)

        # Invalid actions should have very low logits
        assert logits[0, 3:].max() < -1e7
        # Valid actions should be normal
        assert logits[0, :3].min() > -1e7

    def test_no_mask_passes_through(self):
        """Test that no mask doesn't modify logits."""
        actor = ActorHead(feature_dim=256, action_dim=15)
        features = torch.randn(1, 256)

        logits = actor(features, action_mask=None)

        # All logits should be reasonable values
        assert logits.abs().max() < 100


class TestCriticHead:
    """Tests for CriticHead value output."""

    def test_initialization(self):
        """Test critic head initialization."""
        critic = CriticHead(feature_dim=256)
        assert critic.fc.in_features == 256
        assert critic.fc.out_features == 1

    def test_forward_shape(self):
        """Test forward pass shape."""
        critic = CriticHead(feature_dim=256)
        features = torch.randn(32, 256)
        value = critic(features)
        assert value.shape == (32, 1)


class TestActorCritic:
    """Tests for combined ActorCritic network."""

    def test_initialization(self):
        """Test actor-critic initialization."""
        model = ActorCritic(obs_dim=69, action_dim=15, hidden_dim=256)
        assert model.obs_dim == 69
        assert model.action_dim == 15

    def test_forward(self):
        """Test forward pass returns logits and value."""
        model = ActorCritic()
        obs = torch.randn(32, 69)

        logits, value = model(obs)

        assert logits.shape == (32, 15)
        assert value.shape == (32, 1)

    def test_forward_with_mask(self):
        """Test forward pass with action mask."""
        model = ActorCritic()
        obs = torch.randn(32, 69)
        mask = torch.ones(32, 15, dtype=torch.bool)
        mask[:, 10:] = False  # Only first 10 actions valid

        logits, value = model(obs, action_mask=mask)

        assert logits.shape == (32, 15)
        # Masked actions should have low logits
        assert logits[:, 10:].max() < -1e7

    def test_get_action_stochastic(self):
        """Test stochastic action sampling."""
        model = ActorCritic()
        obs = torch.randn(1, 69)

        action, log_prob, value = model.get_action(obs, deterministic=False)

        assert action.shape == (1,)
        assert 0 <= action.item() < 15
        assert log_prob.shape == (1,)
        assert value.shape == (1,)

    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        model = ActorCritic()
        obs = torch.randn(1, 69)

        # Multiple calls with same input should give same action
        action1, _, _ = model.get_action(obs, deterministic=True)
        action2, _, _ = model.get_action(obs, deterministic=True)

        assert action1.item() == action2.item()

    def test_get_action_with_mask(self):
        """Test action sampling respects mask."""
        model = ActorCritic()
        obs = torch.randn(1, 69)

        # Only allow action 5
        mask = torch.zeros(1, 15, dtype=torch.bool)
        mask[0, 5] = True

        action, _, _ = model.get_action(obs, mask, deterministic=True)

        assert action.item() == 5

    def test_evaluate_actions(self):
        """Test action evaluation for PPO update."""
        model = ActorCritic()
        obs = torch.randn(32, 69)
        actions = torch.randint(0, 15, (32,))

        log_probs, values, entropy = model.evaluate_actions(obs, actions)

        assert log_probs.shape == (32,)
        assert values.shape == (32,)
        assert entropy.shape == (32,)
        assert (entropy >= 0).all()  # Entropy should be non-negative

    def test_get_value(self):
        """Test value-only computation."""
        model = ActorCritic()
        obs = torch.randn(32, 69)

        values = model.get_value(obs)

        assert values.shape == (32,)


class TestCardSelectionNetwork:
    """Tests for attention-based card selection."""

    def test_initialization(self):
        """Test card selection network initialization."""
        net = CardSelectionNetwork(
            card_features=6,
            max_cards=8,
            hidden_dim=128,
            num_heads=4,
        )
        assert net.max_cards == 8
        assert net.card_features == 6

    def test_forward_shape(self):
        """Test forward pass shape."""
        net = CardSelectionNetwork()
        card_obs = torch.randn(32, 8, 6)  # Batch, max_cards, features

        scores = net(card_obs)

        assert scores.shape == (32, 8)

    def test_card_masking(self):
        """Test card masking for invalid cards."""
        net = CardSelectionNetwork()
        card_obs = torch.randn(1, 8, 6)

        # Only first 5 cards valid
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[0, :5] = True

        scores = net(card_obs, card_mask=mask)

        # Invalid cards should have very low scores
        assert scores[0, 5:].max() < -1e7


class TestCreatePolicy:
    """Tests for policy factory function."""

    def test_create_default(self):
        """Test creating policy with defaults."""
        policy = create_policy(device="cpu")
        assert isinstance(policy, ActorCritic)
        assert policy.obs_dim == 69
        assert policy.action_dim == 15

    def test_create_custom(self):
        """Test creating policy with custom dimensions."""
        policy = create_policy(
            obs_dim=100,
            action_dim=20,
            hidden_dim=512,
            device="cpu",
        )
        assert policy.obs_dim == 100
        assert policy.action_dim == 20

    def test_device_placement(self):
        """Test model is on correct device."""
        policy = create_policy(device="cpu")
        # Check a parameter is on CPU
        assert next(policy.parameters()).device == torch.device("cpu")


class TestPolicyGradients:
    """Tests for gradient computation."""

    def test_gradients_flow(self):
        """Test that gradients flow through the network."""
        model = ActorCritic()
        obs = torch.randn(32, 69)
        actions = torch.randint(0, 15, (32,))

        log_probs, values, entropy = model.evaluate_actions(obs, actions)

        # Compute a simple loss
        loss = -log_probs.mean() + values.mean() - 0.01 * entropy.mean()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
