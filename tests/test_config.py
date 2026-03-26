"""Config loading and validation tests."""

from emmentalembed.config import EmbedConfig, EmmentalConfig, FoldConfig


def test_embed_config_defaults():
    """EmbedConfig has sensible defaults."""
    cfg = EmbedConfig()
    assert cfg.model == "facebook/esm2_t33_650M_UR50D"
    assert cfg.batch_size == 32
    assert cfg.dtype == "float16"
    assert cfg.layer == -1
    assert cfg.representation == "mean"
    assert cfg.device == "cuda"


def test_fold_config_defaults():
    """FoldConfig has sensible defaults."""
    cfg = FoldConfig()
    assert cfg.method == "chai"
    assert cfg.weights_path == "weights/af3.bin"
    assert cfg.max_seq_len == 1024
    assert cfg.num_recycles == 3
    assert cfg.checkpoint_interval == 50
    assert cfg.use_esm_embeddings is True


def test_config_roundtrip(tmp_path):
    """Config can be saved and reloaded."""
    cfg = EmmentalConfig(
        embed=EmbedConfig(
            fasta_path="test.fasta",
            model="facebook/esm2_t36_3B_UR50D",
            batch_size=16,
        ),
        fold=FoldConfig(
            fasta_path="test.fasta",
            method="af3",
            weights_path="/path/to/weights.bin",
        ),
    )
    path = tmp_path / "test_config.yaml"
    cfg.to_yaml(str(path))
    cfg2 = EmmentalConfig.from_yaml(str(path))

    assert cfg2.embed.fasta_path == "test.fasta"
    assert cfg2.embed.model == "facebook/esm2_t36_3B_UR50D"
    assert cfg2.embed.batch_size == 16
    assert cfg2.fold.method == "af3"
    assert cfg2.fold.weights_path == "/path/to/weights.bin"


def test_load_config_from_yaml(tmp_config):
    """Config loads from YAML fixture."""
    cfg = EmmentalConfig.from_yaml(str(tmp_config))
    assert cfg.embed.model == "facebook/esm2_t33_650M_UR50D"
    assert cfg.fold.method == "chai"


def test_emmental_config_defaults():
    """EmmentalConfig creates valid sub-configs by default."""
    cfg = EmmentalConfig()
    assert isinstance(cfg.embed, EmbedConfig)
    assert isinstance(cfg.fold, FoldConfig)
    assert cfg.embed.model == "facebook/esm2_t33_650M_UR50D"
    assert cfg.fold.method == "chai"
