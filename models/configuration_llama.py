"""LLaMA model configuration with extended RoPE scaling validation.

This module defines the LlamaConfig class that extends PretrainedConfig to support
comprehensive configuration of LLaMA models with various Rotary Position Embedding
(RoPE) scaling strategies. It provides robust validation for all 20+ supported
RoPE scaling types and their parameters.

Key Features:
    - Complete LLaMA architecture configuration (vocab size, hidden size, layers, etc.)
    - Support for Grouped Query Attention (GQA) via num_key_value_heads parameter
    - Comprehensive RoPE scaling validation with deprecation handling
    - Static vs dynamic scaling mode mutual exclusivity enforcement
    - Automatic remapping of deprecated dynamic-* type strings

RoPE Scaling Types Supported:
    Standard: linear, ntk, part-ntk, yarn
    Custom: my-rope, my-rope-scaled, my-rope2, my-rope2-scaled
    Block-based: block-layered, block-layered-scaled
    Frequency-based: freq-smooth, freq-smooth-scaled,
                    freq-reciprocal, freq-reciprocal-scaled,
                    freq-reciprocal-scaled-no-layer, freq-reciprocal-scaled-adaptive
    Dual-position: dual-rope, dual-rope-scaled,
                   inverse-dual-rope, inverse-dual-rope-scaled

Configuration Validation:
    The _rope_scaling_validation() method ensures:
    - rope_scaling is a dictionary with valid 'type' field
    - Type is one of the supported values or a deprecated variant
    - At least one of 'factor' (>1.0) or 'dynamic' (True) is present
    - Factor and dynamic are mutually exclusive (factor wins if both given)
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class LlamaConfig(PretrainedConfig):
    r"""
    Configuration class for storing the configuration of a LlamaModel.

    This class is used to instantiate a LLaMA model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from PretrainedConfig and can be used to control the model
    outputs. Read the documentation from PretrainedConfig for more information.

    Attributes:
        vocab_size (int, optional): Vocabulary size of the LLaMA model. Defaults to 32000.
        hidden_size (int, optional): Dimension of the hidden representations. Defaults to 4096.
        intermediate_size (int, optional): Dimension of the MLP representations. Defaults to 11008.
        num_hidden_layers (int, optional): Number of hidden layers in the Transformer encoder.
            Defaults to 32.
        num_attention_heads (int, optional): Number of attention heads for each attention layer
            in the Transformer encoder. Defaults to 32.
        num_key_value_heads (int, optional): Number of key_value heads for Grouped Query Attention.
            Defaults to num_attention_heads.
        pretraining_tp (int, optional): Tensor parallelism rank used during pretraining.
            Defaults to 1.
        hidden_act (str or function, optional): The non-linear activation function in the decoder.
            Defaults to "silu".
        max_position_embeddings (int, optional): The maximum sequence length that this model might
            ever be used with. Defaults to 2048.
        initializer_range (float, optional): The standard deviation of the truncated_normal_initializer
            for initializing all weight matrices. Defaults to 0.02.
        rms_norm_eps (float, optional): The epsilon used by the rms normalization layers.
            Defaults to 1e-6.
        use_cache (bool, optional): Whether or not the model should return the last key/values
            attentions. Defaults to True.
        tie_word_embeddings (bool, optional): Whether to tie weight embeddings. Defaults to False.
        rope_scaling (dict, optional): Dictionary containing the scaling configuration for the RoPE
            embeddings. Supported types: "linear", "ntk", "part-ntk", "yarn", "my-rope",
            "my-rope-scaled", "my-rope2", "my-rope2-scaled", "block-layered", "block-layered-scaled",
            "freq-smooth", "freq-smooth-scaled", "freq-reciprocal", "freq-reciprocal-scaled",
            "freq-reciprocal-scaled-no-layer", "freq-reciprocal-scaled-adaptive",
            "dual-rope", "dual-rope-scaled", "inverse-dual-rope", "inverse-dual-rope-scaled".

            All types support the same mutually exclusive "factor" / "dynamic" interface:

            * "factor" (float > 1.0) - static mode: the scaling factor is fixed at initialisation
              and frequencies are pre-cached up to max_position_embeddings. Use when the target
              context length is known ahead of time.
            * "dynamic": true - dynamic mode: the effective scaling factor is computed on every
              forward pass as max(1, seq_len / original_L), so the model adapts automatically to
              any sequence length without reloading weights.

            If both "factor" and "dynamic" are supplied simultaneously, "factor" takes priority
            and "dynamic" is ignored with a warning.

            Example (static YaRN, 4x extension)::

                rope_scaling = {"type": "yarn", "factor": 4.0}

            Example (dynamic My RoPE, auto-scaling)::

                rope_scaling = {"type": "my-rope", "dynamic": True}

        attention_bias (bool, optional): Whether to use a bias in the query, key, value and output
            projection layers. Defaults to False.
        attention_dropout (float, optional): The dropout ratio for the attention probabilities.
            Defaults to 0.0.

    Example::

        >>> from transformers import LlamaModel, LlamaConfig
        >>> configuration = LlamaConfig()
        >>> model = LlamaModel(configuration)
        >>> configuration = model.config
    """

    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        original_max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """Validate the rope_scaling configuration.

        Validates that the rope_scaling dictionary contains valid type and scaling mode
        specifications. This method ensures that the RoPE scaling configuration follows
        the expected format and constraints.

        Valid type values:
            - "linear": Position Interpolation (PI)
            - "ntk": NTK-aware scaling
            - "part-ntk": NTK-by-parts scaling
            - "yarn": YaRN
            - "my-rope": layer-aware custom RoPE
            - "my-rope-scaled": scaled layer-aware custom RoPE
            - "my-rope2": multi-scale custom RoPE
            - "my-rope2-scaled": scaled multi-scale custom RoPE
            - "block-layered": Block-Layered RoPE
            - "block-layered-scaled": scaled Block-Layered RoPE
            - "freq-smooth": frequency smoothing RoPE
            - "freq-smooth-scaled": scaled frequency smoothing RoPE
            - "freq-reciprocal": reciprocal frequency RoPE
            - "freq-reciprocal-scaled": scaled reciprocal frequency RoPE
            - "freq-reciprocal-scaled-no-layer": scaled reciprocal frequency RoPE, no layer index
            - "freq-reciprocal-scaled-adaptive": scaled reciprocal frequency RoPE, adaptive scaling
            - "dual-rope": dual RoPE (position only)
            - "dual-rope-scaled": dual RoPE + attention temperature
            - "inverse-dual-rope": inverse dual RoPE (position only)
            - "inverse-dual-rope-scaled": inverse dual RoPE + attention temperature

        All types share the same "factor" / "dynamic" interface:
            - "factor" (float > 1.0): static mode with a fixed scaling ratio.
            - "dynamic": true: dynamic mode; ratio derived at runtime.

        If both are supplied, "factor" wins and "dynamic" is dropped with a warning.
        At least one must be present.

        Deprecated type strings:
            "dynamic-linear", "dynamic-ntk", "dynamic-part-ntk", "dynamic-yarn",
            "dynamic-my-rope", "dynamic-my-rope-scaled", "dynamic-my-rope2",
            "dynamic-my-rope2-scaled", "dynamic-block-layered",
            "dynamic-block-layered-scaled", "dynamic-freq-smooth",
            "dynamic-freq-smooth-scaled", "dynamic-freq-reciprocal",
            "dynamic-freq-reciprocal-scaled", "dynamic-freq-reciprocal-scaled-no-layer",
            "dynamic-freq-reciprocal-scaled-adaptive"
            are remapped to their base type with "dynamic": True.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict):
            raise ValueError(
                "`rope_scaling` must be a dictionary, " f"got {self.rope_scaling}"
            )

        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        rope_scaling_dynamic = self.rope_scaling.get("dynamic", None)

        _deprecated_dynamic_map = {
            "dynamic-linear": "linear",
            "dynamic-ntk": "ntk",
            "dynamic-part-ntk": "part-ntk",
            "dynamic-yarn": "yarn",
            "dynamic-my-rope": "my-rope",
            "dynamic-my-rope-scaled": "my-rope-scaled",
            "dynamic-my-rope2": "my-rope2",
            "dynamic-my-rope2-scaled": "my-rope2-scaled",
            "dynamic-block-layered": "block-layered",
            "dynamic-block-layered-scaled": "block-layered-scaled",
            "dynamic-freq-smooth": "freq-smooth",
            "dynamic-freq-smooth-scaled": "freq-smooth-scaled",
            "dynamic-freq-reciprocal": "freq-reciprocal",
            "dynamic-freq-reciprocal-scaled": "freq-reciprocal-scaled",
            "dynamic-freq-reciprocal-scaled-no-layer": "freq-reciprocal-scaled-no-layer",
            "dynamic-freq-reciprocal-scaled-adaptive": "dynamic-freq-reciprocal-scaled-adaptive",
            "dynamic-dual-rope": "dual-rope",
            "dynamic-dual-rope-scaled": "dual-rope-scaled",
            "dynamic-inverse-dual-rope": "inverse-dual-rope",
            "dynamic-inverse-dual-rope-scaled": "inverse-dual-rope-scaled",
        }
        if rope_scaling_type in _deprecated_dynamic_map:
            new_type = _deprecated_dynamic_map[rope_scaling_type]
            logger.warning(
                f"`rope_scaling` type '{rope_scaling_type}' is deprecated. "
                f"Use '{new_type}' with 'dynamic': True instead."
            )
            self.rope_scaling["type"] = new_type
            # Only inject dynamic=True when factor is absent; otherwise factor wins.
            if rope_scaling_factor is None:
                self.rope_scaling.setdefault("dynamic", True)
                rope_scaling_dynamic = self.rope_scaling.get("dynamic", True)
            rope_scaling_type = new_type

        valid_types = [
            "linear",
            "ntk",
            "part-ntk",
            "yarn",
            "my-rope",
            "my-rope-scaled",
            "my-rope2",
            "my-rope2-scaled",
            "block-layered",
            "block-layered-scaled",
            "freq-smooth",
            "freq-smooth-scaled",
            "freq-reciprocal",
            "freq-reciprocal-scaled",
            "freq-reciprocal-scaled-no-layer",
            "freq-reciprocal-scaled-adaptive",
            "dual-rope",
            "dual-rope-scaled",
            "inverse-dual-rope",
            "inverse-dual-rope-scaled",
        ]
        if rope_scaling_type is None or rope_scaling_type not in valid_types:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of {valid_types}, "
                f"got {rope_scaling_type}"
            )

        # ------------------------------------------------------------------ #
        # Validate optional `dynamic` field                                  #
        # ------------------------------------------------------------------ #
        if rope_scaling_dynamic is not None and not isinstance(
            rope_scaling_dynamic,
            bool,
        ):
            raise ValueError(
                "`rope_scaling.dynamic` must be a boolean if specified, "
                f"got {type(rope_scaling_dynamic)}"
            )

        # ------------------------------------------------------------------ #
        # Mutual exclusivity: factor vs dynamic (applies to all types)       #
        # Both present -> factor wins; dynamic is stripped with a warning.   #
        # ------------------------------------------------------------------ #
        factor_present = rope_scaling_factor is not None
        dynamic_present = rope_scaling_dynamic is True

        if factor_present and dynamic_present:
            logger.warning(
                "`rope_scaling` contains both 'factor' and 'dynamic': these fields "
                "are mutually exclusive. 'factor' takes priority; 'dynamic' will be "
                f"ignored. (factor={rope_scaling_factor})"
            )
            del self.rope_scaling["dynamic"]
            rope_scaling_dynamic = None
            dynamic_present = False

        if not factor_present and not dynamic_present:
            raise ValueError(
                f"`rope_scaling` with type '{rope_scaling_type}' requires exactly "
                "one of 'factor' (float > 1.0) or 'dynamic' (true). Got neither."
            )

        # ------------------------------------------------------------------ #
        # Validate factor value when present                                 #
        # ------------------------------------------------------------------ #
        if factor_present:
            if (
                not isinstance(rope_scaling_factor, (int, float))
                or rope_scaling_factor < 1.0
            ):
                raise ValueError(
                    "`rope_scaling`'s 'factor' must be a float strictly greater than "
                    f"1.0, got {rope_scaling_factor}"
                )
