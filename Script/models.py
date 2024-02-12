# このコードはhttps://github.com/ml-explore/mlx-examples.gitをベースにしています。

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from Script import utils
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten

@dataclass
class ModelArgs:
    # モデルのハイパーパラメータを定義するデータクラス
    hidden_size: int  # 隠れ層のサイズ
    num_hidden_layers: int  # 隠れ層の数
    intermediate_size: int  # フィードフォワードネットワークのサイズ
    num_attention_heads: int  # 注意ヘッドの数
    rms_norm_eps: float  # RMSNorm正規化のイプシロン値
    vocab_size: int  # 語彙サイズ
    num_key_value_heads: int = None  # キーとバリューのためのヘッド数（デフォルトで注意ヘッドの数と同じ）
    rope_theta: float = 10000  # RoPEエンコーディングの角度パラメータ
    rope_traditional: bool = False  # 従来のRoPEエンコーディングを使用するかどうか
    model_type: str = None  # モデルのタイプ（現在は使用されていない）
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None  # RoPEのスケーリングパラメータ

    def __post_init__(self):
        # 初期化後に追加の設定を行うメソッド
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads  # キーとバリューヘッド数が未設定の場合、注意ヘッド数を使用

        if self.rope_scaling:
            # RoPEスケーリングパラメータの検証
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")

    @classmethod
    def from_dict(cls, params):
        # ディクショナリからModelArgsオブジェクトを生成するクラスメソッド
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )



class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.repeats = n_heads // n_kv_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        if self.repeats > 1:
            keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.norm(h), cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = LlamaModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache


def get_config(metadata: dict):
    output = {
        "hidden_size": metadata["llama.embedding_length"],
        "num_hidden_layers": metadata["llama.block_count"],
        "num_attention_heads": metadata["llama.attention.head_count"],
        "intermediate_size": metadata["llama.feed_forward_length"],
        "num_key_value_heads": metadata["llama.attention.head_count_kv"],
        "rms_norm_eps": metadata["llama.attention.layer_norm_rms_epsilon"],
        "vocab_size": len(metadata["tokenizer.ggml.tokens"]),
        "rope_theta": 10000,
        "rope_traditional": True,
    }
    output = {k: v.item() if isinstance(v, mx.array) else v for k, v in output.items()}
    return output


class GGUFTokenizer:
    def __init__(self, metadata):
        self._tokenizer = utils.spm_tokenizer(metadata)

    def encode(self, s: str) -> mx.array:
        return mx.array([self._tokenizer.bos_id()] + self._tokenizer.encode(s))

    @property
    def eos_token_id(self):
        return self._tokenizer.eos_id()

    def decode(self, toks: List[int]) -> str:
        return self._tokenizer.decode(toks)


def translate_weight_names(name):
    name = name.replace("blk.", "model.layers.")
    name = name.replace("ffn_gate", "mlp.gate_proj")
    name = name.replace("ffn_down", "mlp.down_proj")
    name = name.replace("ffn_up", "mlp.up_proj")
    name = name.replace("attn_q", "self_attn.q_proj")
    name = name.replace("attn_k", "self_attn.k_proj")
    name = name.replace("attn_v", "self_attn.v_proj")
    name = name.replace("attn_output", "self_attn.o_proj")
    name = name.replace("attn_norm", "input_layernorm")
    name = name.replace("ffn_norm", "post_attention_layernorm")
    name = name.replace("token_embd", "model.embed_tokens")
    name = name.replace("output_norm", "model.norm")
    name = name.replace("output", "lm_head")
    return name


def load(gguf_file: str, repo: str = None):
    # gguf_fileが存在する場合は、そのファイルからモデルを読み込む。
    # 存在しない場合は、Hugging Faceのリポジトリからダウンロードしてキャッシュする。
    if not Path(gguf_file).exists():
        if repo is None:
            # ファイルが見つからず、リポジトリも提供されていない場合はエラーを発生させる。
            raise ValueError(
                f"Could not find file {gguf_file}, and no Hugging Face"
                " repo provided for download."
            )
        # Hugging Faceのリポジトリからファイルをダウンロードする。
        model_path = snapshot_download(
            repo_id=repo,
            allow_patterns=[gguf_file],
        )
        if not (Path(model_path) / gguf_file).exists():
            # リポジトリにファイルが存在しない場合はエラーを発生させる。
            raise ValueError(f"File {gguf_file} not in repo {repo}.")
        gguf_file = str(Path(model_path) / gguf_file)

    # モデルを読み込むことに成功した場合、情報メッセージを表示する。
    print(f"[INFO] Loading model from {gguf_file}")
    # mx.loadを使用して、重みとメタデータを読み込む。
    weights, metadata = mx.load(gguf_file, return_metadata=True)
    # ファイルタイプに基づいて量子化の設定を行う（または行わない）。
    gguf_ft = metadata["general.file_type"]
    if gguf_ft == 0 or gguf_ft == 1:
        # 全ての重みが32ビット浮動小数点数（F32）か、ほとんどが16ビット浮動小数点数（F16）。
        quantization = None
    elif gguf_ft == 2 or gguf_ft == 3:
        # ほとんどが4ビット量子化。
        quantization = {"group_size": 32, "bits": 4}
    elif gguf_ft == 7:
        # ほとんどが8ビット量子化。
        quantization = {"group_size": 32, "bits": 8}
    else:
        # サポートされていない量子化を使用している場合は警告を表示し、float16にキャストする。
        quantization = None
        print("[WARNING] Using unsupported GGUF quantization. Casting to float16.")

    # 重みの名前をモデルの内部名に変換する。
    weights = {translate_weight_names(k): v for k, v in weights.items()}
    # メタデータからモデルの設定を取得し、ModelArgsオブジェクトを作成する。
    config = get_config(metadata)
    # Modelオブジェクトを初期化する。
    model = Model(ModelArgs(**config))
    if quantization is not None:
        # 量子化が必要な場合、LMヘッド（言語モデルの出力層）を量子化する。
        qm = model if "lm_head.scales" in weights else model.model
        nn.QuantizedLinear.quantize_module(
            qm,
            **quantization,
        )

    # 重みのデキュアンタイズ（量子化解除）を行う関数を定義する。
    def dequantize(k):
        weight = weights.pop(f"{k}.weight")
        scales = weights.pop(f"{k}.scales")
        biases = weights.pop(f"{k}.biases")
        weights[f"{k}.weight"] = mx.dequantize(
            weight, scales=scales, biases=biases, **quantization
        )

    # 埋め込み層の重みをデキュアンタイズする。
    dequantize("model.embed_tokens")

    # トークナイザを初期化する。
    tokenizer = GGUFTokenizer(metadata)
    # モデルに重みをロードする。
    model.load_weights(list(weights.items()))
    # モデルとトークナイザを返す。
    return model, tokenizer


def generate(prompt: mx.array, model: Model, temp: float = 0.0):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y
