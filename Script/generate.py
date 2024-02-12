# このコードは、指定されたプロンプトからテキストを生成するスクリプトです。
# 指定されたモデルとトークナイザを使用して、最大トークン数までまたはEOS（End Of Sentence）トークンが現れるまでテキストを生成します。
# 生成されたテキストはリアルタイムで標準出力に出力され、プロンプトの処理速度と生成速度（トークン毎秒）も計算されます。
# このコードはhttps://github.com/ml-explore/mlx-examples.gitをベースにしています。

import argparse
import time

import mlx.core as mx  # mlx.coreモジュールをmxとしてインポート
import models  # modelsモジュールをインポート

def generate(
    model: models.Model,  # モデルクラスのインスタンス
    tokenizer: models.GGUFTokenizer,  # トークナイザクラスのインスタンス
    prompt: str,  # プロンプト文字列
    max_tokens: int,  # 生成する最大トークン数
    temp: float = 0.0,  # サンプリング温度
):
    prompt = tokenizer.encode(prompt)  # プロンプトをトークン化

    tic = time.time()  # 現在の時刻を記録
    tokens = []  # 生成されたトークンを格納するリスト
    skip = 0  # 既に出力された文字数
    for token, n in zip(
        models.generate(prompt, model, args.temp),  # モデルを使用してトークンを生成
        range(args.max_tokens),  # 最大トークン数まで繰り返し
    ):
        if token == tokenizer.eos_token_id:  # EOSトークンが出たらループを抜ける
            break

        if n == 0:
            prompt_time = time.time() - tic  # プロンプトの処理時間を計算
            tic = time.time()  # 次の計測のため現在時刻を更新

        tokens.append(token.item())  # 生成されたトークンをリストに追加
        s = tokenizer.decode(tokens)  # トークンリストを文字列にデコード

        if '\ufffd' not in s[skip:]:
            print(s[skip:], end="", flush=True)# 未出力の部分のみを出力
            skip = len(s) # 出力済みの文字数を更新

    print(tokenizer.decode(tokens)[skip:], flush=True)  # 最後の出力
    gen_time = time.time() - tic  # 生成にかかった時間を計算
    print("=" * 10)
    if len(tokens) == 0:
        print("No tokens generated for this prompt")  # トークンが1つも生成されなかった場合のメッセージ
        return
    prompt_tps = prompt.size / prompt_time  # プロンプトのトークン毎秒（TPS）
    gen_tps = (len(tokens) - 1) / gen_time  # 生成のTPS
    print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")  # プロンプトのTPSを出力
    print(f"Generation: {gen_tps:.3f} tokens-per-sec")  # 生成のTPSを出力


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")  # コマンドライン引数のパーサーを作成
    parser.add_argument(
        "--gguf",
        type=str,
        help="The GGUF file to load (and optionally download).",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="The Hugging Face repo if downloading from the Hub.",
    )

    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
        default="In the beginning the Universe was created.",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")  # 乱数生成器のシード値

    args = parser.parse_args()  # 引数を解析
    mx.random.seed(args.seed)  # 乱数生成器のシードを設定
    model, tokenizer = models.load(args.gguf, args.repo)  # モデルとトークナイザをロード
    generate(model, tokenizer, args.prompt, args.max_tokens, args.temp)  # テキスト生成関数を呼び出し
