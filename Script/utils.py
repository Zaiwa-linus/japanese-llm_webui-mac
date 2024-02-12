import sentencepiece as spm  # SentencePieceライブラリをインポート
import sentencepiece.sentencepiece_model_pb2 as model  # SentencePieceのモデルプロトコルをインポート

def spm_tokenizer(metadata):
    # メタデータから特別なトークンIDとトークンリストを取得
    tokens = metadata["tokenizer.ggml.tokens"]
    bos = metadata["tokenizer.ggml.bos_token_id"].item()  # 開始トークンID
    eos = metadata["tokenizer.ggml.eos_token_id"].item()  # 終了トークンID
    unk = metadata["tokenizer.ggml.unknown_token_id"].item()  # 未知トークンID

    # 正規化の設定を定義
    normalizer_spec = model.NormalizerSpec(
        name="identity",  # 正規化の名前（ここでは恒等変換）
        precompiled_charsmap=b"",  # 事前にコンパイルされた文字マップ（使用しない）
        add_dummy_prefix=True,  # ダミーの接頭辞を追加（SentencePieceの要件）
        remove_extra_whitespaces=False,  # 余分な空白を削除しない
        normalization_rule_tsv=b"",  # 正規化ルール（使用しない）
    )

    # トレーナーの設定を定義
    trainer_spec = model.TrainerSpec(
        model_type="BPE",  # モデルタイプ（ここではバイトペアエンコーディング）
        vocab_size=len(tokens),  # 語彙サイズ
        input_format="text",  # 入力形式
        # 以下、トークン化の詳細な設定
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        treat_whitespace_as_suffix=False,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        vocabulary_output_piece_score=True,
        byte_fallback=True,
        unk_id=unk,  # 未知トークンID
        bos_id=bos,  # 開始トークンID
        eos_id=eos,  # 終了トークンID
        pad_id=-1,  # パディングトークンID（使用しない）
        # 特別なトークン
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        pretokenization_delimiter="",  # 事前トークン化の区切り文字（使用しない）
    )

    # モデルプロトコルを構成し、トークンとそのスコア、タイプを追加
    m = model.ModelProto(trainer_spec=trainer_spec, normalizer_spec=normalizer_spec)
    scores = metadata.get("tokenizer.ggml.scores", None)  # トークンのスコア
    scores = scores.tolist() if scores is not None else None
    token_types = metadata.get("tokenizer.ggml.token_type", None)  # トークンタイプ
    token_types = token_types.tolist() if token_types is not None else None

    for i, token in enumerate(tokens):
        score = scores[i] if scores else 0  # スコアが指定されていない場合は0を使用
        token_type = token_types[i] if token_types else 0  # トークンタイプが指定されていない場合は0を使用
        # トークンをモデルに追加
        m.pieces.append(
            model.ModelProto.SentencePiece(piece=token, score=score, type=token_type)
        )

    # モデルプロトコルからSentencePieceプロセッサを初期化
    tokenizer = spm.SentencePieceProcessor(model_proto=m.SerializeToString())
    return tokenizer  # トークナイザを返す
