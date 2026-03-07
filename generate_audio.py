#!/usr/bin/env python3
"""
百人一首 上の句 音声生成ツール
Gemini TTS API を使って上の句の音声データを生成します。

使い方:
    python generate_audio.py [オプション]

オプション:
    --ids 1 2 3      指定した番号の歌のみ生成 (省略時は全100首)
    --range 1 10     指定した範囲の歌を生成 (1番から10番まで)
    --config FILE    設定ファイルのパス (デフォルト: tts_config.json)
    --poems FILE     歌データのパス (デフォルト: poems.json)
    --prompt FILE    プロンプト設定のパス (デフォルト: tts_prompt.json)
    --overwrite      既存の音声ファイルを上書き
    --dry-run        実際には生成せず、対象を表示するだけ
"""

import argparse
import base64
import json
import os
import sys
import time
import wave
from pathlib import Path


def load_json(path: str) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dotenv(dotenv_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env file into process env."""
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ) and len(value) >= 2:
            value = value[1:-1]

        # Keep already-exported variables as highest priority.
        os.environ.setdefault(key, value)


def save_wav(pcm_bytes: bytes, output_path: str, sample_rate: int = 24000) -> None:
    """16bit PCM バイト列を WAV ファイルとして保存する。"""
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)   # mono
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def build_text(poem: dict, prompt_config: dict) -> str:
    """プロンプト設定に基づいてTTSに渡すテキストを生成する。"""
    use_kana = prompt_config.get("use_kana", True)
    upper = poem.get("upper_kana") if use_kana else poem.get("upper")
    if not upper:
        upper = poem.get("upper", "")

    template = prompt_config.get("text_template", "{upper}")
    prefix = prompt_config.get("prefix", "")
    suffix = prompt_config.get("suffix", "")
    text = template.replace("{upper}", upper)
    return f"{prefix}{text}{suffix}".strip()


def generate_audio_for_poem(
    poem: dict,
    config: dict,
    prompt_config: dict,
    client,
    output_dir: Path,
    overwrite: bool = False,
) -> bool:
    """
    一首分の音声を生成して保存する。
    Returns True on success, False on skip, raises on error.
    """
    poem_id = poem["id"]
    ext = config.get("output_format", "wav")
    output_path = output_dir / f"{poem_id}.{ext}"

    if output_path.exists() and not overwrite:
        print(f"  [{poem_id:3d}] スキップ (既存): {output_path.name}")
        return False

    text = build_text(poem, prompt_config)
    model = config.get("model", "gemini-2.5-flash-preview-tts")
    voice_name = config.get("voice_name", "Kore")
    sample_rate = config.get("sample_rate", 24000)
    system_instruction = prompt_config.get("system_instruction", "")

    from google import genai
    from google.genai import types

    contents = text
    speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name=voice_name,
            )
        )
    )

    def build_generate_config(system_text: str | None):
        cfg = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config,
        )
        if system_text:
            cfg.system_instruction = system_text
        return cfg

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=build_generate_config(system_instruction),
        )
    except Exception as e:
        # Some TTS endpoints intermittently return 500 when system instruction is used.
        if system_instruction and "500" in str(e):
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=build_generate_config(None),
            )
        else:
            raise

    audio_part = response.candidates[0].content.parts[0]
    raw_data = audio_part.inline_data.data

    # Gemini returns base64-encoded PCM or raw bytes depending on SDK version
    if isinstance(raw_data, str):
        pcm_bytes = base64.b64decode(raw_data)
    else:
        pcm_bytes = raw_data

    save_wav(pcm_bytes, str(output_path), sample_rate)
    print(f"  [{poem_id:3d}] 生成完了: {output_path.name}  ({len(pcm_bytes)//2} samples)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="百人一首 上の句 音声生成ツール (Gemini TTS)"
    )
    parser.add_argument("--ids", nargs="+", type=int, help="生成する歌の番号 (例: 1 5 10)")
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                        help="生成する番号の範囲 (例: 1 10)")
    parser.add_argument("--config", default="tts_config.json", help="設定ファイルのパス")
    parser.add_argument("--poems", default="poems.json", help="歌データのパス")
    parser.add_argument("--prompt", default="tts_prompt.json", help="プロンプト設定のパス")
    parser.add_argument("--overwrite", action="store_true", help="既存ファイルを上書き")
    parser.add_argument("--dry-run", action="store_true", help="実際には生成しない")
    args = parser.parse_args()

    # Load configuration files
    script_dir = Path(__file__).parent
    load_dotenv(script_dir / ".env")
    config = load_json(script_dir / args.config)
    poems_data = load_json(script_dir / args.poems)
    prompt_config = load_json(script_dir / args.prompt)

    # Determine target poem IDs
    all_ids = {p["id"] for p in poems_data}
    if args.ids:
        target_ids = set(args.ids) & all_ids
        invalid = set(args.ids) - all_ids
        if invalid:
            print(f"警告: 存在しない番号が指定されました: {sorted(invalid)}", file=sys.stderr)
    elif args.range:
        start, end = args.range
        target_ids = {i for i in range(start, end + 1) if i in all_ids}
    else:
        target_ids = all_ids

    target_poems = sorted(
        [p for p in poems_data if p["id"] in target_ids],
        key=lambda p: p["id"]
    )

    print(f"対象: {len(target_poems)} 首")

    if args.dry_run:
        print("--- Dry run モード (実際には生成しません) ---")
        for poem in target_poems:
            text = build_text(poem, prompt_config)
            print(f"  [{poem['id']:3d}] {poem['poet']}: {text}")
        return

    # Setup API client
    api_key_env = config.get("api_key_env", "GEMINI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"エラー: 環境変数 {api_key_env} が設定されていません。", file=sys.stderr)
        print(f"  export {api_key_env}=your_api_key_here", file=sys.stderr)
        print(f"  または {script_dir / '.env'} に {api_key_env}=... を記載", file=sys.stderr)
        sys.exit(1)

    try:
        from google import genai
    except ImportError:
        print("エラー: google-genai パッケージがインストールされていません。", file=sys.stderr)
        print("  pip install google-genai", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Create output directory
    output_dir = script_dir / config.get("output_dir", "audio")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate audio for each poem
    retry_count = config.get("retry_count", 3)
    retry_delay = config.get("retry_delay_sec", 2.0)
    success_count = 0
    skip_count = 0
    error_count = 0

    for i, poem in enumerate(target_poems, 1):
        print(f"({i}/{len(target_poems)}) {poem['poet']} 第{poem['id']}首", end=" ")
        print(f"「{poem['upper']}」")

        for attempt in range(1, retry_count + 1):
            try:
                result = generate_audio_for_poem(
                    poem, config, prompt_config, client, output_dir, args.overwrite
                )
                if result:
                    success_count += 1
                else:
                    skip_count += 1
                break
            except Exception as e:
                if attempt < retry_count:
                    print(f"  エラー (試行 {attempt}/{retry_count}): {e} — {retry_delay}秒後にリトライ")
                    time.sleep(retry_delay * attempt)
                else:
                    print(f"  エラー: {e}", file=sys.stderr)
                    error_count += 1

    print()
    print(f"=== 完了 ===")
    print(f"  生成: {success_count} 首")
    print(f"  スキップ: {skip_count} 首")
    print(f"  エラー: {error_count} 首")
    print(f"  保存先: {output_dir}")


if __name__ == "__main__":
    main()
