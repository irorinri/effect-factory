"""UI translations.

Source strings are English.  ``tr("Saved look “{name}”", name=n)`` returns the
text in the active language and fills in ``str.format`` fields.

Effects and looks carry their own translations:

* plugins: ``EFFECT["i18n"] = {"ja": {"name": ..., "description": ...,
  "params": {key: (label, help)}, "choices": {value: label},
  "asset": {"label": ..., "hint": ..., "builtin": {token: label}}}}``
* looks: ``"i18n": {"ja": {"name": ..., "description": ..., "tags": [...]}}``

Common parameter labels (Palette, Glow, Brightness ...) are translated from
the shared dictionary below, so plugins only need to list their own terms.
"""

import locale
import os
import subprocess
import sys

LANGUAGES = (("ja", "日本語"), ("en", "English"))
_NAMES = dict(LANGUAGES)
_lang = "en"


def detect_language():
    """'ja' when the operating system UI is Japanese, else 'en'."""
    if os.name == "nt":
        try:
            import ctypes
            lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            return "ja" if (lang_id & 0x3FF) == 0x11 else "en"  # LANG_JAPANESE
        except Exception:
            pass
    for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        value = os.environ.get(var)
        if value and value not in ("C", "POSIX", "C.UTF-8"):
            return "ja" if value.lower().startswith("ja") else "en"
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["defaults", "read", "-g", "AppleLanguages"], capture_output=True, text=True,
                                 timeout=2).stdout
            first = next((ln.strip(' \t",()') for ln in out.splitlines() if ln.strip(' \t",()')), "")
            return "ja" if first.lower().startswith("ja") else "en"
        except Exception:
            pass
    try:
        loc = (locale.getlocale()[0] or "").lower()
        if loc.startswith("ja") or "japanese" in loc:
            return "ja"
    except Exception:
        pass
    return "en"


def resolve_language(code):
    """'ja' / 'en' as given; anything else ('auto', None) follows the OS."""
    code = str(code or "").lower()
    return code if code in _NAMES else detect_language()


def set_language(code):
    global _lang
    _lang = resolve_language(code)
    return _lang


def language():
    return _lang


def language_name(code):
    return _NAMES.get(code, code)


def tr(text, **fields):
    """Translate a UI string and fill in its ``{fields}``."""
    out = JA.get(text, text) if _lang == "ja" else text
    if fields:
        try:
            return out.format(**fields)
        except (KeyError, IndexError, ValueError):
            return text.format(**fields)
    return out


# ---------------------------------------------------------------------------
# Effects, parameters, palettes and looks

def _table(obj):
    """Translation table of a plugin or look for the active language."""
    if _lang == "en":
        return {}
    data = getattr(obj, "i18n", None) if not isinstance(obj, dict) else obj.get("i18n")
    table = (data or {}).get(_lang) if isinstance(data, dict) else None
    return table if isinstance(table, dict) else {}


def _entry(entry, index):
    if isinstance(entry, str):
        return entry if index == 0 else None
    if isinstance(entry, (list, tuple)):
        return entry[index] if len(entry) > index else None
    if isinstance(entry, dict):
        return entry.get(("label", "help")[index])
    return None


def pretty_choice(value):
    return str(value).replace("_", " ").capitalize()


def effect_name(plugin):
    return _table(plugin).get("name") or plugin.name


def effect_description(plugin):
    return _table(plugin).get("description") or tr(plugin.description or "")


def param_label(plugin, pdesc):
    english = str(pdesc.get("label") or pdesc.get("key", ""))
    entry = (_table(plugin).get("params") or {}).get(pdesc.get("key"))
    return _entry(entry, 0) or tr(english)


def param_help(plugin, pdesc):
    english = str(pdesc.get("help") or "")
    entry = (_table(plugin).get("params") or {}).get(pdesc.get("key"))
    return _entry(entry, 1) or tr(english)


def choice_label(plugin, pdesc, value):
    label = (_table(plugin).get("choices") or {}).get(str(value))
    return label or tr(pretty_choice(value))


def asset_text(plugin, key, default):
    return (_table(plugin).get("asset") or {}).get(key) or tr(default)


def asset_choice(plugin, token, default):
    return ((_table(plugin).get("asset") or {}).get("builtin") or {}).get(token) or tr(default)


def category_name(name):
    return tr(str(name))


def palette_label(name):
    name = str(name or "")
    if _lang == "ja" and name in PALETTES_JA:
        return PALETTES_JA[name]
    return name.capitalize()


def look_name(look, fallback=""):
    return _table(look or {}).get("name") or (look or {}).get("name") or fallback


def look_description(look):
    return _table(look or {}).get("description") or (look or {}).get("description", "")


def look_tags(look):
    return list(_table(look or {}).get("tags") or (look or {}).get("tags", []) or [])


PALETTES_JA = {
    "white": "ホワイト", "mono": "モノクロ", "cool": "クール", "warm": "ウォーム", "cyan": "シアン",
    "blue": "ブルー", "purple": "パープル", "magenta": "マゼンタ", "gold": "ゴールド", "amber": "アンバー",
    "green": "グリーン", "aurora": "オーロラ", "sunset": "サンセット", "neon": "ネオン", "ocean": "オーシャン",
    "sakura": "桜", "ember": "残り火", "fire": "ファイア", "pastel": "パステル", "rainbow": "レインボー",
    "ice": "アイス", "lavender": "ラベンダー", "cyber": "サイバー", "vapor": "ヴェイパー", "royal": "ロイヤル",
    "forest": "フォレスト",
}

# ---------------------------------------------------------------------------
# Japanese dictionary (English source text -> Japanese)

JA = {
    # categories, groups, formats
    "All": "すべて",
    "Particles": "パーティクル",
    "Light": "光",
    "Atmosphere": "空気感",
    "Graphic": "グラフィック",
    "Glitch": "グリッチ",
    "Other": "その他",
    "Mine": "マイルック",
    "Shape & Amount": "形と量",
    "Motion": "動き",
    "Color": "色",
    "Finish": "仕上げ",
    "Camera": "カメラ",
    "More": "その他",
    "Custom": "カスタム",
    "Asset": "素材",
    "Use any transparent PNG.": "任意の透過 PNG を使えます。",
    "Black background. Use Screen / Add blend in your editor.": "黒背景。編集ソフトでスクリーンまたは加算で合成してください。",
    "ProRes 4444 with transparency derived from brightness.": "明るさから透明度を作ったProRes 4444。",
    "Numbered transparent PNG frames in a folder.": "フォルダー内に連番の透過PNGを書き出します。",

    # shared parameter vocabulary
    "Palette": "パレット",
    "Glow": "グロー",
    "Glow Size": "グローの広がり",
    "Brightness": "明るさ",
    "Grain": "グレイン",
    "Film Grain": "フィルムグレイン",
    "Softness": "やわらかさ",
    "Blur": "ぼかし",
    "Speed": "速度",
    "Drift Speed": "漂う速さ",
    "Fall Speed": "落下速度",
    "Direction": "方向",
    "Amount": "量",
    "Density": "密度",
    "Intensity": "強さ",
    "Size": "サイズ",
    "Min Size": "最小サイズ",
    "Max Size": "最大サイズ",
    "Thickness": "太さ",
    "Spread": "ばらつき",
    "Twinkle": "きらめき",
    "Shimmer": "きらめき",
    "Sway": "揺れ",
    "Flicker": "ちらつき",
    "Rotation": "回転",
    "Center X": "中心 X",
    "Center Y": "中心 Y",
    "Stars": "星の数",
    "Colour Fringe": "色収差",
    "Contrast": "コントラスト",
    "Zoom": "ズーム",
    "Zooms the frame (below 100% tiles the effect). Mouse wheel on the preview works too.":
        "フレームを拡大・縮小します（100% 未満ではエフェクトをタイル状に並べます）。プレビュー上のマウスホイールでも操作できます。",

    # main window
    "Pick a look on the left · Space plays · R surprises you · Ctrl+E exports":
        "左のルックを選択 · Space で再生 · R でおまかせ · Ctrl+E で書き出し",
    "Undo (Ctrl+Z)": "元に戻す (Ctrl+Z)",
    "Redo (Ctrl+Y)": "やり直す (Ctrl+Y)",
    "Language / 言語": "Language / 言語",
    "Switch to the light theme": "ライトテーマに切り替え",
    "Switch to the dark theme": "ダークテーマに切り替え",
    "Keyboard shortcuts (F1)": "キーボードショートカット (F1)",
    "Surprise me – tasteful random tweak (R)": "おまかせ – センスよくランダムに調整 (R)",
    "Surprise me": "おまかせ",
    "Export the video (Ctrl+E)": "動画を書き出す (Ctrl+E)",
    "Export": "書き出し",
    "Adjust": "調整",
    "Explore": "探索",
    "Ready": "準備完了",
    "Cancel": "キャンセル",
    "Show the log": "ログを表示",
    "ffmpeg: checking…": "ffmpeg: 確認中…",
    "No effect plugins were found in the effects folder.": "effects フォルダーにエフェクトプラグインが見つかりません。",
    "Match the system ({name})": "システムに合わせる（{name}）",
    "Finish or cancel the export before changing the appearance": "表示を切り替える前に、書き出しを完了するかキャンセルしてください",
    "Light theme": "ライトテーマ",
    "Dark theme": "ダークテーマ",
    "Language: {name}": "言語: {name}",
    "Preview failed – see log": "プレビューに失敗 – ログを確認してください",
    "Saved still frame · {name}": "静止画を保存しました · {name}",
    "Show": "表示",
    "Show in folder": "フォルダーで表示",

    # library
    "Library": "ライブラリ",
    "Search looks…": "ルックを検索…",
    "No looks match your search.": "一致するルックがありません。",
    "rendering…": "レンダリング中…",
    "★ mine": "★ マイ",
    "{shown} of {total}": "{shown} / {total} 件",
    "{total} looks": "{total} 件",
    "Effect defaults": "エフェクトの初期設定",
    "Open": "開く",
    "Delete look…": "ルックを削除…",
    "Show file": "ファイルを表示",
    "Open my looks folder": "マイルックのフォルダーを開く",
    "Delete your look “{name}”?": "ルック「{name}」を削除しますか？",
    "Deleted look “{name}”": "ルック「{name}」を削除しました",
    "Look: {name}": "ルック: {name}",
    "Effect: {name}": "エフェクト: {name}",

    # preview, transport, timeline
    "Preparing preview…": "プレビューを準備中…",
    "Rendering…": "レンダリング中…",
    "{w}×{h} · {fps} fps · preview {pw}×{ph}": "{w}×{h} · {fps} fps · プレビュー {pw}×{ph}",
    "Black": "黒",
    "Scene": "風景",
    "Image…": "画像…",
    "Preview the overlay Screen-blended over a backdrop.\nExports are always black-background (or transparent).":
        "背景にスクリーン合成してプレビューします。\n書き出しは常に黒背景（または透過）です。",
    "Draft": "下書き",
    "Good": "標準",
    "Full": "フル",
    "Preview resolution. Draft is fastest; Full renders at display resolution.":
        "プレビューの解像度。下書きが最速、フルは表示サイズのままレンダリングします。",
    "Play / pause (Space)": "再生 / 一時停止 (Space)",
    "Back to start (Home)": "先頭に戻る (Home)",
    "Loop": "ループ",
    "Loop length. Drag sideways or double-click to type.": "ループの長さ。左右にドラッグするか、ダブルクリックで入力します。",
    "Seamless": "シームレス",
    "Make the clip loop seamlessly. Effects that cannot loop natively get a short cross-fade.":
        "クリップを継ぎ目なくループさせます。そのままループできないエフェクトには短いクロスフェードがかかります。",
    "✓ native loop": "✓ ネイティブループ",
    "blend {sec} s": "ブレンド {sec} 秒",
    "Timeline": "タイムライン",
    "Save the current look as marker {m} at the playhead (key {key}).":
        "現在の状態を再生位置にマーカー {m} として保存します（キー {key}）。",
    "Clear": "消去",
    "Blend back to start": "先頭へ戻す",
    "When looping, the last marker blends back into the first one.": "ループ時に、最後のマーカーから最初のマーカーへ戻るようにブレンドします。",
    "Click or drag to scrub. Drag a marker to the right to hold its look (xx / yy / zz); "
    "Shift+drag moves a marker; right-click removes it.":
        "クリックやドラッグで再生位置を移動。マーカーを右へドラッグするとその状態をホールド（xx / yy / zz）、"
        "Shift+ドラッグで移動、右クリックで削除します。",
    "loop blend": "ループブレンド",
    "Remove marker {m}": "マーカー {m} を削除",
    "Remove hold {m}": "ホールド {m} を削除",
    "Unsaved edit – press 1/2/3 to store it as X/Y/Z": "未保存の変更 – 1/2/3 キーで X/Y/Z に保存",
    "Saved {m} at {time}": "{m} を {time} に保存しました",
    "Marker {m}": "マーカー {m}",
    "Editing marker {m} – changes update it": "マーカー {m} を編集中 – 変更はマーカーに反映されます",
    "Hold {m} until {time}": "{m} を {time} までホールド",
    "Drag {m} to the right to hold it": "{m} を右へドラッグするとホールドできます",
    "Marker drag": "マーカーの移動",
    "Markers cleared": "マーカーを消去しました",
    "Clear markers": "マーカーを消去",
    "Remove {m}": "{m} を削除",
    "Blend back": "先頭へ戻す",
    "Loop length": "ループの長さ",
    "Loop is off: the clip plays once from start to end.": "ループはオフです。クリップは最初から最後まで1回だけ再生されます。",
    "✓ This effect loops natively – motion is snapped to the loop length, no blending needed.":
        "✓このエフェクトはそのままループします。動きがループの長さにそろえられるので、ブレンドは不要です。",
    "The first {sec} s blend with the continuation past the end, so the loop point is invisible.":
        "最初の{sec}秒を終端の続きとブレンドするので、ループのつなぎ目は見えません。",

    # inspector: adjust
    "Save look": "ルックを保存",
    "Save these settings as your own look (Ctrl+S)": "この設定をマイルックとして保存 (Ctrl+S)",
    "Reset": "リセット",
    "Reset every parameter back to the look": "すべてのパラメーターをルックの値に戻す",
    "Surprise": "おまかせ",
    "Surprise me: tasteful random tweak (R)": "おまかせ: センスよくランダムに調整 (R)",
    "Show advanced ({n})": "詳細設定を表示（{n}）",
    "Custom PNG…": "カスタム PNG…",
    "Look · {name}": "ルック · {name}",
    "Custom (no look)": "カスタム（ルックなし）",
    "  ·  {n} edited": "  ·  {n} 項目を変更",
    "{help}\nDouble-click to reset · click the number to type.": "{help}\nダブルクリックでリセット · 数値をクリックして入力",
    "Built-in shape · {name}": "内蔵の形 · {name}",
    "Custom PNG · {name}": "カスタム PNG · {name}",
    "Choose a transparent PNG": "透過 PNG を選択",
    "Particle shape": "パーティクルの形",
    "Edit {name}": "{name} を変更",
    "Reset {name}": "{name} をリセット",
    "Reset all": "すべてリセット",
    "Parameters reset to the look": "パラメーターをルックの値に戻しました",
    "Nothing to change – try a stronger setting": "変更できる項目がありません – 強さを上げてみてください",
    "Choose a backdrop image": "背景画像を選択",
    "Images": "画像",
    "All files": "すべてのファイル",

    # inspector: explore
    "Randomise tastefully (R)": "センスよくランダムに調整 (R)",
    "Subtle": "控えめ",
    "Balanced": "ほどよく",
    "Wild": "大胆",
    "Keep colours": "色を固定",
    "Keep shape": "形を固定",
    "Keep motion": "動きを固定",
    "Variations": "バリエーション",
    "New suggestions": "新しい案を表示",
    "Shuffle": "シャッフル",
    "Click a variation to apply it. Undo with Ctrl+Z.": "クリックで適用、Ctrl+Z で元に戻せます。",
    "Seed & variation": "シードとバリエーション",
    "Previous variation": "前のバリエーション",
    "Next variation": "次のバリエーション",
    "Seed": "シード",
    "New variation after each export": "書き出しのたびに新しいバリエーション",
    "Looks with ranges pick new values per variation; the seed also reshuffles particles.":
        "範囲指定のあるルックはバリエーションごとに値を選び直します。シードはパーティクルの配置も変えます。",
    "History": "履歴",
    "Start": "開始",
    "Edit": "変更",
    "Undo": "元に戻す",
    "Redo": "やり直す",
    "Surprise! Changed {n} settings · Ctrl+Z to undo": "おまかせ！ {n} 項目を変更しました · Ctrl+Z で元に戻せます",
    "Variation {n}": "バリエーション {n}",
    "Variation #{n}": "バリエーション #{n}",
    "Variation applied · Ctrl+Z to undo · Shuffle for more": "バリエーションを適用しました · Ctrl+Z で元に戻す · シャッフルで別の案",

    # inspector: export
    "Frame": "フレーム",
    "W": "幅",
    "H": "高さ",
    "Cross-fade": "クロスフェード",
    "File": "ファイル",
    "MOV + alpha": "MOV + アルファ",
    "PNG seq": "PNG 連番",
    "Quality": "画質",
    "Standard": "標準",
    "High": "高画質",
    "Max": "最高",
    "Encoder": "エンコーダー",
    "Auto ({encoder})": "自動（{encoder}）",
    "Output": "出力先",
    "Choose folder": "フォルダーを選択",
    "Open the output folder": "出力フォルダーを開く",
    "File prefix": "ファイル名の先頭",
    "Locate ffmpeg": "ffmpeg の場所を指定",
    "Looking for ffmpeg…": "ffmpeg を検索中…",
    "Checking ffmpeg…": "ffmpeg を確認中…",
    "none": "なし",
    "✓ ffmpeg {version} · {path}\nEncoders: {encoders}": "✓ ffmpeg {version} · {path}\nエンコーダー: {encoders}",
    "ffmpeg not found. Install it (ffmpeg.org) or locate ffmpeg.exe above.\nPNG sequences still export without ffmpeg.":
        "ffmpegが見つかりません。インストールする（ffmpeg.org）か、上でffmpeg.exeの場所を指定してください。\n"
        "PNG連番はffmpegがなくても書き出せます。",
    "ffmpeg not found": "ffmpeg なし",
    "Render": "レンダリング",
    "Export video": "動画を書き出す",
    "Render the full clip (Ctrl+E)": "クリップ全体をレンダリング (Ctrl+E)",
    "Draft MP4": "下書き MP4",
    "Quick half-resolution MP4 to check timing": "タイミング確認用に半分の解像度の MP4 をすばやく作成",
    "Still PNG": "静止画 PNG",
    "Save the frame under the playhead as a PNG": "再生位置のフレームを PNG で保存",
    "ZIP package": "ZIP パッケージ",
    "Bundle the last export with README / LICENSE for asset stores": "前回の書き出しを README / LICENSE と一緒にまとめる（素材販売向け）",
    "Copy command": "コマンドをコピー",
    "Copy the ffmpeg command line": "ffmpeg のコマンドラインをコピー",
    "{w}×{h} · {fps} fps · {sec} s · {frames} frames": "{w}×{h} · {fps} fps · {sec} 秒 · {frames} フレーム",
    "ffmpeg was not found.\n\nInstall ffmpeg (https://ffmpeg.org/download.html) or locate the executable in the "
    "Export tab. PNG sequences can be exported without it.":
        "ffmpeg が見つかりません。\n\nffmpeg（https://ffmpeg.org/download.html）をインストールするか、"
        "書き出しタブで実行ファイルの場所を指定してください。PNG 連番は ffmpeg がなくても書き出せます。",
    "Cannot create the output folder:\n{error}": "出力フォルダーを作成できません:\n{error}",
    "Exporting…": "書き出し中…",
    "Exporting · {progress}": "書き出し中 · {progress}",
    " · {time} left": " · 残り {time}",
    "Cancelling…": "キャンセル中…",
    "Exported {name} in {sec} s": "{name} を書き出しました（{sec} 秒）",
    "Last export: {path}\n(click to show in folder)": "前回の書き出し: {path}\n（クリックでフォルダーを表示）",
    "Export cancelled": "書き出しをキャンセルしました",
    "Export failed – see log": "書き出しに失敗しました – ログを確認してください",
    "Export failed:\n\n{error}": "書き出しに失敗しました:\n\n{error}",
    "Rendering still…": "静止画をレンダリング中…",
    "Export a video first, then bundle it as a ZIP package.": "先に動画を書き出してから、ZIP パッケージにまとめてください。",
    "Created {name}": "{name} を作成しました",
    "Copied the ffmpeg command to the clipboard": "ffmpeg コマンドをクリップボードにコピーしました",

    # save dialog
    "Save as your look": "マイルックとして保存",
    "Your looks appear in the library under “Mine”.": "保存したルックはライブラリの「マイルック」に表示されます。",
    "My {name}": "マイ {name}",
    "Name": "名前",
    "Description (optional)": "説明（任意）",
    "Save": "保存",
    "A built-in look already uses that name.": "その名前は内蔵のルックで使われています。",
    "Replace your look “{name}”?": "ルック「{name}」を上書きしますか？",
    "Could not save the look:\n{error}": "ルックを保存できませんでした:\n{error}",
    "Saved look “{name}”": "ルック「{name}」を保存しました",

    # dialogs
    "Log": "ログ",
    "Keyboard shortcuts": "キーボードショートカット",
    "Close": "閉じる",
    "Play / pause": "再生 / 一時停止",
    "Back to start": "先頭に戻る",
    "Step one frame": "1 フレーム移動",
    "Save marker X / Y / Z at the playhead": "再生位置にマーカー X / Y / Z を保存",
    "Undo / redo": "元に戻す / やり直す",
    "Search the library": "ライブラリを検索",
    "Reset zoom": "ズームをリセット",
    "Mouse wheel on preview": "プレビュー上でマウスホイール",
    "Double-click a slider": "スライダーをダブルクリック",
    "Reset it to the look": "ルックの値に戻す",
    "Shift + drag a slider": "Shift + スライダーをドラッグ",
    "Fine adjustment": "微調整",
    "Drag a marker right": "マーカーを右へドラッグ",
    "Hold its look (xx / yy / zz)": "その状態をホールド（xx / yy / zz）",
    "Shift + drag a marker": "Shift + マーカーをドラッグ",
    "Move it": "マーカーを移動",
}
