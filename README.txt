EffectFactory v2 (黒背景オーバーレイ素材 生成アプリ)
====================================================

起動:
  1) Python 3.10+ 推奨
  2) 依存:
     - numpy
     - pillow
     ※ Tkinter は通常Python同梱
  3) ffmpeg を用意（PATHに通すか、UIでffmpeg.exeを指定）
  4) 実行:
     python effect_factory.py

出力:
  - *.mp4 : 黒背景素材（PV側で Screen/Add 合成）
  - *_thumb.png : 先頭フレームのサムネ
  - *.json : 生成条件と seed 記録（再現用）
  - _state.json : counter方式の連番seed管理（出力フォルダ直下）
  - _preview/ : プレビュー出力フォルダ（低解像度・短尺）

v2 の追加点:
  - ループ保証（頭尾一致）:
      チェックONで「最初と最後のフレームが同じ」になるようにサンプリングします。
      編集ソフトでそのままループしても“切れ目”が出にくくなります。
  - プレビュー生成:
      低解像度 + 短尺のMP4を先に作り、見た目確認してから本番書き出しできます。
      プレビュー後にそのまま本番生成すると “同じseed” で一致した見た目になります。

プラグイン:
  effects/*.py : 1ファイル = 1エフェクト
  EFFECT = {id,name,params,build_cache,render_frame} を定義してください。
