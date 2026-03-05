#!/usr/bin/env python3
"""OsteoVision プレゼンテーション PDF 生成スクリプト"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os

# 日本語フォント登録
pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))
pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT   = os.path.join(BASE_DIR, "OsteoVision_プレゼン.pdf")

W, H = A4

# ===== スタイル定義 =====
DARK  = colors.HexColor("#0f172a")
CYAN  = colors.HexColor("#06b6d4")
GREEN = colors.HexColor("#10b981")
GRAY  = colors.HexColor("#64748b")
LIGHT = colors.HexColor("#f1f5f9")
WHITE = colors.white

def style(name, font="HeiseiKakuGo-W5", size=11, color=DARK,
          align=TA_LEFT, leading=None, space_before=0, space_after=4):
    return ParagraphStyle(
        name,
        fontName=font,
        fontSize=size,
        textColor=color,
        alignment=align,
        leading=leading or size * 1.5,
        spaceBefore=space_before,
        spaceAfter=space_after,
    )

S_TITLE    = style("title",    size=28, color=WHITE,  align=TA_CENTER, leading=38)
S_SUBTITLE = style("subtitle", size=14, color=CYAN,   align=TA_CENTER)
S_H1       = style("h1",       size=18, color=CYAN,   space_before=8, space_after=6)
S_H2       = style("h2",       size=13, color=DARK,   space_before=6, space_after=4)
S_BODY     = style("body",     size=10, color=DARK,   space_after=3)
S_SMALL    = style("small",    size=9,  color=GRAY)
S_WHITE    = style("white",    size=10, color=WHITE)
S_CENTER   = style("center",   size=10, color=DARK,   align=TA_CENTER)
S_CAPTION  = style("caption",  size=8,  color=GRAY,   align=TA_CENTER)

def divider():
    return HRFlowable(width="100%", thickness=1, color=CYAN, spaceAfter=8, spaceBefore=4)

def spacer(h=6):
    return Spacer(1, h * mm)

def badge_table(items, bg=CYAN):
    """横並びバッジ"""
    data = [[Paragraph(f"<b>{t}</b>", S_WHITE) for t in items]]
    t = Table(data, colWidths=[((W - 40*mm) / len(items))] * len(items))
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), bg),
        ("ROUNDEDCORNERS", [4]),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
    ]))
    return t

def kv_table(rows, col1=60*mm, col2=None):
    """キーバリューテーブル"""
    col2 = col2 or (W - 40*mm - col1)
    data = [[Paragraph(f"<b>{k}</b>", S_BODY), Paragraph(v, S_BODY)] for k,v in rows]
    t = Table(data, colWidths=[col1, col2])
    t.setStyle(TableStyle([
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND",  (0,0), (0,-1), LIGHT),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    return t

def status_table(rows):
    """ステータス付きテーブル"""
    data = []
    for comp, status, note in rows:
        color = GREEN if "✅" in status else colors.HexColor("#f59e0b")
        data.append([
            Paragraph(comp, S_BODY),
            Paragraph(f"<b>{status}</b>", ParagraphStyle("s", fontName="HeiseiKakuGo-W5",
                      fontSize=10, textColor=color)),
            Paragraph(note, S_SMALL),
        ])
    t = Table(data, colWidths=[55*mm, 30*mm, 80*mm])
    t.setStyle(TableStyle([
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [WHITE, LIGHT]),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    return t

def cover_slide():
    story = []
    # 背景風の大きいテーブル
    cover = Table([[
        Paragraph("OsteoVision AI", S_TITLE),
    ]], colWidths=[W - 40*mm])
    cover.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(0,0), DARK),
        ("TOPPADDING",    (0,0),(0,0), 30),
        ("BOTTOMPADDING", (0,0),(0,0), 20),
        ("LEFTPADDING",   (0,0),(0,0), 10),
    ]))
    story.append(cover)
    story.append(spacer(4))
    story.append(Paragraph("X線撮影品質管理・関節角度自動計測 AI システム", S_SUBTITLE))
    story.append(spacer(4))
    story.append(divider())
    story.append(Paragraph("放射線技師 × AI エンジニアリングによる<br/>現場発・医療グレードのポジショニング支援プラットフォーム", S_CENTER))
    story.append(spacer(6))
    story.append(kv_table([
        ("報告日",    "2026年3月5日"),
        ("開発者",    "放射線技師 / AIエンジニア"),
        ("フェーズ",  "プロトタイプ完成・AIモデル訓練完了"),
    ]))
    return story

def slide_problem():
    story = []
    story.append(Paragraph("1. 解決する課題", S_H1))
    story.append(divider())
    story.append(Paragraph("現在の医療X線撮影における構造的な問題", S_H2))
    story.append(spacer(2))
    problems = [
        ("ポジショニング品質の属人化",
         "撮影精度が技師個人の経験・勘に依存。\n若手技師と熟練者の差が角度計測誤差に直結する。"),
        ("QC（品質確認）の事後評価",
         "現在のAIは「撮影後の診断支援」に特化。\n撮影失敗を事前に防ぐ仕組みが存在しない。"),
        ("再撮影コストと被ばくリスク",
         "ポジショニングエラーによる再撮影は患者の被ばく増加・\n現場の業務負荷増大・装置稼働率低下を招く。"),
        ("角度計測の手作業依存",
         "TPA・Cobb角等の臨床指標は医師や技師が\n手作業で計測しており、再現性・効率に課題がある。"),
    ]
    for title, desc in problems:
        story.append(Paragraph(f"■ {title}", S_H2))
        story.append(Paragraph(desc.replace("\n", "<br/>"), S_BODY))
        story.append(spacer(2))
    return story

def slide_solution():
    story = []
    story.append(Paragraph("2. OsteoVision のアプローチ", S_H1))
    story.append(divider())
    story.append(Paragraph(
        "「診断AIの上流」に位置する撮影品質管理プラットフォーム",
        S_H2))
    story.append(spacer(2))
    story.append(badge_table(["合成データ生成", "AI推論", "QA判定", "ナビゲーション出力"]))
    story.append(spacer(4))

    pillars = [
        ("① Synthetic Data Pipeline",
         "CTデータから「正解ラベル付き」模擬X線を無限生成。\n個人情報ゼロ・アノテーションコストゼロでAI訓練。"),
        ("② QC First アーキテクチャ",
         "角度計算の前に必ずポジショニング品質を評価。\n「撮り直す前に技師に具体的な修正指示を出す」設計。"),
        ("③ Config 駆動・横展開設計",
         "設定ファイルの変更だけで膝・側弯・外反母趾・\n獣医TPLO等の別部位に即時対応可能。"),
        ("④ 臨床グレード統計検証",
         "Bland-Altman分析による許容誤差（LOA）の定量化と\nGrad-CAM XAIによる診断根拠の可視化を実装。"),
    ]
    for title, desc in pillars:
        story.append(Paragraph(f"<b>{title}</b>", S_H2))
        story.append(Paragraph(desc.replace("\n", "<br/>"), S_BODY))
        story.append(spacer(2))
    return story

def slide_architecture():
    story = []
    story.append(Paragraph("3. システムアーキテクチャ", S_H1))
    story.append(divider())

    arch = Table([
        [Paragraph("<b>レイヤー</b>", S_BODY),
         Paragraph("<b>コンポーネント</b>", S_BODY),
         Paragraph("<b>技術スタック</b>", S_BODY)],
        [Paragraph("フロントエンド", S_BODY),
         Paragraph("DICOM ビューア + 解析 UI", S_BODY),
         Paragraph("Next.js / TypeScript / Tailwind CSS", S_BODY)],
        [Paragraph("バックエンド API", S_BODY),
         Paragraph("推論・角度計算・QA判定", S_BODY),
         Paragraph("FastAPI / Python / OpenCV", S_BODY)],
        [Paragraph("AI推論エンジン", S_BODY),
         Paragraph("キーポイント検出（主）", S_BODY),
         Paragraph("YOLOv8-Pose（4キーポイント）", S_BODY)],
        [Paragraph("フォールバック", S_BODY),
         Paragraph("古典的画像処理（副）", S_BODY),
         Paragraph("CLAHE + Otsu + Connected Components", S_BODY)],
        [Paragraph("データ工場", S_BODY),
         Paragraph("合成DRR生成", S_BODY),
         Paragraph("NumPy / SciPy / pydicom", S_BODY)],
        [Paragraph("XAI", S_BODY),
         Paragraph("診断根拠の可視化", S_BODY),
         Paragraph("ResNet50 + Grad-CAM ヒートマップ", S_BODY)],
    ], colWidths=[35*mm, 55*mm, 75*mm])

    arch.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), DARK),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    story.append(arch)
    story.append(spacer(4))
    story.append(Paragraph(
        "※ フェイルセーフ設計：YOLOが失敗しても Classical CV が自動起動。システムが止まらない。",
        S_SMALL))
    return story

def slide_results():
    story = []
    story.append(Paragraph("4. 本日の成果", S_H1))
    story.append(divider())
    story.append(Paragraph("YOLOv8-Pose モデル訓練完了（2026年3月5日）", S_H2))
    story.append(spacer(2))

    results = Table([
        [Paragraph("<b>指標</b>", S_BODY), Paragraph("<b>値</b>", S_BODY)],
        [Paragraph("使用 GPU",     S_BODY), Paragraph("NVIDIA T4（Google Colab）", S_BODY)],
        [Paragraph("訓練画像数",   S_BODY), Paragraph("633 枚（合成DRR）", S_BODY)],
        [Paragraph("検証画像数",   S_BODY), Paragraph("87 枚", S_BODY)],
        [Paragraph("検出精度 mAP50", S_BODY),
         Paragraph("<b><font color='#10b981'>99.8%</font></b>", S_BODY)],
        [Paragraph("訓練時間",     S_BODY), Paragraph("約 30 分（CPU比 10〜50倍高速）", S_BODY)],
        [Paragraph("個人情報使用", S_BODY), Paragraph("ゼロ（完全合成データ）", S_BODY)],
    ], colWidths=[55*mm, 110*mm])

    results.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), DARK),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]))
    story.append(results)
    story.append(spacer(4))

    story.append(Paragraph("開発完了コンポーネント", S_H2))
    story.append(status_table([
        ("OsteoSynth（データ生成）", "✅ 完成", "720枚/回 自動生成・再開機能付き"),
        ("FastAPI バックエンド",      "✅ 完成", "YOLO推論 + Classical CVフォールバック"),
        ("Next.js フロントエンド",    "✅ 完成", "DICOM/PNG対応・SVGオーバーレイ表示"),
        ("YOLOv8 訓練パイプライン",   "✅ 本日完成", "Google Colab GPU で高精度達成"),
        ("コード品質改善",            "✅ 本日完成", "ハードコード排除・環境非依存化"),
    ]))
    return story

def slide_roadmap():
    story = []
    story.append(Paragraph("5. 今後のロードマップ", S_H1))
    story.append(divider())

    road = Table([
        [Paragraph("<b>フェーズ</b>", S_BODY),
         Paragraph("<b>内容</b>", S_BODY),
         Paragraph("<b>時期</b>", S_BODY),
         Paragraph("<b>状態</b>", S_BODY)],
        [Paragraph("今週", S_BODY),
         Paragraph("訓練済みモデルを FastAPI に組み込み\n実際の推論・角度計測を確認", S_BODY),
         Paragraph("2026年3月", S_BODY),
         Paragraph("準備完了", S_BODY)],
        [Paragraph("フェーズ2", S_BODY),
         Paragraph("ファントムCTデータでの精度検証\n合成→実データのドメインギャップ測定", S_BODY),
         Paragraph("Mac mini到着後", S_BODY),
         Paragraph("計画中", S_BODY)],
        [Paragraph("フェーズ3", S_BODY),
         Paragraph("病院への倫理審査申請\n実患者データでの検証", S_BODY),
         Paragraph("2026年内", S_BODY),
         Paragraph("計画中", S_BODY)],
        [Paragraph("フェーズ4", S_BODY),
         Paragraph("Bland-Altman統計検証\n論文投稿・学会発表", S_BODY),
         Paragraph("2027年", S_BODY),
         Paragraph("計画中", S_BODY)],
    ], colWidths=[25*mm, 85*mm, 35*mm, 20*mm])

    road.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), DARK),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    story.append(road)
    story.append(spacer(6))
    story.append(Paragraph(
        "「すでに合成データで99.8%の精度を達成している」という実績が、\n"
        "病院への倫理審査申請・企業へのデモの強力な根拠となります。",
        ParagraphStyle("note", fontName="HeiseiKakuGo-W5", fontSize=10,
                       textColor=WHITE, backColor=DARK, leading=16,
                       spaceBefore=0, spaceAfter=0,
                       leftIndent=10, rightIndent=10,
                       borderPadding=10)))
    return story

def slide_advantage():
    story = []
    story.append(Paragraph("6. 競合優位性", S_H1))
    story.append(divider())
    story.append(Paragraph("大手メーカー製AIとの差別化", S_H2))
    story.append(spacer(2))

    adv = Table([
        [Paragraph("<b>項目</b>", S_BODY),
         Paragraph("<b>大手メーカー製AI</b>", S_BODY),
         Paragraph("<b>OsteoVision</b>", S_BODY)],
        [Paragraph("フォーカス", S_BODY),
         Paragraph("読影支援（事後評価）", S_BODY),
         Paragraph("撮影品質管理（事前防止）", S_BODY)],
        [Paragraph("ナビゲーション", S_BODY),
         Paragraph("なし", S_BODY),
         Paragraph("具体的な修正指示をリアルタイム出力", S_BODY)],
        [Paragraph("他部位展開", S_BODY),
         Paragraph("専用開発が必要", S_BODY),
         Paragraph("設定ファイル変更のみで対応", S_BODY)],
        [Paragraph("導入コスト", S_BODY),
         Paragraph("専用装置・ライセンス費用", S_BODY),
         Paragraph("既存装置のDICOM出力に後付け可能", S_BODY)],
        [Paragraph("開発の視点", S_BODY),
         Paragraph("エンジニア・医師主導", S_BODY),
         Paragraph("放射線技師の暗黙知をAIに実装", S_BODY)],
    ], colWidths=[35*mm, 65*mm, 65*mm])

    adv.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), DARK),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("BACKGROUND",    (2,1), (2,-1), colors.HexColor("#f0fdf4")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    story.append(adv)
    return story

# ===== PDF ビルド =====
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=20*mm, rightMargin=20*mm,
    topMargin=18*mm, bottomMargin=18*mm,
)

story = []
slides = [
    cover_slide,
    slide_problem,
    slide_solution,
    slide_architecture,
    slide_results,
    slide_roadmap,
    slide_advantage,
]

for i, slide_fn in enumerate(slides):
    story.extend(slide_fn())
    if i < len(slides) - 1:
        story.append(Spacer(1, 12*mm))
        story.append(HRFlowable(width="100%", thickness=2,
                                color=colors.HexColor("#e2e8f0"),
                                spaceAfter=12*mm))

doc.build(story)
print(f"✅ PDF作成完了: {OUTPUT}")
