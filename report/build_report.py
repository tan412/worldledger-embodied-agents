"""Build both manuscripts and their six paired vector figures from public inputs.

Run: python report/build_report.py
Requires reportlab and a CJK TrueType font (WORLDLEDGER_CJK_FONT).
Figures are conceptual unless their caption identifies public JSON measurements.
"""
from pathlib import Path
import hashlib
import html
import json
import os
import re

from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Circle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Preformatted

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"
WIDTH = A4[0] - 92
INK = colors.HexColor("#183442")
MUTED = colors.HexColor("#526871")
BLUE = colors.HexColor("#24668C")
GREEN = colors.HexColor("#237B65")
RED = colors.HexColor("#B65248")
AMBER = colors.HexColor("#98712C")
PALE = colors.HexColor("#EFF4F6")
LINE = colors.HexColor("#BFCDD3")

CJK_FONT = Path(os.environ.get("WORLDLEDGER_CJK_FONT", "/Library/Fonts/Arial Unicode.ttf"))
if not CJK_FONT.is_file():
    raise SystemExit("Set WORLDLEDGER_CJK_FONT to a CJK TrueType font before building.")
pdfmetrics.registerFont(TTFont("WorldLedgerCJK", str(CJK_FONT)))


def load(name):
    return json.loads((ROOT / "examples" / name).read_text())


def label(d, x, y, text, zh, size=9, color=INK, anchor="middle"):
    d.add(String(x, y, text, fontName="WorldLedgerCJK" if zh else "Helvetica",
                 fontSize=size, fillColor=color, textAnchor=anchor))


def box(d, x, y, w, h, lines, zh, dashed=False, color=INK):
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=PALE,
               strokeColor=color if dashed else LINE,
               strokeDashArray=[4, 3] if dashed else None, strokeWidth=.8))
    start = y + h / 2 + (len(lines) - 1) * 7
    for i, text in enumerate(lines):
        label(d, x + w / 2, start - i * 14 - 3, text, zh, color=color)


def arrow(d, x1, y1, x2, y2, color=BLUE):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1))
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** .5
    ux, uy = dx / length, dy / length
    d.add(Polygon([x2, y2, x2 - 5 * ux + 2.4 * uy, y2 - 5 * uy - 2.4 * ux,
                   x2 - 5 * ux - 2.4 * uy, y2 - 5 * uy + 2.4 * ux],
                  fillColor=color, strokeColor=color))


def system(zh):
    d = Drawing(WIDTH, 278)
    words = (
        [["已有数据", "记录与标注"], ["候选动作", "任务与初态"],
         ["适配与能力盘点", "来源 / 单位 / 时钟"], ["任务仿真", "控制 / 状态 / 接触"],
         ["条件化检查", "六态逐项裁决"], ["验证工作进程", "新仿真 + 控制重放"],
         ["回执 / 轨迹 / 哈希 / 汇总"], ["外部事务控制器", "验收后提交"],
         ["外部资产与数据", "模型 / 原始归档"]]
        if zh else
        [["Existing data", "records + annotations"], ["Proposed action", "task + initial state"],
         ["Adapters + inventory", "origin / units / clocks"], ["Task simulation", "controls / state / contacts"],
         ["Conditional checks", "six-state claim ledger"], ["Verification worker", "fresh run + control replay"],
         ["Receipts / trajectories / hashes / summaries"], ["External transaction service", "commit after acceptance"],
         ["External assets + data", "models / raw archives"]]
    )
    for i, x in enumerate((10, 268)):
        for row, y in enumerate((221, 152, 83)):
            box(d, x, y, 225, 48, words[row * 2 + i], zh)
            if row < 2:
                arrow(d, x + 112.5, y - 1, x + 112.5, y - 18)
    box(d, 10, 13, 225, 49, words[8], zh, dashed=True)
    box(d, 268, 13, 225, 49, words[7], zh, dashed=True)
    arrow(d, 380, 82, 380, 63)
    # Cross-path shared outputs are named in a side note, avoiding implied commits.
    label(d, WIDTH / 2, 3,
          "共同产物：回执、轨迹、来源记录；提交服务另行提供" if zh else
          "Shared outputs: evidence records; the commit service is supplied separately",
          zh, size=8, color=MUTED)
    return d


def verification(zh):
    d = Drawing(WIDTH, 285)
    steps = (
        [["锁定上下文并绑定候选"], ["检查身份与绑定"], ["执行仿真 / 保存证据"],
         ["检查必需项 / 独立重放"], ["验收授权 / 检查基准未变"]]
        if zh else
        [["Lock context and bind candidate"], ["Check identity and bindings"],
         ["Execute simulation / save evidence"], ["Check required claims / replay controls"],
         ["Authorize / confirm unchanged base"]]
    )
    for i, lines in enumerate(steps):
        y = 246 - i * 43
        box(d, 15, y, 270, 32, lines, zh)
        if i < 4:
            arrow(d, 150, y - 1, 150, y - 10)
    box(d, 330, 196, 161, 63,
        ["未评估", "必需证据不足或不一致"] if zh else
        ["NOT EVALUATED", "missing / inconsistent evidence"], zh, color=AMBER)
    box(d, 330, 92, 161, 63,
        ["拒绝", "可信评估发现违规"] if zh else
        ["REJECTED", "witnessed evaluated violation"], zh, color=RED)
    arrow(d, 285, 219, 329, 219, AMBER)
    arrow(d, 285, 133, 329, 133, RED)
    box(d, 15, 13, 270, 36,
        ["接受：推进已接受修订"] if zh else ["ACCEPTED: advance the accepted revision"],
        zh, color=GREEN)
    arrow(d, 150, 74, 150, 50, GREEN)
    label(d, 410, 38, "全部结果保留证据" if zh else "Retain all outcome records", zh, size=9)
    return d


def data_views(zh):
    d = Drawing(WIDTH, 243)
    label(d, 12, 230, "控制与积分状态（示意）" if zh else "Applied control and integration state (schematic)",
          zh, anchor="start", size=10)
    for i, x in enumerate((50, 210, 370)):
        d.add(Circle(x, 192, 4, fillColor=BLUE, strokeColor=None))
        label(d, x, 209, ["s0", "s1", "s2"][i], False)
        label(d, x, 171, ["t = 0", "t = 2 ms", "t = 4 ms"][i], False, size=8)
        if i < 2:
            arrow(d, x + 7, 192, x + 152, 192)
            label(d, x + 80, 199, "u" + str(i), False)
    label(d, 430, 187, "... sN", False)
    label(d, WIDTH / 2, 148,
          "N 个控制量对应 N+1 个状态；控制量与参考值分开" if zh else
          "N applied controls; N+1 states; actuator controls differ from references", zh, size=9)
    box(d, 115, 92, 270, 37,
        ["完整证据记录"] if zh else ["Full evidence record"], zh)
    for x in (80, 250, 420):
        arrow(d, 250, 91, x, 63)
    terms = ([["执行主体观测", "仅当前与历史输入"], ["标签与掩码", "未知不填为零"],
              ["审计与重放", "完整状态与控制"]]
             if zh else [["Actor observations", "present / past inputs only"],
                         ["Labels + masks", "unknown is not zero"],
                         ["Audit + replay", "full states + controls"]])
    for x, text in zip((5, 175, 345), terms):
        box(d, x, 12, 150, 50, text, zh)
    return d


def routes(zh):
    d = Drawing(WIDTH, 150)
    d.add(Rect(8, 18, WIDTH - 16, 119, fillColor=PALE, strokeColor=LINE))
    d.add(Rect(226, 32, 48, 60, fillColor=colors.HexColor("#DCC3BE"), strokeColor=RED))
    label(d, 250, 60, "障碍" if zh else "Obstacle", zh, size=8)
    for x, text in ((46, "起点" if zh else "Start"), (454, "目标" if zh else "Goal")):
        d.add(Circle(x, 46, 4, fillColor=BLUE, strokeColor=None))
        label(d, x, 27, text, zh, size=8)
    d.add(Line(53, 46, 222, 46, strokeColor=RED, strokeDashArray=[4, 3]))
    d.add(Line(278, 46, 447, 46, strokeColor=RED, strokeDashArray=[4, 3]))
    for a, b in (((50, 50), (116, 113)), ((116, 113), (384, 113)), ((384, 113), (450, 50))):
        arrow(d, *a, *b, color=BLUE)
    label(d, 250, 120, "抬高候选" if zh else "Raised candidate", zh, size=8)
    label(d, 130, 55, "直接候选" if zh else "Direct candidate", zh, size=8, color=RED)
    label(d, WIDTH / 2, 3, "概念示意，非实测轨迹" if zh else "Conceptual drawing, not a measured trajectory", zh, size=8, color=MUTED)
    return d


def counts(zh):
    d = Drawing(WIDTH, 158)
    multi = load("multi-robot-summary.json")
    tx = load("transaction-summary.json")
    teacher = load("teacher-data-summary.json")
    rows = [
        ("机械臂轨迹" if zh else "Arm trajectories", multi["status_counts"]["success"],
         multi["status_counts"]["rejected"]),
        ("事务候选" if zh else "Service candidates", tx["counts"]["accepted"], tx["counts"]["rejected"]),
        ("教师回合" if zh else "Teacher episodes", teacher["accepted"], teacher["rejected"]),
    ]
    for i, (name, accepted, rejected) in enumerate(rows):
        y = 113 - i * 35
        label(d, 4, y + 6, name, zh, anchor="start", size=8.5)
        left, width = 134, 287
        fraction = accepted / (accepted + rejected)
        for x, w, c, value in ((left, width * fraction, GREEN, accepted),
                               (left + width * fraction, width * (1 - fraction), RED, rejected)):
            d.add(Rect(x, y, w, 21, fillColor=c, strokeColor=None))
            label(d, x + w / 2, y + 7, str(value), False, color=colors.white, size=9)
        label(d, 466, y + 6, "n=" + str(accepted + rejected), False, size=9)
    label(d, 5, 145, "绿：成功/接受；红：拒绝；分母各自独立" if zh else
          "Green: successful/accepted; red: rejected; denominators are separate", zh, anchor="start", size=8)
    label(d, 5, 16,
          f"另有 {tx['counts']['not_evaluated']} 个未评估服务事件，未计入候选条形。" if zh else
          f"Plus {tx['counts']['not_evaluated']} unevaluated service events, outside the candidate bar.",
          zh, anchor="start", size=8, color=AMBER)
    return d


def audio(zh):
    d = Drawing(WIDTH, 283)
    terms = (
        [["人工编写的双手控制"], ["键程与手指接触"], ["检测音符事件"],
         ["乐谱评估", "核对谱段"], ["音频与 MIDI", "按事件合成"]]
        if zh else
        [["Authored two-hand control"], ["Key travel + finger contact"], ["Detected note events"],
         ["Score evaluation", "check score slots"], ["Audio + MIDI", "synthesize from events"]]
    )
    for i in range(3):
        box(d, 124, 247 - i * 44, 254, 31, terms[i], zh)
        if i < 2:
            arrow(d, 251, 246 - i * 44, 251, 235 - i * 44)
    for x, term in ((26, terms[3]), (285, terms[4])):
        box(d, x, 97, 190, 45, term, zh)
        arrow(d, 251, 158, x + 95, 143)
    rows = load("contact-audio-results.json")["rows"]
    for x, row in zip((26, 285), rows):
        m = row["metrics"]
        normal = row["variant"] == "nominal"
        title = ("正常 / 接受" if normal else "省略左手 / 拒绝") if zh else (
            "Nominal / accepted" if normal else "Omit left / rejected")
        text = (f"{m['actual_notes']} 音符；谱段 {m['passed_slots']}/{m['total_slots']}" if zh else
                f"{m['actual_notes']} notes; slots {m['passed_slots']}/{m['total_slots']}")
        box(d, x, 21, 190, 51, [title, text], zh, color=GREEN if normal else RED)
    label(d, WIDTH / 2, 4, "两次试验均为 31.65 秒；逐次报告结果" if zh else
          "31.65 seconds per trial; outcomes reported individually", zh, size=8, color=MUTED)
    return d


DRAWINGS = {
    "01-system": system, "02-verification": verification, "03-data": data_views,
    "04-routes": routes, "05-counts": counts, "06-audio": audio,
}


def build(lang):
    zh = lang == "zh"
    body_font = "WorldLedgerCJK" if zh else "Helvetica"
    heading_font = body_font if zh else "Helvetica-Bold"
    styles = {
        "title": ParagraphStyle("title", fontName=heading_font, fontSize=23, leading=29,
                                textColor=INK, spaceAfter=15),
        "h2": ParagraphStyle("h2", fontName=heading_font, fontSize=14.5, leading=20,
                             textColor=INK, spaceAfter=11, wordWrap="CJK" if zh else None),
        "h3": ParagraphStyle("h3", fontName=heading_font, fontSize=10.8, leading=15,
                             spaceBefore=8, spaceAfter=6, textColor=INK),
        "body": ParagraphStyle("body", fontName=body_font, fontSize=10 if zh else 9.5,
                               leading=15.5 if zh else 13.3, spaceAfter=8, alignment=TA_LEFT,
                               wordWrap="CJK" if zh else None),
        "caption": ParagraphStyle("caption", fontName=body_font, fontSize=8.3,
                                  leading=12, spaceAfter=12, textColor=MUTED,
                                  wordWrap="CJK" if zh else None),
        "equation": ParagraphStyle("equation", fontName="Courier", fontSize=8,
                                   leading=12, spaceAfter=9),
    }
    drawings = {}
    for name, fn in DRAWINGS.items():
        drawing = fn(zh)
        filename = name + "-" + lang + ".svg"
        renderSVG.drawToFile(drawing, str(FIGURES / filename))
        # SVGs use platform fallback families; the PDF embeds the supplied font.
        svg_path = FIGURES / filename
        svg = svg_path.read_text()
        svg_path.write_text(svg.replace("WorldLedgerCJK", "Arial Unicode MS, Noto Sans CJK SC, sans-serif"))
        drawings[filename] = drawing
    source = REPORT / ("technical-report-zh.md" if zh else "technical-report.md")
    target = REPORT / ("WorldLedger-technical-report-zh.pdf" if zh else "WorldLedger-technical-report.pdf")
    story = []
    for block in source.read_text().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block == "<!-- pagebreak -->":
            story.append(PageBreak())
            continue
        match = re.fullmatch(r"!\[(.*?)\]\(figures/(.*?)\)", block, re.S)
        if match:
            story.append(drawings[match[2]])
            story.append(Paragraph(html.escape(match[1]), styles["caption"]))
            continue
        if block.startswith("[[equation]] "):
            story.append(Preformatted(block.removeprefix("[[equation]] "), styles["equation"]))
            continue
        style, text = "body", block.replace("\n", " ")
        for prefix, name in (("### ", "h3"), ("## ", "h2"), ("# ", "title")):
            if block.startswith(prefix):
                style, text = name, block[len(prefix):]
                break
        story.append(Paragraph(html.escape(text), styles[style]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(46, 43, A4[0] - 46, 43)
        canvas.setFont(body_font, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(46, 30, "WORLDLEDGER | " + ("图文候选版 0.2" if zh else "ILLUSTRATED RC 0.2"))
        canvas.drawRightString(A4[0] - 46, 30, f"2026-09-23 | {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(target), pagesize=A4, rightMargin=46, leftMargin=46,
                            topMargin=43, bottomMargin=58,
                            title="WorldLedger: A Verification-Centered Architecture for Embodied Agents",
                            author="WorldLedger project",
                            subject="Illustrated release candidate 0.2; no DOI registered")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return {"manuscript": source.name, "pdf": target.name,
            "manuscript_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "pdf_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "figures": list(drawings), "explicit_sections": source.read_text().count("<!-- pagebreak -->") + 1}


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    manifest = {"version": "0.2", "date": "2026-09-23",
                "fresh_simulation_rerun": False,
                "figure_data": ["examples/multi-robot-summary.json", "examples/transaction-summary.json",
                                "examples/teacher-data-summary.json", "examples/contact-audio-results.json"],
                "builds": [build("en"), build("zh")]}
    (REPORT / "build-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
